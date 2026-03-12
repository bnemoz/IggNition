//! CLI logic for the `iggnition run` command.
//!
//! Shared between the standalone Rust binary (`src/cli.rs`) and the PyO3
//! `_cli_main` entry point (`src/python_api.rs`).

use std::io;
use std::path::{Path, PathBuf};

use clap::{Args, Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};

use crate::batch::{run_batch, run_batch_with_fallback_warning, BatchConfig};
use crate::error::IgnitionError;
use crate::io::output::{write_results, OutputFormat};
use crate::io::{detect_format, InputFormat};

// ─── Clap argument types ───────────────────────────────────────────────────────

#[derive(Parser)]
#[command(
    name = "iggnition",
    version,
    about = "Ultra-fast antibody variable domain Aho numbering",
    long_about = None,
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Number sequences in a FASTA, TSV, CSV, or Parquet file.
    Run(RunArgs),
}

#[derive(Args)]
struct RunArgs {
    /// Input file (.fasta/.fa, .tsv/.csv, .parquet)
    input: PathBuf,

    /// Output file (default: stdout as TSV).
    /// Extension determines format: .parquet → Parquet, others → TSV.
    output: Option<PathBuf>,

    /// Nucleotide sequence column name (TSV/Parquet, generic mode)
    #[arg(long, default_value = "sequence")]
    nt_col: String,

    /// Amino acid sequence column name (TSV/Parquet, generic mode).
    /// Pass an empty string to disable.
    #[arg(long, default_value = "sequence_aa")]
    aa_col: String,

    /// Locus/chain column name (e.g. `locus` → `IGH`/`IGK`/`IGL`)
    #[arg(long, default_value = "locus")]
    locus_col: String,

    /// Paired mode: each row has both a heavy and a light chain sequence
    #[arg(long)]
    paired: bool,

    /// Heavy-chain NT column (paired mode)
    #[arg(long, default_value = "sequence:0")]
    nt_col_heavy: String,

    /// Heavy-chain AA column (paired mode)
    #[arg(long, default_value = "sequence_aa:0")]
    aa_col_heavy: String,

    /// Light-chain NT column (paired mode)
    #[arg(long, default_value = "sequence:1")]
    nt_col_light: String,

    /// Light-chain AA column (paired mode)
    #[arg(long, default_value = "sequence_aa:1")]
    aa_col_light: String,

    /// Fallback mode: no AA sequence supplied, auto-detect reading frame
    #[arg(long)]
    no_aa: bool,

    /// Output one row per codon (Aho position) instead of per nucleotide
    #[arg(long)]
    per_codon: bool,

    /// Output wide format: one row per sequence with positional columns
    #[arg(long)]
    wide: bool,

    /// Number of Rayon worker threads (default: all cores)
    #[arg(long)]
    threads: Option<usize>,

    /// Show progress bar and informational messages
    #[arg(long, short = 'v')]
    verbose: bool,
}

// ─── Public entry point ────────────────────────────────────────────────────────

/// Parse `args` and run the CLI command.
///
/// Does **not** call `std::process::exit` — errors are returned.
pub fn run_cli(args: Vec<String>) -> Result<(), IgnitionError> {
    let cli = match Cli::try_parse_from(args) {
        Ok(c) => c,
        Err(e) => {
            // --help and --version "succeed" — print and return Ok so callers exit 0.
            if matches!(
                e.kind(),
                clap::error::ErrorKind::DisplayHelp | clap::error::ErrorKind::DisplayVersion
            ) {
                let _ = e.print();
                return Ok(());
            }
            let _ = e.print();
            return Err(IgnitionError::Io(e.to_string()));
        }
    };

    match cli.command {
        Commands::Run(a) => run_command(a),
    }
}

// ─── `run` subcommand ──────────────────────────────────────────────────────────

fn run_command(args: RunArgs) -> Result<(), IgnitionError> {
    let fmt = detect_format(&args.input).ok_or_else(|| {
        IgnitionError::Io(format!(
            "Unknown file format for '{}'. \
             Supported extensions: .fasta/.fa/.fna, .tsv/.txt, .csv, .parquet/.pq",
            args.input.display()
        ))
    })?;

    // ── Read inputs ────────────────────────────────────────────────────────────
    let inputs = read_inputs(&args, fmt)?;
    let total = inputs.len();

    if args.verbose {
        eprintln!(
            "iggnition: {} sequences loaded from {}",
            total,
            args.input.display()
        );
    }

    // ── Progress bar (only when writing to file and verbose) ─────────────────
    let pb = build_progress_bar(total, args.output.is_some() && args.verbose);

    let progress_fn = {
        let pb = pb.clone();
        move |done: usize| {
            if let Some(ref pb) = pb {
                pb.set_position(done as u64);
            }
        }
    };

    // ── Run batch ──────────────────────────────────────────────────────────────
    let config = BatchConfig {
        num_threads: args.threads,
        ..Default::default()
    };

    let result = if args.no_aa {
        run_batch_with_fallback_warning(&inputs, &config, Some(&progress_fn))
    } else {
        run_batch(&inputs, &config, Some(&progress_fn))
    };

    if let Some(ref pb) = pb {
        pb.finish_with_message(format!(
            "{} numbered, {} errors",
            result.results.len(),
            result.errors.len()
        ));
    }

    // ── Report errors to stderr ────────────────────────────────────────────────
    for err in &result.errors {
        eprintln!(
            "WARN  seq {:>6} [{}]: {}",
            err.sequence_id, err.chain, err.message
        );
    }

    if args.verbose {
        eprintln!(
            "iggnition: {} OK, {} errors",
            result.results.len(),
            result.errors.len()
        );
    }

    // ── Output format ──────────────────────────────────────────────────────────
    let out_fmt = match (args.per_codon, args.wide) {
        (true, _) => OutputFormat::PerCodon,
        (_, true) => OutputFormat::Wide,
        _ => OutputFormat::PerNucleotide,
    };

    // ── Write output ───────────────────────────────────────────────────────────
    match &args.output {
        None => {
            let stdout = io::stdout();
            let mut w = io::BufWriter::new(stdout.lock());
            write_results(&mut w, &result.results, out_fmt)?;
        }
        Some(out_path) => {
            write_output(out_path, &result, out_fmt)?;
        }
    }

    Ok(())
}

// ─── Format-specific readers ───────────────────────────────────────────────────

fn read_inputs(
    args: &RunArgs,
    fmt: InputFormat,
) -> Result<Vec<crate::batch::BatchInput>, IgnitionError> {
    use crate::io::fasta::{read_fasta_file, FastaReaderConfig};
    use crate::io::tsv::{read_tsv_file, read_tsv_paired_file, TsvReaderConfig};

    match fmt {
        InputFormat::Fasta => {
            let config = FastaReaderConfig {
                paired_nt_aa: args.paired,
            };
            read_fasta_file(&args.input, &config)
        }

        InputFormat::Tsv | InputFormat::Csv => {
            let delimiter = if fmt == InputFormat::Csv { b',' } else { b'\t' };
            if args.paired {
                read_tsv_paired_file(
                    &args.input,
                    &args.nt_col_heavy,
                    &args.aa_col_heavy,
                    &args.nt_col_light,
                    &args.aa_col_light,
                    delimiter,
                )
            } else {
                let config = TsvReaderConfig {
                    nt_col: args.nt_col.clone(),
                    aa_col: if args.no_aa {
                        None
                    } else {
                        Some(args.aa_col.clone())
                    },
                    locus_col: Some(args.locus_col.clone()),
                    delimiter,
                };
                read_tsv_file(&args.input, &config)
            }
        }

        InputFormat::Parquet => {
            #[cfg(feature = "io_parquet")]
            {
                use crate::io::parquet::{read_parquet_file, ParquetConfig};
                let config = ParquetConfig {
                    nt_col: args.nt_col.clone(),
                    aa_col: if args.no_aa {
                        None
                    } else {
                        Some(args.aa_col.clone())
                    },
                    locus_col: Some(args.locus_col.clone()),
                };
                read_parquet_file(&args.input, &config)
            }
            #[cfg(not(feature = "io_parquet"))]
            Err(IgnitionError::Io(
                "Parquet input requires the `io_parquet` feature. \
                 Rebuild with: cargo build --features io_parquet"
                    .to_string(),
            ))
        }
    }
}

// ─── Output writer ─────────────────────────────────────────────────────────────

fn write_output(
    path: &Path,
    result: &crate::core::types::BatchResult,
    fmt: OutputFormat,
) -> Result<(), IgnitionError> {
    use crate::io::tsv::write_tsv_file;

    match detect_format(path) {
        Some(InputFormat::Parquet) => {
            #[cfg(feature = "io_parquet")]
            {
                use crate::io::parquet::{write_parquet_errors, write_parquet_file};
                write_parquet_file(path, &result.results)?;
                if !result.errors.is_empty() {
                    let err_path = path.with_extension("errors.parquet");
                    eprintln!("iggnition: writing errors to {}", err_path.display());
                    write_parquet_errors(&err_path, &result.errors)?;
                }
                Ok(())
            }
            #[cfg(not(feature = "io_parquet"))]
            Err(IgnitionError::Io(
                "Parquet output requires the `io_parquet` feature.".to_string(),
            ))
        }
        _ => write_tsv_file(path, &result.results, fmt),
    }
}

// ─── Progress bar helper ───────────────────────────────────────────────────────

fn build_progress_bar(total: usize, writing_to_file: bool) -> Option<ProgressBar> {
    if total < 500 || !writing_to_file {
        return None;
    }
    let pb = ProgressBar::new(total as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] \
             {pos}/{len} seqs ({per_sec}, eta {eta})",
        )
        .unwrap()
        .progress_chars("=>-"),
    );
    Some(pb)
}
