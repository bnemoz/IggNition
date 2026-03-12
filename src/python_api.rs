use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::batch::{run_batch, run_batch_with_fallback_warning, BatchConfig, BatchInput};
use crate::core::types::{BatchResult, ChainType};

fn chain_from_str(s: &str) -> Option<ChainType> {
    match s.to_uppercase().as_str() {
        "H" | "IGH" | "HEAVY" => Some(ChainType::Heavy),
        "K" | "IGK" | "KAPPA" => Some(ChainType::Kappa),
        "L" | "IGL" | "LAMBDA" => Some(ChainType::Lambda),
        _ => None,
    }
}

/// Convert a `BatchResult` into a `(results_dict, errors_dict)` pair.
///
/// Both dicts use parallel lists (columnar format) that the Python side
/// can pass directly to `polars.DataFrame(...)`.
fn result_to_py(py: Python, result: BatchResult) -> PyResult<(PyObject, PyObject)> {
    let capacity: usize = result.results.iter().map(|r| r.positions.len()).sum();

    let mut seq_ids: Vec<u32> = Vec::with_capacity(capacity);
    let mut chains: Vec<String> = Vec::with_capacity(capacity);
    let mut nt_positions: Vec<u32> = Vec::with_capacity(capacity);
    let mut aho_positions: Vec<u32> = Vec::with_capacity(capacity);
    let mut codon_positions: Vec<u32> = Vec::with_capacity(capacity);
    // Pack nucleotides/amino_acids as raw bytes — Python converts with bytes.decode()
    let mut nucleotides: Vec<u8> = Vec::with_capacity(capacity);
    let mut amino_acids: Vec<u8> = Vec::with_capacity(capacity);

    for r in &result.results {
        let cs = r.chain.as_str().to_string();
        for pos in &r.positions {
            seq_ids.push(r.sequence_id);
            chains.push(cs.clone());
            nt_positions.push(pos.nt_position as u32);
            aho_positions.push(pos.aho_position as u32);
            codon_positions.push(pos.codon_position as u32);
            nucleotides.push(pos.nucleotide);
            amino_acids.push(pos.amino_acid);
        }
    }

    let res = PyDict::new(py);
    res.set_item("sequence_id", seq_ids)?;
    res.set_item("chain", chains)?;
    res.set_item("nt_position", nt_positions)?;
    res.set_item("aho_position", aho_positions)?;
    res.set_item("codon_position", codon_positions)?;
    // Send as bytes; Python side does `list(nt_bytes.decode('ascii'))`
    res.set_item("nucleotide", pyo3::types::PyBytes::new(py, &nucleotides))?;
    res.set_item("amino_acid", pyo3::types::PyBytes::new(py, &amino_acids))?;

    let errs = PyDict::new(py);
    let err_seq_ids: Vec<u32> = result.errors.iter().map(|e| e.sequence_id).collect();
    let err_chains: Vec<String> = result.errors.iter().map(|e| e.chain.as_str().to_string()).collect();
    let err_msgs: Vec<String> = result.errors.iter().map(|e| e.message.clone()).collect();
    errs.set_item("sequence_id", err_seq_ids)?;
    errs.set_item("chain", err_chains)?;
    errs.set_item("error", err_msgs)?;

    Ok((res.into(), errs.into()))
}

/// Run Aho numbering on a batch of sequences.
///
/// All four lists must have the same length.
///
/// Args:
///   sequence_ids: 0-based row indices (u32).
///   nt_seqs: Nucleotide sequences (str or None → treated as empty, collected as error).
///   aa_seqs: Amino acid sequences (str or None → fallback auto-translate mode).
///   chains: Chain codes ("H"/"IGH", "K"/"IGK", "L"/"IGL", or None → auto-detect).
///   num_threads: Rayon worker threads (None → all cores).
///
/// Returns:
///   (results_dict, errors_dict): dicts of parallel lists.
///     results_dict keys: sequence_id, chain, nt_position, aho_position,
///                        codon_position, nucleotide (bytes), amino_acid (bytes)
///     errors_dict  keys: sequence_id, chain, error
#[pyfunction]
#[pyo3(signature = (sequence_ids, nt_seqs, aa_seqs, chains, num_threads=None))]
pub fn _run_batch(
    py: Python,
    sequence_ids: Vec<u32>,
    nt_seqs: Vec<Option<String>>,
    aa_seqs: Vec<Option<String>>,
    chains: Vec<Option<String>>,
    num_threads: Option<usize>,
) -> PyResult<(PyObject, PyObject)> {
    let n = sequence_ids.len();
    if nt_seqs.len() != n || aa_seqs.len() != n || chains.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "All input lists must have the same length \
             (sequence_ids={n}, nt_seqs={}, aa_seqs={}, chains={})",
            nt_seqs.len(),
            aa_seqs.len(),
            chains.len(),
        )));
    }

    let inputs: Vec<BatchInput> = (0..n)
        .map(|i| {
            BatchInput::new(
                sequence_ids[i],
                nt_seqs[i].as_deref().unwrap_or("").as_bytes().to_vec(),
                aa_seqs[i].as_deref().map(|s| s.as_bytes().to_vec()),
                chains[i].as_deref().and_then(chain_from_str),
            )
        })
        .collect();

    let config = BatchConfig { num_threads, ..Default::default() };

    let batch_result = py.allow_threads(|| {
        run_batch_with_fallback_warning::<fn(usize)>(&inputs, &config, None)
    });

    result_to_py(py, batch_result)
}

/// Run Aho numbering on sequences from a FASTA file.
///
/// Args:
///   path: Path to the FASTA file.
///   paired_nt_aa: If True, alternating records are NT/AA pairs.
///   num_threads: Rayon worker threads (None → all cores).
///
/// Returns:
///   (results_dict, errors_dict) — same format as `_run_batch`.
#[pyfunction]
#[pyo3(signature = (path, paired_nt_aa=false, num_threads=None))]
pub fn _run_fasta(
    py: Python,
    path: &str,
    paired_nt_aa: bool,
    num_threads: Option<usize>,
) -> PyResult<(PyObject, PyObject)> {
    use std::path::Path;

    use crate::io::fasta::{read_fasta_file, FastaReaderConfig};

    let fasta_config = FastaReaderConfig { paired_nt_aa };
    let inputs = read_fasta_file(Path::new(path), &fasta_config)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

    let config = BatchConfig { num_threads, ..Default::default() };

    let batch_result = py.allow_threads(|| {
        run_batch::<fn(usize)>(&inputs, &config, None)
    });

    result_to_py(py, batch_result)
}

/// Run the `ignition` CLI using Python's `sys.argv`.
///
/// Returns the process exit code (0 = success, 1 = error).
/// Does **not** call `std::process::exit` — the caller is responsible.
#[pyfunction]
pub fn _cli_main(py: Python) -> PyResult<i32> {
    // Read sys.argv (not std::env::args) so Python console-script wrappers work.
    let sys = py.import("sys")?;
    let argv: Vec<String> = sys.getattr("argv")?.extract()?;

    let code = py.allow_threads(|| {
        match crate::cli_runner::run_cli(argv) {
            Ok(()) => 0,
            Err(e) => {
                let msg = e.to_string();
                if !msg.contains("clap") && !msg.contains("Usage:") {
                    eprintln!("ignition: error: {e}");
                }
                1
            }
        }
    });
    Ok(code)
}

/// Register all Python-visible functions into the module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_run_batch, m)?)?;
    m.add_function(wrap_pyfunction!(_run_fasta, m)?)?;
    m.add_function(wrap_pyfunction!(_cli_main, m)?)?;
    Ok(())
}
