// These imports are only used by the stub functions below (non-io_parquet build).
#[cfg(not(feature = "io_parquet"))]
use std::path::Path;
#[cfg(not(feature = "io_parquet"))]
use crate::batch::BatchInput;
#[cfg(not(feature = "io_parquet"))]
use crate::core::types::NumberingResult;
#[cfg(not(feature = "io_parquet"))]
use crate::error::{IgnitionError, NumberingError};

/// Configuration for reading Parquet files.
#[derive(Debug, Clone)]
pub struct ParquetConfig {
    /// Column name containing the nucleotide sequence.
    pub nt_col: String,
    /// Column name containing the amino acid sequence (optional).
    pub aa_col: Option<String>,
    /// Column name containing the IMGT locus / chain code.
    pub locus_col: Option<String>,
}

impl Default for ParquetConfig {
    fn default() -> Self {
        Self {
            nt_col: "sequence".to_string(),
            aa_col: Some("sequence_aa".to_string()),
            locus_col: Some("locus".to_string()),
        }
    }
}

// ─── Feature-gated implementations ───────────────────────────────────────────

#[cfg(feature = "io_parquet")]
mod inner {
    use polars::prelude::*;
    use std::path::Path;

    use crate::batch::BatchInput;
    use crate::core::types::{ChainType, NumberingResult};
    use crate::error::{IgnitionError, NumberingError};

    use super::ParquetConfig;

    fn chain_from_locus(locus: &str) -> Option<ChainType> {
        match locus.to_uppercase().as_str() {
            "IGH" | "H" => Some(ChainType::Heavy),
            "IGK" | "K" => Some(ChainType::Kappa),
            "IGL" | "L" => Some(ChainType::Lambda),
            _ => None,
        }
    }

    pub fn read_parquet_file(
        path: &Path,
        config: &ParquetConfig,
    ) -> Result<Vec<BatchInput>, IgnitionError> {
        let mut file = std::fs::File::open(path)
            .map_err(|e| IgnitionError::Io(format!("{}: {}", path.display(), e)))?;

        let df = ParquetReader::new(&mut file)
            .finish()
            .map_err(|e| IgnitionError::Io(e.to_string()))?;

        df_to_inputs(df, config)
    }

    pub fn df_to_inputs(
        df: DataFrame,
        config: &ParquetConfig,
    ) -> Result<Vec<BatchInput>, IgnitionError> {
        let nt_col = df
            .column(&config.nt_col)
            .map_err(|_| IgnitionError::Io(format!("Column '{}' not found", config.nt_col)))?;
        // polars 0.46: Column::as_series() returns Option<&Series>; use .str()
        // directly on Column instead.
        let nt_strs = nt_col
            .str()
            .map_err(|e: PolarsError| IgnitionError::Io(e.to_string()))?;

        let aa_strs_opt: Option<StringChunked> = config
            .aa_col
            .as_ref()
            .and_then(|col| df.column(col).ok())
            .and_then(|c| c.str().ok().cloned());

        let locus_strs_opt: Option<StringChunked> = config
            .locus_col
            .as_ref()
            .and_then(|col| df.column(col).ok())
            .and_then(|c| c.str().ok().cloned());

        let n = df.height();
        let mut inputs = Vec::with_capacity(n);

        for i in 0..n {
            let nt_seq = nt_strs
                .get(i)
                .unwrap_or("")
                .trim()
                .as_bytes()
                .to_vec();

            let aa_seq = aa_strs_opt
                .as_ref()
                .and_then(|s| s.get(i))
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(|s| s.as_bytes().to_vec());

            let chain = locus_strs_opt
                .as_ref()
                .and_then(|s| s.get(i))
                .and_then(|s| chain_from_locus(s.trim()));

            inputs.push(BatchInput::new(i as u32, nt_seq, aa_seq, chain));
        }

        Ok(inputs)
    }

    /// Write numbering results to a Parquet file (per-nucleotide schema).
    pub fn write_parquet_file(
        path: &Path,
        results: &[NumberingResult],
    ) -> Result<(), IgnitionError> {
        // Flatten to columnar vecs
        let capacity: usize = results.iter().map(|r| r.positions.len()).sum();
        let mut seq_ids: Vec<u32> = Vec::with_capacity(capacity);
        let mut chains: Vec<&str> = Vec::with_capacity(capacity);
        let mut nt_positions: Vec<u32> = Vec::with_capacity(capacity);
        let mut aho_positions: Vec<u32> = Vec::with_capacity(capacity);
        let mut codon_positions: Vec<u32> = Vec::with_capacity(capacity);
        let mut nucleotides: Vec<String> = Vec::with_capacity(capacity);
        let mut amino_acids: Vec<String> = Vec::with_capacity(capacity);

        for result in results {
            let chain_str = result.chain.as_str();
            for pos in &result.positions {
                seq_ids.push(result.sequence_id);
                chains.push(chain_str);
                nt_positions.push(pos.nt_position as u32);
                aho_positions.push(pos.aho_position as u32);
                codon_positions.push(pos.codon_position as u32);
                nucleotides.push((pos.nucleotide as char).to_string());
                amino_acids.push((pos.amino_acid as char).to_string());
            }
        }

        let mut df = df! {
            "sequence_id"   => seq_ids.as_slice(),
            "chain"         => chains.as_slice(),
            "nt_position"   => nt_positions.as_slice(),
            "aho_position"  => aho_positions.as_slice(),
            "codon_position"=> codon_positions.as_slice(),
            "nucleotide"    => nucleotides.iter().map(|s| s.as_str()).collect::<Vec<_>>().as_slice(),
            "amino_acid"    => amino_acids.iter().map(|s| s.as_str()).collect::<Vec<_>>().as_slice(),
        }
        .map_err(|e| IgnitionError::Io(e.to_string()))?;

        let mut file = std::fs::File::create(path)
            .map_err(|e| IgnitionError::Io(format!("{}: {}", path.display(), e)))?;

        ParquetWriter::new(&mut file)
            .finish(&mut df)
            .map_err(|e| IgnitionError::Io(e.to_string()))?;

        Ok(())
    }

    /// Write numbering errors to a Parquet file.
    pub fn write_parquet_errors(
        path: &Path,
        errors: &[NumberingError],
    ) -> Result<(), IgnitionError> {
        let seq_ids: Vec<u32> = errors.iter().map(|e| e.sequence_id).collect();
        let chains: Vec<&str> = errors.iter().map(|e| e.chain.as_str()).collect();
        let messages: Vec<&str> = errors.iter().map(|e| e.message.as_str()).collect();

        let mut df = df! {
            "sequence_id" => seq_ids.as_slice(),
            "chain"       => chains.as_slice(),
            "error"       => messages.as_slice(),
        }
        .map_err(|e| IgnitionError::Io(e.to_string()))?;

        let mut file = std::fs::File::create(path)
            .map_err(|e| IgnitionError::Io(format!("{}: {}", path.display(), e)))?;

        ParquetWriter::new(&mut file)
            .finish(&mut df)
            .map_err(|e| IgnitionError::Io(e.to_string()))?;

        Ok(())
    }
}

// ─── Public surface (feature-gated) ──────────────────────────────────────────

#[cfg(feature = "io_parquet")]
pub use inner::{df_to_inputs, read_parquet_file, write_parquet_errors, write_parquet_file};

/// Stub: feature `io_parquet` not enabled.
#[cfg(not(feature = "io_parquet"))]
pub fn read_parquet_file(
    path: &Path,
    _config: &ParquetConfig,
) -> Result<Vec<BatchInput>, IgnitionError> {
    Err(IgnitionError::Io(format!(
        "{}: Parquet support not compiled in (enable the `io_parquet` feature)",
        path.display()
    )))
}

/// Stub: feature `io_parquet` not enabled.
#[cfg(not(feature = "io_parquet"))]
pub fn write_parquet_file(
    path: &Path,
    _results: &[NumberingResult],
) -> Result<(), IgnitionError> {
    Err(IgnitionError::Io(format!(
        "{}: Parquet support not compiled in (enable the `io_parquet` feature)",
        path.display()
    )))
}

/// Stub: feature `io_parquet` not enabled.
#[cfg(not(feature = "io_parquet"))]
pub fn write_parquet_errors(
    path: &Path,
    _errors: &[NumberingError],
) -> Result<(), IgnitionError> {
    Err(IgnitionError::Io(format!(
        "{}: Parquet support not compiled in (enable the `io_parquet` feature)",
        path.display()
    )))
}
