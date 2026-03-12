use std::io::Read;
use std::path::Path;

use crate::batch::BatchInput;
use crate::core::types::{ChainType, NumberingResult};
use crate::error::IgnitionError;
use crate::io::output::{write_results, OutputFormat};

/// Configuration for reading TSV/CSV files.
#[derive(Debug, Clone)]
pub struct TsvReaderConfig {
    /// Column name containing the nucleotide sequence.
    pub nt_col: String,
    /// Column name containing the amino acid sequence (optional).
    pub aa_col: Option<String>,
    /// Column name containing the IMGT locus / chain code (e.g. `IGH`, `IGK`, `IGL`).
    pub locus_col: Option<String>,
    /// Field delimiter — `b'\t'` for TSV, `b','` for CSV.
    pub delimiter: u8,
}

impl Default for TsvReaderConfig {
    fn default() -> Self {
        Self {
            nt_col: "sequence".to_string(),
            aa_col: Some("sequence_aa".to_string()),
            locus_col: Some("locus".to_string()),
            delimiter: b'\t',
        }
    }
}

fn chain_from_locus(locus: &str) -> Option<ChainType> {
    match locus.to_uppercase().as_str() {
        "IGH" | "H" => Some(ChainType::Heavy),
        "IGK" | "K" => Some(ChainType::Kappa),
        "IGL" | "L" => Some(ChainType::Lambda),
        _ => None,
    }
}

/// Read a TSV/CSV file and produce a `Vec<BatchInput>`.
///
/// Column names are configured via `TsvReaderConfig`. The `sequence_id` field
/// of each `BatchInput` is set to the 0-based row index.
pub fn read_tsv_file(
    path: &Path,
    config: &TsvReaderConfig,
) -> Result<Vec<BatchInput>, IgnitionError> {
    let file = std::fs::File::open(path)
        .map_err(|e| IgnitionError::Io(format!("{}: {}", path.display(), e)))?;
    read_tsv_reader(file, config)
}

pub fn read_tsv_reader<R: Read>(
    reader: R,
    config: &TsvReaderConfig,
) -> Result<Vec<BatchInput>, IgnitionError> {
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(config.delimiter)
        .flexible(true)
        .from_reader(reader);

    let headers = rdr
        .headers()
        .map_err(|e| IgnitionError::Io(e.to_string()))?
        .clone();

    let nt_idx = headers
        .iter()
        .position(|h| h == config.nt_col.as_str())
        .ok_or_else(|| {
            IgnitionError::Io(format!("Column '{}' not found in TSV/CSV", config.nt_col))
        })?;

    let aa_idx = config
        .aa_col
        .as_ref()
        .and_then(|col| headers.iter().position(|h| h == col.as_str()));

    let locus_idx = config
        .locus_col
        .as_ref()
        .and_then(|col| headers.iter().position(|h| h == col.as_str()));

    let mut inputs = Vec::new();
    for (row_idx, result) in rdr.records().enumerate() {
        let record = result.map_err(|e| IgnitionError::Io(e.to_string()))?;

        let nt_seq = record
            .get(nt_idx)
            .unwrap_or("")
            .trim()
            .as_bytes()
            .to_vec();

        let aa_seq = aa_idx
            .and_then(|i| record.get(i))
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| s.as_bytes().to_vec());

        let chain = locus_idx
            .and_then(|i| record.get(i))
            .and_then(|s| chain_from_locus(s.trim()));

        inputs.push(BatchInput::new(row_idx as u32, nt_seq, aa_seq, chain));
    }

    Ok(inputs)
}

/// Read a TSV/CSV file in paired mode — one row contains both a heavy-chain and
/// a light-chain sequence in separate columns.
///
/// Produces two `BatchInput` entries per row (heavy with `chain=H`, light with
/// `chain=None` for auto K/L detection). Rows with an empty NT field are skipped.
pub fn read_tsv_paired_file(
    path: &Path,
    nt_col_heavy: &str,
    aa_col_heavy: &str,
    nt_col_light: &str,
    aa_col_light: &str,
    delimiter: u8,
) -> Result<Vec<BatchInput>, IgnitionError> {
    let file = std::fs::File::open(path)
        .map_err(|e| IgnitionError::Io(format!("{}: {}", path.display(), e)))?;
    read_tsv_paired_reader(
        file, nt_col_heavy, aa_col_heavy, nt_col_light, aa_col_light, delimiter,
    )
}

pub fn read_tsv_paired_reader<R: Read>(
    reader: R,
    nt_col_heavy: &str,
    aa_col_heavy: &str,
    nt_col_light: &str,
    aa_col_light: &str,
    delimiter: u8,
) -> Result<Vec<BatchInput>, IgnitionError> {
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(delimiter)
        .flexible(true)
        .from_reader(reader);

    let headers = rdr
        .headers()
        .map_err(|e| IgnitionError::Io(e.to_string()))?
        .clone();

    let nt_h_idx = headers
        .iter()
        .position(|h| h == nt_col_heavy)
        .ok_or_else(|| IgnitionError::Io(format!("Column '{}' not found", nt_col_heavy)))?;
    let aa_h_idx = headers.iter().position(|h| h == aa_col_heavy);
    let nt_l_idx = headers.iter().position(|h| h == nt_col_light);
    let aa_l_idx = headers.iter().position(|h| h == aa_col_light);

    let mut inputs = Vec::new();
    for (row_idx, result) in rdr.records().enumerate() {
        let record = result.map_err(|e| IgnitionError::Io(e.to_string()))?;

        // Heavy entry
        let heavy_nt = record.get(nt_h_idx).unwrap_or("").trim().as_bytes().to_vec();
        if !heavy_nt.is_empty() {
            let heavy_aa = aa_h_idx
                .and_then(|i| record.get(i))
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(|s| s.as_bytes().to_vec());
            inputs.push(BatchInput::new(
                row_idx as u32,
                heavy_nt,
                heavy_aa,
                Some(crate::core::types::ChainType::Heavy),
            ));
        }

        // Light entry (chain auto-detected → Kappa or Lambda)
        if let Some(nt_l) = nt_l_idx {
            let light_nt = record.get(nt_l).unwrap_or("").trim().as_bytes().to_vec();
            if !light_nt.is_empty() {
                let light_aa = aa_l_idx
                    .and_then(|i| record.get(i))
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(|s| s.as_bytes().to_vec());
                inputs.push(BatchInput::new(row_idx as u32, light_nt, light_aa, None));
            }
        }
    }

    Ok(inputs)
}

/// Write numbering results to a TSV/CSV file.
///
/// The output format follows the same schema as stdout: per-nucleotide by
/// default. Uses `\t` as separator regardless of `config` — callers can choose
/// `OutputFormat` independently of how the file was read.
pub fn write_tsv_file(
    path: &Path,
    results: &[NumberingResult],
    format: OutputFormat,
) -> Result<(), IgnitionError> {
    let file = std::fs::File::create(path)
        .map_err(|e| IgnitionError::Io(format!("{}: {}", path.display(), e)))?;
    let mut w = std::io::BufWriter::new(file);
    write_results(&mut w, results, format)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tsv(with_aa: bool) -> String {
        let seq1 = "CAGGTGCAGCTGGTGCAG";
        let aa1 = "QVQLVQ";
        let seq2 = "ATGATGATG";
        if with_aa {
            format!(
                "sequence_id\tsequence\tsequence_aa\tlocus\nrow0\t{}\t{}\tIGH\nrow1\t{}\t\tIGK\n",
                seq1, aa1, seq2
            )
        } else {
            format!(
                "sequence_id\tsequence\tlocus\nrow0\t{}\tIGH\nrow1\t{}\tIGK\n",
                seq1, seq2
            )
        }
    }

    #[test]
    fn test_tsv_reader_with_aa() {
        let tsv = make_tsv(true);
        let config = TsvReaderConfig::default();
        let inputs = read_tsv_reader(tsv.as_bytes(), &config).unwrap();
        assert_eq!(inputs.len(), 2);
        assert_eq!(inputs[0].chain, Some(ChainType::Heavy));
        assert_eq!(inputs[1].chain, Some(ChainType::Kappa));
        assert!(inputs[0].aa_seq.is_some());
        assert_eq!(inputs[0].aa_seq.as_deref().unwrap(), b"QVQLVQ");
        // Empty AA field → None
        assert!(inputs[1].aa_seq.is_none());
    }

    #[test]
    fn test_tsv_reader_without_aa_col() {
        let tsv = make_tsv(false);
        let config = TsvReaderConfig { aa_col: None, ..TsvReaderConfig::default() };
        let inputs = read_tsv_reader(tsv.as_bytes(), &config).unwrap();
        assert_eq!(inputs.len(), 2);
        assert!(inputs[0].aa_seq.is_none());
    }

    #[test]
    fn test_tsv_reader_missing_nt_col() {
        let tsv = "other_col\nATG\n";
        let config = TsvReaderConfig::default();
        assert!(read_tsv_reader(tsv.as_bytes(), &config).is_err());
    }

    #[test]
    fn test_tsv_reader_chain_detection() {
        let tsv = "sequence\tlocus\nATG\tIGH\nATG\tIGK\nATG\tIGL\nATG\tunknown\n";
        let config = TsvReaderConfig { aa_col: None, ..TsvReaderConfig::default() };
        let inputs = read_tsv_reader(tsv.as_bytes(), &config).unwrap();
        assert_eq!(inputs[0].chain, Some(ChainType::Heavy));
        assert_eq!(inputs[1].chain, Some(ChainType::Kappa));
        assert_eq!(inputs[2].chain, Some(ChainType::Lambda));
        assert_eq!(inputs[3].chain, None);
    }

    #[test]
    fn test_csv_reader() {
        let csv = "sequence,sequence_aa,locus\nATGCATG,M,IGH\n";
        let config = TsvReaderConfig { delimiter: b',', ..TsvReaderConfig::default() };
        let inputs = read_tsv_reader(csv.as_bytes(), &config).unwrap();
        assert_eq!(inputs.len(), 1);
        assert_eq!(inputs[0].chain, Some(ChainType::Heavy));
        assert_eq!(inputs[0].nt_seq, b"ATGCATG");
    }

    #[test]
    fn test_tsv_sequence_ids_are_row_indices() {
        let tsv = "sequence\nATG\nATG\nATG\n";
        let config = TsvReaderConfig {
            aa_col: None,
            locus_col: None,
            ..TsvReaderConfig::default()
        };
        let inputs = read_tsv_reader(tsv.as_bytes(), &config).unwrap();
        assert_eq!(inputs.len(), 3);
        assert_eq!(inputs[0].sequence_id, 0);
        assert_eq!(inputs[1].sequence_id, 1);
        assert_eq!(inputs[2].sequence_id, 2);
    }
}
