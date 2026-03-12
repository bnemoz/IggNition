use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use crate::batch::BatchInput;
use crate::core::types::ChainType;
use crate::error::IgnitionError;

/// Parse chain type from a FASTA header, or return `None` for auto-detection.
///
/// Recognises suffixes `_H`, `_K`, `_L`, `_HEAVY`, `_KAPPA`, `_LAMBDA`, `_VH`, `_VK`, `_VL`
/// and IMGT locus codes `IGH`, `IGK`, `IGL` anywhere in the header.
fn chain_from_header(header: &str) -> Option<ChainType> {
    let h = header.to_uppercase();
    // Suffix checks (more specific first)
    if h.ends_with("_HEAVY") || h.ends_with("_VH") || h.ends_with("_H") {
        return Some(ChainType::Heavy);
    }
    if h.ends_with("_KAPPA") || h.ends_with("_VK") || h.ends_with("_K") {
        return Some(ChainType::Kappa);
    }
    if h.ends_with("_LAMBDA") || h.ends_with("_VL") || h.ends_with("_L") {
        return Some(ChainType::Lambda);
    }
    // IMGT locus anywhere in header
    if h.contains("IGH") || h.contains("HEAVY") {
        return Some(ChainType::Heavy);
    }
    if h.contains("IGK") || h.contains("KAPPA") {
        return Some(ChainType::Kappa);
    }
    if h.contains("IGL") || h.contains("LAMBDA") {
        return Some(ChainType::Lambda);
    }
    None
}

/// Detect whether a sequence is amino acid (vs nucleotide).
///
/// A sequence is treated as AA if it contains letters beyond A/T/G/C/N/- (case-insensitive).
fn is_amino_acid(seq: &[u8]) -> bool {
    const NT_CHARS: &[u8] = b"ATGCNatgcn-";
    seq.iter().any(|&b| !NT_CHARS.contains(&b))
}

/// A raw FASTA record (header without `>`, raw sequence bytes)
struct FastaRecord {
    header: String,
    sequence: Vec<u8>,
}

/// Read all FASTA records from a reader.
fn parse_fasta<R: Read>(reader: R) -> Result<Vec<FastaRecord>, IgnitionError> {
    let buf = BufReader::new(reader);
    let mut records = Vec::new();
    let mut current_header: Option<String> = None;
    let mut current_seq: Vec<u8> = Vec::new();

    for line in buf.lines() {
        let line = line.map_err(|e| IgnitionError::Io(e.to_string()))?;
        let line = line.trim_end();
        if let Some(rest) = line.strip_prefix('>') {
            if let Some(header) = current_header.take() {
                records.push(FastaRecord { header, sequence: std::mem::take(&mut current_seq) });
            }
            current_header = Some(rest.to_string());
        } else if !line.is_empty() {
            current_seq.extend_from_slice(line.as_bytes());
        }
    }
    if let Some(header) = current_header {
        records.push(FastaRecord { header, sequence: current_seq });
    }
    Ok(records)
}

/// Configuration for reading FASTA files.
#[derive(Debug, Clone, Default)]
pub struct FastaReaderConfig {
    /// If true, paired NT+AA records are expected (alternating or by naming convention).
    /// If false, each record is treated as NT-only (AA auto-detected or absent).
    pub paired_nt_aa: bool,
}

/// Read a FASTA file and produce a Vec of `BatchInput`.
///
/// ## Supported formats
///
/// **Auto-paired mode** (`paired_nt_aa = true`):
/// - Alternating records: `>id_nt` / `>id_aa` (pairs must be consecutive)
/// - The AA record is identified by ending in `_aa`, `_AA`, `_prot`, or `_PROT`
///
/// **Single-record mode** (`paired_nt_aa = false`):
/// - Each record is one NT sequence (AA omitted → fallback mode)
/// - If the sequence content looks like amino acids, it is stored as AA with no NT
///
/// In both modes, chain type is parsed from the header.
pub fn read_fasta_file(
    path: &Path,
    config: &FastaReaderConfig,
) -> Result<Vec<BatchInput>, IgnitionError> {
    let file = std::fs::File::open(path)
        .map_err(|e| IgnitionError::Io(format!("{}: {}", path.display(), e)))?;
    read_fasta_reader(file, config)
}

pub fn read_fasta_reader<R: Read>(
    reader: R,
    config: &FastaReaderConfig,
) -> Result<Vec<BatchInput>, IgnitionError> {
    let records = parse_fasta(reader)?;
    if config.paired_nt_aa {
        paired_records_to_inputs(records)
    } else {
        single_records_to_inputs(records)
    }
}

fn is_aa_record_name(header: &str) -> bool {
    let h = header.to_uppercase();
    h.ends_with("_AA") || h.ends_with("_PROT") || h.ends_with("_PROTEIN")
}

fn paired_records_to_inputs(records: Vec<FastaRecord>) -> Result<Vec<BatchInput>, IgnitionError> {
    let mut inputs = Vec::new();
    let mut i = 0;
    let mut seq_id: u32 = 0;

    while i < records.len() {
        let rec = &records[i];
        if is_aa_record_name(&rec.header) {
            // Orphan AA record — skip
            i += 1;
            continue;
        }
        // Check if next record is the paired AA
        if i + 1 < records.len() && is_aa_record_name(&records[i + 1].header) {
            let nt_rec = rec;
            let aa_rec = &records[i + 1];
            let chain = chain_from_header(&nt_rec.header);
            inputs.push(BatchInput::new(
                seq_id,
                nt_rec.sequence.clone(),
                Some(aa_rec.sequence.clone()),
                chain,
            ));
            i += 2;
        } else {
            // NT-only
            let chain = chain_from_header(&rec.header);
            inputs.push(BatchInput::new(seq_id, rec.sequence.clone(), None, chain));
            i += 1;
        }
        seq_id += 1;
    }
    Ok(inputs)
}

fn single_records_to_inputs(records: Vec<FastaRecord>) -> Result<Vec<BatchInput>, IgnitionError> {
    records
        .into_iter()
        .enumerate()
        .map(|(idx, rec)| {
            let chain = chain_from_header(&rec.header);
            if is_amino_acid(&rec.sequence) {
                // Detected as AA-only (no NT) — unusual but handle gracefully
                Ok(BatchInput::new(idx as u32, vec![], Some(rec.sequence), chain))
            } else {
                Ok(BatchInput::new(idx as u32, rec.sequence, None, chain))
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEAVY_NT: &str = "CAGGTGCAGCTGGTGCAGTCTGGAGCTGAGGTGAAGAAGCCTGGGGCCTCAGTGAAGGTCTCCTGCAAGGCTTCTGGTTACACCTTTACCAGCTATGGTATCAGCTGGGTGCGACAGGCCCCTGGACAAGGGCTTGAGTGGATGGGGTGGATCAGCGCTTACAATGGTAACACAAACTATGCACAGAAGCTCCAGGGCAGAGTCACGATGACCACAGACACATCCACGAGCACAGCCTACATGGAGCTGAGGAGCCTGAGATCTGACGACACGGCCGTGTATTACTGTGCGAGA";
    const HEAVY_AA: &str = "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYGISWVRQAPGQGLEWMGWISAYNGNTNYAQKLQGRVTMTTDTSTSTAYMELRSLRSDDTAVYYCAR";

    fn make_nt_only_fasta() -> String {
        format!(">seq0_H\n{}\n>seq1_K\nATGATGATG\n", HEAVY_NT)
    }

    fn make_paired_fasta() -> String {
        format!(
            ">seq0_H\n{}\n>seq0_aa\n{}\n>seq1_K\nATGATGATG\n",
            HEAVY_NT, HEAVY_AA
        )
    }

    #[test]
    fn test_nt_only_fasta_chain_detection() {
        let fasta = make_nt_only_fasta();
        let inputs = read_fasta_reader(fasta.as_bytes(), &FastaReaderConfig::default()).unwrap();
        assert_eq!(inputs.len(), 2);
        assert_eq!(inputs[0].chain, Some(ChainType::Heavy));
        assert_eq!(inputs[1].chain, Some(ChainType::Kappa));
        assert!(inputs[0].aa_seq.is_none());
    }

    #[test]
    fn test_nt_only_fasta_sequence_content() {
        let fasta = make_nt_only_fasta();
        let inputs = read_fasta_reader(fasta.as_bytes(), &FastaReaderConfig::default()).unwrap();
        assert_eq!(inputs[0].nt_seq.as_slice(), HEAVY_NT.as_bytes());
    }

    #[test]
    fn test_paired_fasta_aa_pairing() {
        let fasta = make_paired_fasta();
        let config = FastaReaderConfig { paired_nt_aa: true };
        let inputs = read_fasta_reader(fasta.as_bytes(), &config).unwrap();
        assert_eq!(inputs.len(), 2);
        assert!(inputs[0].aa_seq.is_some());
        assert_eq!(inputs[0].aa_seq.as_deref().unwrap(), HEAVY_AA.as_bytes());
        assert!(inputs[1].aa_seq.is_none()); // no paired AA for seq1
    }

    #[test]
    fn test_chain_from_header() {
        assert_eq!(chain_from_header("seq_H"), Some(ChainType::Heavy));
        assert_eq!(chain_from_header("VHseq_VH"), Some(ChainType::Heavy));
        assert_eq!(chain_from_header("antibody_K"), Some(ChainType::Kappa));
        assert_eq!(chain_from_header("seq1_L"), Some(ChainType::Lambda));
        assert_eq!(chain_from_header("IGKV1-2*01"), Some(ChainType::Kappa));
        assert_eq!(chain_from_header("randomheader"), None);
    }

    #[test]
    fn test_is_amino_acid() {
        assert!(is_amino_acid(b"QVQLVQSGA")); // Q is not in NT_CHARS
        assert!(!is_amino_acid(b"ATGCATGC")); // pure NT
        assert!(!is_amino_acid(b"ATGCNNN")); // NT with N
    }

    #[test]
    fn test_multiline_fasta() {
        let fasta = ">seq0_H\nATGC\nATGC\nATGC\n";
        let inputs = read_fasta_reader(fasta.as_bytes(), &FastaReaderConfig::default()).unwrap();
        assert_eq!(inputs[0].nt_seq, b"ATGCATGCATGC");
    }
}
