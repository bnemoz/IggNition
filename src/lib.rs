pub mod batch;
pub mod cli_runner;
pub mod core;
pub mod error;
pub mod io;

#[cfg(feature = "python")]
mod python_api;

pub use batch::{run_batch, run_batch_with_fallback_warning, BatchConfig, BatchInput};
pub use core::types::{BatchResult, ChainType, NumberingResult, NtPosition};
pub use error::IgnitionError;

use core::aho::number_sequence;
use core::frame::{resolve_with_aa, resolve_without_aa};

/// Number a single antibody chain sequence.
///
/// # Arguments
/// * `sequence_id` - External identifier for this sequence
/// * `nt_seq`      - Nucleotide sequence (ASCII, A/T/G/C)
/// * `aa_seq` - Amino acid sequence. If `None`, the reading frame is
///   auto-detected (fallback mode — emits a warning).
/// * `chain`       - Chain type (Heavy, Kappa, Lambda)
///
/// # Returns
/// A `NumberingResult` with one `NtPosition` per nucleotide position in the
/// full Aho frame (max_aho × 3 entries), or an `IgnitionError`.
pub fn number_chain(
    sequence_id: u32,
    nt_seq: &[u8],
    aa_seq: Option<&[u8]>,
    chain: ChainType,
) -> Result<NumberingResult, IgnitionError> {
    let frame = match aa_seq {
        Some(aa) => resolve_with_aa(nt_seq, aa)?,
        None => {
            // Fallback mode
            eprintln!(
                "WARNING [iggnition]: no AA sequence supplied for sequence {}; \
                 auto-detecting reading frame. This is not the designed use case.",
                sequence_id
            );
            resolve_without_aa(nt_seq, Some(chain))?
        }
    };

    let nt_trimmed = &nt_seq[frame.nt_start..];
    number_sequence(sequence_id, nt_trimmed, &frame.aa_seq, chain)
}

/// Number a single sequence, auto-detecting the chain type.
///
/// Tries Heavy, Kappa, Lambda in order, returns the result with the
/// highest alignment score.
pub fn number_chain_auto(
    sequence_id: u32,
    nt_seq: &[u8],
    aa_seq: Option<&[u8]>,
) -> Result<NumberingResult, IgnitionError> {
    use crate::core::aho::identify_v_germline;
    use crate::core::frame::resolve_with_aa;

    let chains = [ChainType::Heavy, ChainType::Kappa, ChainType::Lambda];

    let aa_resolved = match aa_seq {
        Some(aa) => aa.to_vec(),
        None => {
            let fr = resolve_without_aa(nt_seq, None)?;
            fr.aa_seq
        }
    };

    // Pick the chain type with the best V germline score
    let best_chain = chains
        .iter()
        .filter_map(|&c| {
            identify_v_germline(&aa_resolved, c).map(|h| (c, h.score))
        })
        .max_by_key(|&(_, score)| score)
        .map(|(c, _)| c)
        .ok_or(IgnitionError::GermlineNotFound)?;

    let frame = match aa_seq {
        Some(aa) => resolve_with_aa(nt_seq, aa)?,
        None => resolve_without_aa(nt_seq, Some(best_chain))?,
    };

    let nt_trimmed = &nt_seq[frame.nt_start..];
    number_sequence(sequence_id, nt_trimmed, &frame.aa_seq, best_chain)
}

// ── PyO3 module entry point ────────────────────────────────────────────────────
// The function name must match `module-name` in pyproject.toml
// (iggnition._ignition → fn _ignition).
#[cfg(feature = "python")]
#[pyo3::pymodule]
fn _ignition(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    python_api::register(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A real IGHV1-18*01 VDJ rearrangement (truncated for test)
    const HEAVY_NT: &[u8] =
        b"CAGGTGCAGCTGGTGCAGTCTGGAGCTGAGGTGAAGAAGCCTGGGGCCTCAGTGAAGGTCTCCTGCAAGGCTTCTGGTTACACCTTTACCAGCTATGGTATCAGCTGGGTGCGACAGGCCCCTGGACAAGGGCTTGAGTGGATGGGGTGGATCAGCGCTTACAATGGTAACACAAACTATGCACAGAAGCTCCAGGGCAGAGTCACGATGACCACAGACACATCCACGAGCACAGCCTACATGGAGCTGAGGAGCCTGAGATCTGACGACACGGCCGTGTATTACTGTGCGAGA";
    const HEAVY_AA: &[u8] =
        b"QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYGISWVRQAPGQGLEWMGWISAYNGNTNYAQKLQGRVTMTTDTSTSTAYMELRSLRSDDTAVYYCAR";

    #[test]
    fn test_number_chain_heavy() {
        let result = number_chain(0, HEAVY_NT, Some(HEAVY_AA), ChainType::Heavy);
        assert!(result.is_ok(), "Numbering failed: {:?}", result.err());
        let r = result.unwrap();
        assert_eq!(r.chain, ChainType::Heavy);
        assert_eq!(
            r.positions.len(),
            ChainType::Heavy.max_aho_position() as usize * 3
        );
        // First codon positions should be (1, 1), (1, 2), (1, 3)
        assert_eq!(r.positions[0].aho_position, 1);
        assert_eq!(r.positions[0].codon_position, 1);
        assert_eq!(r.positions[1].codon_position, 2);
        assert_eq!(r.positions[2].codon_position, 3);
        println!("Germline: {}", r.germline_id);
    }

    #[test]
    fn test_number_chain_auto_detects_heavy() {
        let result = number_chain_auto(0, HEAVY_NT, Some(HEAVY_AA));
        assert!(result.is_ok(), "Auto-detect failed: {:?}", result.err());
        let r = result.unwrap();
        assert_eq!(r.chain, ChainType::Heavy);
    }
}
