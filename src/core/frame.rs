use crate::core::translate::{find_frame, translate_all_frames};
use crate::core::align::align;
use crate::core::germline::v_germlines;
use crate::core::types::ChainType;
use crate::error::IgnitionError;

/// Frame resolution result: the trimmed nucleotide sequence and amino acid sequence,
/// along with the offset into the original nt sequence where the CDS starts.
pub struct FrameResult {
    /// Byte offset into the original `nt` where the coding sequence starts
    pub nt_start: usize,
    /// Amino acid sequence (no stops)
    pub aa_seq: Vec<u8>,
}

/// Resolve the reading frame when the AA sequence is known.
///
/// Returns the start offset of the AA sequence in `nt`.
pub fn resolve_with_aa(nt: &[u8], aa: &[u8]) -> Result<FrameResult, IgnitionError> {
    match find_frame(nt, aa) {
        Some(offset) => Ok(FrameResult {
            nt_start: offset,
            aa_seq: aa.to_vec(),
        }),
        None => Err(IgnitionError::FrameResolutionFailed),
    }
}

/// Fallback: resolve reading frame without an AA sequence.
///
/// Translates all three frames, aligns each against the best-matching
/// V germline for the given chain type (or all chain types if unknown),
/// and picks the frame with the highest alignment score.
///
/// Emits a warning — caller should surface this to the user.
pub fn resolve_without_aa(nt: &[u8], hint_chain: Option<ChainType>) -> Result<FrameResult, IgnitionError> {
    let frames = translate_all_frames(nt);

    // Choose germlines to test against
    let chains: &[ChainType] = match hint_chain {
        Some(c) => match c {
            ChainType::Heavy => &[ChainType::Heavy],
            ChainType::Kappa => &[ChainType::Kappa],
            ChainType::Lambda => &[ChainType::Lambda],
        },
        None => &[ChainType::Heavy, ChainType::Kappa, ChainType::Lambda],
    };

    let mut best_score = i32::MIN;
    let mut best_frame = 0usize;
    let mut best_aa: Vec<u8> = Vec::new();

    for (frame_idx, aa_frame) in frames.iter().enumerate() {
        if aa_frame.len() < 50 {
            // Too short to be a valid V gene (min ~100aa)
            continue;
        }
        // Score against all germlines, take best
        let frame_best = chains
            .iter()
            .flat_map(|&chain| v_germlines(chain))
            .map(|g| {
                let target: Vec<u8> = g.residues.iter().map(|&(_, _, aa)| aa).collect();
                let aln = align(aa_frame, &target, 11, 1);
                aln.score
            })
            .max()
            .unwrap_or(i32::MIN);

        if frame_best > best_score {
            best_score = frame_best;
            best_frame = frame_idx;
            best_aa = aa_frame.clone();
        }
    }

    if best_aa.is_empty() {
        return Err(IgnitionError::FrameResolutionFailed);
    }

    let nt_start = best_frame; // for simple cases; find_frame would be more precise
    Ok(FrameResult {
        nt_start,
        aa_seq: best_aa,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_with_aa_frame0() {
        // CAG GTG CAG CTG → Q V Q L (frame 0)
        let nt = b"CAGGTGCAGCTG";
        let aa = b"QVQL";
        let res = resolve_with_aa(nt, aa).unwrap();
        assert_eq!(res.nt_start, 0);
        assert_eq!(res.aa_seq, b"QVQL");
    }

    #[test]
    fn test_resolve_with_aa_frame1() {
        // Prepend one extra byte
        let nt = b"XCAGGTGCAGCTG";
        let aa = b"QVQL";
        let res = resolve_with_aa(nt, aa).unwrap();
        assert_eq!(res.nt_start, 1);
    }

    #[test]
    fn test_resolve_with_aa_not_found() {
        let nt = b"AAAAAAAAAA";
        let aa = b"QVQL";
        let result = resolve_with_aa(nt, aa);
        assert!(result.is_err());
    }
}
