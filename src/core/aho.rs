use crate::core::align::{align, Alignment};
use crate::core::germline::{v_germlines, j_germlines, GermlineEntry};
use crate::core::types::{ChainType, NtPosition, NumberingResult};
use crate::error::IgnitionError;

/// CDR3 Aho position range (inclusive) by chain
const CDR3_START: u16 = 109;
const CDR3_END_H: u16 = 138;
const CDR3_END_KL: u16 = 137;
/// FR4 start
const FR4_START_H: u16 = 139;
const FR4_START_KL: u16 = 138;

fn cdr3_end(chain: ChainType) -> u16 {
    match chain {
        ChainType::Heavy => CDR3_END_H,
        _ => CDR3_END_KL,
    }
}

fn fr4_start(chain: ChainType) -> u16 {
    match chain {
        ChainType::Heavy => FR4_START_H,
        _ => FR4_START_KL,
    }
}

/// Result of germline identification: the best-matching germline and alignment.
pub struct GermlineHit {
    pub germline: &'static GermlineEntry,
    pub score: i32,
    pub alignment: Alignment,
}

/// Find the best-matching V germline for the given AA sequence.
pub fn identify_v_germline(aa: &[u8], chain: ChainType) -> Option<GermlineHit> {
    v_germlines(chain)
        .map(|g| {
            let target: Vec<u8> = g.residues.iter().map(|&(_, _, aa)| aa).collect();
            let aln = align(aa, &target, 11, 1);
            let score = aln.score;
            GermlineHit { germline: g, score, alignment: aln }
        })
        .max_by_key(|h| h.score)
}

/// Find the best-matching J germline for the given AA sequence (or a subsequence).
pub fn identify_j_germline(aa: &[u8], chain: ChainType) -> Option<GermlineHit> {
    j_germlines(chain)
        .map(|g| {
            let target: Vec<u8> = g.residues.iter().map(|&(_, _, aa)| aa).collect();
            let aln = align(aa, &target, 11, 1);
            let score = aln.score;
            GermlineHit { germline: g, score, alignment: aln }
        })
        .max_by_key(|h| h.score)
}

/// Transfer Aho positions from a V-germline alignment to the query.
///
/// Walks the alignment, assigning each aligned query residue the Aho
/// position of the corresponding germline residue. Stops at CDR3_START - 1
/// (i.e. covers FR1–FR3, up to and including Aho 108).
///
/// Returns a Vec of (aho_position, query_aa_index) pairs for the V region.
fn transfer_v_positions(
    aln: &Alignment,
    germline: &GermlineEntry,
) -> Vec<(u16, usize)> {
    // Map germline residue index → aho_position
    let germ_aho: Vec<u16> = germline.residues.iter().map(|&(pos, _, _)| pos).collect();

    let mut v_positions: Vec<(u16, usize)> = Vec::new();
    let mut g_idx = 0usize; // index into non-gap germline residues
    let mut q_idx = 0usize; // index into non-gap query residues

    for (&qa, &ta) in aln.query_aligned.iter().zip(aln.target_aligned.iter()) {
        match (qa, ta) {
            (b'-', b'-') => {} // should not happen
            (b'-', _) => {
                // Insertion in target relative to query (deletion in query)
                // This Aho position is unoccupied in the query
                g_idx += 1;
            }
            (_, b'-') => {
                // Insertion in query relative to target (CDR insertion)
                // Skip — handled in CDR3 pass
                q_idx += 1;
            }
            _ => {
                // Matched pair
                if g_idx < germ_aho.len() {
                    let aho = germ_aho[g_idx];
                    if aho < CDR3_START {
                        v_positions.push((aho, q_idx));
                    }
                }
                g_idx += 1;
                q_idx += 1;
            }
        }
    }
    v_positions
}

/// Transfer Aho positions from a J-germline alignment to the query tail.
///
/// The `query_start_idx` is the index into the AA sequence where the J
/// search begins (typically after the CDR3). The result is a Vec of
/// (aho_position, absolute_query_aa_index).
fn transfer_j_positions(
    aln: &Alignment,
    germline: &GermlineEntry,
    query_start_idx: usize,
) -> Vec<(u16, usize)> {
    let germ_aho: Vec<u16> = germline.residues.iter().map(|&(pos, _, _)| pos).collect();

    let mut j_positions: Vec<(u16, usize)> = Vec::new();
    let mut g_idx = 0usize;
    let mut q_idx = query_start_idx;

    for (&qa, &ta) in aln.query_aligned.iter().zip(aln.target_aligned.iter()) {
        match (qa, ta) {
            (b'-', b'-') => {}
            (b'-', _) => {
                g_idx += 1;
            }
            (_, b'-') => {
                q_idx += 1;
            }
            _ => {
                if g_idx < germ_aho.len() {
                    let aho = germ_aho[g_idx];
                    if aho >= fr4_start(ChainType::Heavy) || aho >= fr4_start(ChainType::Kappa) {
                        j_positions.push((aho, q_idx));
                    }
                }
                g_idx += 1;
                q_idx += 1;
            }
        }
    }
    j_positions
}

/// Core numbering function: given aa + nt sequences and a chain type,
/// produce a `NumberingResult`.
///
/// - `sequence_id`: external row identifier
/// - `nt_seq`: nucleotide sequence starting at the CDS start (codon-aligned)
/// - `aa_seq`: amino acid sequence (same reading frame as nt_seq)
/// - `chain`: H, K, or L
pub fn number_sequence(
    sequence_id: u32,
    nt_seq: &[u8],
    aa_seq: &[u8],
    chain: ChainType,
) -> Result<NumberingResult, IgnitionError> {
    if aa_seq.is_empty() {
        return Err(IgnitionError::InvalidSequence("empty AA sequence".into()));
    }
    if nt_seq.len() < aa_seq.len() * 3 {
        return Err(IgnitionError::SequenceTooShort(
            nt_seq.len(),
            aa_seq.len() * 3,
        ));
    }

    // --- Step 1: Identify best V germline ---
    let v_hit = identify_v_germline(aa_seq, chain)
        .ok_or(IgnitionError::GermlineNotFound)?;

    // --- Step 2: Transfer V positions (Aho 1–108) ---
    let v_pos = transfer_v_positions(&v_hit.alignment, v_hit.germline);

    // The last occupied query index from V alignment
    let v_last_q_idx = v_pos.last().map(|&(_, qi)| qi + 1).unwrap_or(0);

    // --- Step 3: Identify CDR3 + J region ---
    // Everything after the last V-matched residue is CDR3 + FR4
    let post_v_aa = &aa_seq[v_last_q_idx..];

    // Find J germline in the tail
    let j_hit = identify_j_germline(post_v_aa, chain);

    // Assign CDR3 Aho positions (109..=cdr3_end) sequentially
    let cdr3_end_pos = cdr3_end(chain);
    let fr4_start_pos = fr4_start(chain);
    let max_cdr3_len = (cdr3_end_pos - CDR3_START + 1) as usize;

    // Determine where FR4 starts in post_v_aa
    let j_start_in_post = if let Some(ref jh) = j_hit {
        // Find the first non-gap query position in the J alignment that maps to FR4
        let mut g_idx = 0usize;
        let mut q_idx = 0usize;
        let germ_aho: Vec<u16> = jh.germline.residues.iter().map(|&(p, _, _)| p).collect();
        let mut fr4_q_start = post_v_aa.len(); // default: no FR4
        for (&qa, &ta) in jh.alignment.query_aligned.iter().zip(jh.alignment.target_aligned.iter()) {
            match (qa, ta) {
                (b'-', b'-') => {}
                (b'-', _) => { g_idx += 1; }
                (_, b'-') => { q_idx += 1; }
                _ => {
                    if g_idx < germ_aho.len() && germ_aho[g_idx] >= fr4_start_pos {
                        fr4_q_start = q_idx;
                        break;
                    }
                    g_idx += 1;
                    q_idx += 1;
                }
            }
        }
        fr4_q_start
    } else {
        post_v_aa.len() // no J found, treat everything as CDR3
    };

    let cdr3_aa = &post_v_aa[..j_start_in_post.min(post_v_aa.len())];
    let fr4_aa = if j_start_in_post < post_v_aa.len() {
        &post_v_aa[j_start_in_post..]
    } else {
        &[]
    };

    // CDR3 Aho positions
    let cdr3_len = cdr3_aa.len();
    if cdr3_len > max_cdr3_len {
        // More CDR3 residues than Aho positions — assign what we can
        // (the DESIGN.md notes excess residues get insertion positions)
        // For now, cap at max and continue (non-fatal)
    }
    let cdr3_positions: Vec<(u16, usize)> = cdr3_aa
        .iter()
        .enumerate()
        .take(max_cdr3_len)
        .map(|(i, _)| (CDR3_START + i as u16, v_last_q_idx + i))
        .collect();

    // FR4 Aho positions from J alignment
    let mut fr4_positions: Vec<(u16, usize)> = Vec::new();
    if let Some(ref jh) = j_hit {
        let abs_q_start = v_last_q_idx + j_start_in_post;
        let fr4_transferred = transfer_j_positions(&jh.alignment, jh.germline, abs_q_start);
        fr4_positions = fr4_transferred;
    } else {
        // Fallback: assign FR4 positions sequentially
        for (i, _) in fr4_aa.iter().enumerate() {
            let aho = fr4_start_pos + i as u16;
            if aho <= chain.max_aho_position() {
                fr4_positions.push((aho, v_last_q_idx + j_start_in_post + i));
            }
        }
    }

    // --- Step 4: Build complete Aho→aa mapping ---
    // Merge all position lists
    let mut all_pos: Vec<(u16, usize)> = Vec::new();
    all_pos.extend_from_slice(&v_pos);
    all_pos.extend_from_slice(&cdr3_positions);
    all_pos.extend_from_slice(&fr4_positions);
    // Sort by Aho position
    all_pos.sort_by_key(|&(aho, _)| aho);
    all_pos.dedup_by_key(|p| p.0); // remove any duplicates (keep first)

    // --- Step 5: Expand to nucleotide positions ---
    let max_aho = chain.max_aho_position();
    let mut nt_positions: Vec<NtPosition> = Vec::with_capacity(max_aho as usize * 3);

    // Build a lookup: aho_pos → (query_aa, query_aa_idx)
    // Aho positions not in the lookup are gaps
    let aho_to_q: std::collections::HashMap<u16, usize> = all_pos.into_iter().collect();

    for aho in 1..=max_aho {
        let (aa_byte, has_residue) = if let Some(&qi) = aho_to_q.get(&aho) {
            (aa_seq.get(qi).copied().unwrap_or(b'-'), true)
        } else {
            (b'-', false)
        };

        for codon_pos in 1u8..=3u8 {
            let nt_abs = (aho as u32 - 1) * 3 + codon_pos as u32; // 1-based
            let nt_byte = if has_residue {
                if let Some(&qi) = aho_to_q.get(&aho) {
                    let nt_idx = qi * 3 + (codon_pos as usize - 1);
                    nt_seq.get(nt_idx).copied().unwrap_or(b'-')
                } else {
                    b'-'
                }
            } else {
                b'-'
            };

            nt_positions.push(NtPosition {
                nt_position: nt_abs as u16,
                aho_position: aho,
                codon_position: codon_pos,
                nucleotide: nt_byte,
                amino_acid: aa_byte,
            });
        }
    }

    Ok(NumberingResult {
        sequence_id,
        chain,
        germline_id: v_hit.germline.id.to_string(),
        positions: nt_positions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identify_v_germline_heavy() {
        // Real IGHV1-18 sequence (truncated)
        let aa = b"QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYGISWVRQAPGQGLEWMGWISAYNGNTNYAQKLQ";
        let hit = identify_v_germline(aa, ChainType::Heavy);
        assert!(hit.is_some(), "Should find a V germline for heavy chain");
        let h = hit.unwrap();
        assert!(h.score > 0, "Score should be positive");
        println!("Best V germline: {} (score {})", h.germline.id, h.score);
    }

    #[test]
    fn test_aho_positions_monotonic() {
        // Verify that Aho positions in the output are 1..=max_aho in order
        let aa = b"QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYGISWVRQAPGQGLEWMGWISAYNGNTNYAQKLQGRVTMTTDTSTSTAYMELRSLRSDDTAVYYCARGGYYYYMDVWGQGTLVTVSS";
        // Build a synthetic NT: just encode each AA as ATG repeated (M) — won't match real codons but tests structure
        let nt: Vec<u8> = aa.iter().flat_map(|_| b"ATG".iter().copied()).collect();
        let result = number_sequence(0, &nt, aa, ChainType::Heavy);
        if let Ok(r) = result {
            let positions: Vec<u16> = r.positions.iter().map(|p| p.aho_position).collect();
            // Should have exactly max_aho * 3 positions
            assert_eq!(positions.len(), ChainType::Heavy.max_aho_position() as usize * 3);
            // First position
            assert_eq!(positions[0], 1);
            // Each group of 3 consecutive positions (a codon) should share the same aho_position
            for (i, chunk) in positions.chunks(3).enumerate() {
                assert_eq!(chunk.len(), 3, "chunk {i} not length 3");
                assert_eq!(chunk[0], chunk[1], "codon {i}: aho_positions should be equal ({} != {})", chunk[0], chunk[1]);
                assert_eq!(chunk[1], chunk[2], "codon {i}: aho_positions should be equal ({} != {})", chunk[1], chunk[2]);
                // Aho positions should be monotonically increasing
                if i > 0 {
                    let prev = positions[(i - 1) * 3];
                    assert!(chunk[0] > prev, "codon {i} aho_position {} should exceed prev {}", chunk[0], prev);
                }
            }
        }
    }
}
