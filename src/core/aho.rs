use crate::core::align::{
    align_with_workspace, score_bigram, score_ungapped, AlignWorkspace, Alignment,
};
use crate::core::germline::{germline_aa_seq, j_germlines, v_germlines, GermlineEntry};
use crate::core::types::{ChainType, NtPosition, NumberingResult};
use crate::error::IgnitionError;

/// Stage 1 (bigram): shortlist this many candidates for ungapped scoring.
const BIGRAM_K: usize = 40;
/// Stage 2 (ungapped BLOSUM62): pass this many candidates to full NW with traceback.
const UNGAPPED_K: usize = 4;

/// CDR3 Aho position range (inclusive) by chain.
///
/// Canonical Aho boundaries (matching regions.py CDR_REGIONS / NUMBERING_CROSSREF):
///   Heavy : CDR3 109..=137, FR4 138..=149 (Aho 138 = FR4-start / J-gene Trp)
///   K / L : CDR3 109..=138, FR4 139..=148 (Aho 139 = FR4-start)
/// NOTE: the heavy and light values were previously swapped here, which (together
/// with the J-transfer offset below) caused FR4 to be mis-numbered / dropped.
const CDR3_START: u16 = 109;
const CDR3_END_H: u16 = 137;
const CDR3_END_KL: u16 = 138;
/// FR4 start
const FR4_START_H: u16 = 138;
const FR4_START_KL: u16 = 139;

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

/// AHo numbers CDR3 **symmetrically**, not left-to-right: residues fill inward
/// from both ends toward a central apex gap, so structurally equivalent
/// positions line up across antibodies of different CDR3 length. The N-terminal
/// residues ascend from 109; the C-terminal residues descend from `cdr3_end`
/// (137 for heavy, 138 for K/L); the apex stays gapped for shorter loops.
///
/// The odd residue (for odd lengths) goes to the **N-side for heavy** and the
/// **C-side for K/L** — this is the empirically observed Honegger–Plückthun /
/// ANARCI convention (verified against ANARCI across the human repertoire).
///
/// Returns the AHo position for each CDR3 query residue, in query order, so the
/// i-th returned value is the AHo position of the i-th CDR3 residue. The result
/// is capped at the CDR3 capacity (`cdr3_end - 109 + 1`): heavy 29, K/L 30.
/// Lengths beyond that would require AHo apex *insertion letters*, which the
/// integer-only output cannot represent — callers handle/flag overflow.
fn cdr3_aho_positions(chain: ChainType, len: usize) -> Vec<u16> {
    let cdr3_end_pos = cdr3_end(chain);
    let capacity = (cdr3_end_pos - CDR3_START + 1) as usize;
    let l = len.min(capacity);

    // Split: heavy puts the odd residue on the N-side (ceil), K/L on the C-side.
    let n_count = match chain {
        ChainType::Heavy => l.div_ceil(2),
        _ => l / 2,
    };
    let c_count = l - n_count;
    // C-side occupies the top `c_count` positions ending at cdr3_end, ascending.
    let c_start = cdr3_end_pos - c_count as u16 + 1;

    let mut positions = Vec::with_capacity(l);
    for i in 0..l {
        if i < n_count {
            positions.push(CDR3_START + i as u16); // N-side: 109, 110, ...
        } else {
            positions.push(c_start + (i - n_count) as u16); // C-side, ascending
        }
    }
    positions
}

/// Result of germline identification: the best-matching germline and alignment.
pub struct GermlineHit {
    pub germline: &'static GermlineEntry,
    pub score: i32,
    pub alignment: Alignment,
}

/// Find the best-matching V germline for the given AA sequence.
///
/// Uses bigram pre-filter to shortlist the top `KMER_PREFILTER_K` candidates
/// before running full NW. Pass an `AlignWorkspace` to avoid per-call allocation.
pub fn identify_v_germline(aa: &[u8], chain: ChainType) -> Option<GermlineHit> {
    let mut ws = AlignWorkspace::new();
    identify_v_germline_ws(aa, chain, &mut ws)
}

pub fn identify_v_germline_ws(
    aa: &[u8],
    chain: ChainType,
    ws: &mut AlignWorkspace,
) -> Option<GermlineHit> {
    // --- Tier 1: bigram pre-filter (all germlines, O(n) each) ---
    let all: Vec<&'static GermlineEntry> = v_germlines(chain).collect();
    if all.is_empty() {
        return None;
    }
    let mut tier1: Vec<(u16, &'static GermlineEntry)> = all
        .iter()
        .map(|&g| (score_bigram(aa, &germline_aa_seq(g)), g))
        .collect();

    let bigram_k = BIGRAM_K.min(tier1.len());
    tier1.select_nth_unstable_by(bigram_k - 1, |a, b| b.0.cmp(&a.0));
    let tier1_top = &tier1[..bigram_k];

    // --- Tier 2: ungapped BLOSUM62 score (top bigram_k, O(n) each) ---
    let mut tier2: Vec<(i32, &'static GermlineEntry)> = tier1_top
        .iter()
        .map(|&(_, g)| (score_ungapped(aa, &germline_aa_seq(g)), g))
        .collect();

    let ungapped_k = UNGAPPED_K.min(tier2.len());
    tier2.select_nth_unstable_by(ungapped_k - 1, |a, b| b.0.cmp(&a.0));
    let candidates = &tier2[..ungapped_k];

    // --- Tier 3: full NW with traceback (top ungapped_k only) ---
    candidates
        .iter()
        .map(|&(_, g)| {
            let target = germline_aa_seq(g);
            // V germline: free the query 3' overhang so the CDR3+FR4 tail does
            // not get spuriously absorbed into the V alignment.
            let aln = align_with_workspace(aa, &target, 11, 1, false, true, ws);
            let score = aln.score;
            GermlineHit {
                germline: g,
                score,
                alignment: aln,
            }
        })
        .max_by_key(|h| h.score)
}

/// Find the best-matching J germline for the given AA sequence.
pub fn identify_j_germline(aa: &[u8], chain: ChainType) -> Option<GermlineHit> {
    let mut ws = AlignWorkspace::new();
    identify_j_germline_ws(aa, chain, &mut ws)
}

pub fn identify_j_germline_ws(
    aa: &[u8],
    chain: ChainType,
    ws: &mut AlignWorkspace,
) -> Option<GermlineHit> {
    // J germlines are few (≤14 for heavy) — no pre-filter needed
    j_germlines(chain)
        .map(|g| {
            let target = germline_aa_seq(g);
            // J germline: free the query 5' overhang so the CDR3 prefix does not
            // get spuriously absorbed into the J alignment (the J framework/FR4
            // sits at the 3' end of the query tail).
            let aln = align_with_workspace(aa, &target, 11, 1, true, false, ws);
            let score = aln.score;
            GermlineHit {
                germline: g,
                score,
                alignment: aln,
            }
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
fn transfer_v_positions(aln: &Alignment, germline: &GermlineEntry) -> Vec<(u16, usize)> {
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
    fr4_start_pos: u16,
) -> Vec<(u16, usize)> {
    let germ_aho: Vec<u16> = germline.residues.iter().map(|&(pos, _, _)| pos).collect();

    let mut j_positions: Vec<(u16, usize)> = Vec::new();
    let mut g_idx = 0usize;
    // `query_start_idx` MUST be the absolute query index of the FIRST query
    // residue covered by this (global) alignment — i.e. the start of post_v_aa,
    // which is `v_last_q_idx`. We then walk the whole alignment and let q_idx
    // advance naturally. (Passing the FR4 offset here would double-count the
    // CDR3 stretch and shift every FR4 residue to the wrong query position.)
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
                    // Keep only the FR4 portion of the J alignment, using the
                    // chain-specific FR4 start (heavy 138, K/L 139).
                    if aho >= fr4_start_pos {
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
    let mut ws = AlignWorkspace::new();
    number_sequence_ws(sequence_id, nt_seq, aa_seq, chain, &mut ws)
}

/// Like `number_sequence` but accepts a reusable workspace (avoids per-call allocation).
pub fn number_sequence_ws(
    sequence_id: u32,
    nt_seq: &[u8],
    aa_seq: &[u8],
    chain: ChainType,
    ws: &mut AlignWorkspace,
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
    let v_hit = identify_v_germline_ws(aa_seq, chain, ws).ok_or(IgnitionError::GermlineNotFound)?;

    // --- Step 2: Transfer V positions (Aho 1–108) ---
    let v_pos = transfer_v_positions(&v_hit.alignment, v_hit.germline);

    // The last occupied query index from V alignment
    let v_last_q_idx = v_pos.last().map(|&(_, qi)| qi + 1).unwrap_or(0);

    // --- Step 3: Identify CDR3 + J region ---
    let post_v_aa = &aa_seq[v_last_q_idx..];

    // Find J germline in the tail
    let j_hit = identify_j_germline_ws(post_v_aa, chain, ws);

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
        for (&qa, &ta) in jh
            .alignment
            .query_aligned
            .iter()
            .zip(jh.alignment.target_aligned.iter())
        {
            match (qa, ta) {
                (b'-', b'-') => {}
                (b'-', _) => {
                    g_idx += 1;
                }
                (_, b'-') => {
                    q_idx += 1;
                }
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

    // CDR3 Aho positions — symmetric AHo fill (see cdr3_aho_positions). Residues
    // beyond the integer-slot capacity (heavy 29 / K-L 30) cannot be represented
    // without apex insertion letters; they are dropped here (vanishingly rare for
    // well-formed antibodies — quantified in the audit) and the capped count is
    // assigned symmetrically.
    let cdr3_len = cdr3_aa.len();
    let cdr3_slots = cdr3_aho_positions(chain, cdr3_len); // len == min(cdr3_len, max_cdr3_len)
    let _ = max_cdr3_len;
    let cdr3_positions: Vec<(u16, usize)> = cdr3_slots
        .iter()
        .enumerate()
        .map(|(i, &aho)| (aho, v_last_q_idx + i))
        .collect();

    // FR4 Aho positions from J alignment
    let mut fr4_positions: Vec<(u16, usize)> = Vec::new();
    if let Some(ref jh) = j_hit {
        // The J alignment is global over post_v_aa, whose first residue sits at
        // absolute query index v_last_q_idx. transfer_j_positions walks the whole
        // alignment and filters to FR4 (aho >= fr4_start_pos) itself, so we pass
        // v_last_q_idx as the base — NOT v_last_q_idx + j_start_in_post.
        let fr4_transferred =
            transfer_j_positions(&jh.alignment, jh.germline, v_last_q_idx, fr4_start_pos);
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

    /// Regression guard: the Rust CDR3/FR4 boundary constants MUST stay in sync
    /// with the canonical region map in `python/iggnition/regions.py`
    /// (`CDR_REGIONS` / `NUMBERING_CROSSREF`). Those define, per chain:
    ///     Heavy : CDR3 109..=137, FR4 138..=149  (Aho 138 = FR4 start / J-gene Trp)
    ///     K / L : CDR3 109..=138, FR4 139..=148  (Aho 139 = FR4 start)
    /// These two sources were previously swapped on the Rust side, which mangled
    /// every FR4 (the J framework was dropped/mis-numbered). If you change one
    /// side, change the other — this test fails loudly if they diverge again.
    #[test]
    fn test_cdr3_fr4_boundaries_match_canonical_region_map() {
        assert_eq!(CDR3_START, 109, "CDR3 starts at Aho 109 for all chains");
        assert_eq!(CDR3_END_H, 137, "heavy CDR3 ends at Aho 137 (regions.py)");
        assert_eq!(FR4_START_H, 138, "heavy FR4 starts at Aho 138 (regions.py)");
        assert_eq!(CDR3_END_KL, 138, "K/L CDR3 ends at Aho 138 (regions.py)");
        assert_eq!(FR4_START_KL, 139, "K/L FR4 starts at Aho 139 (regions.py)");
        // FR4 must begin exactly one position past CDR3 end, per chain.
        assert_eq!(FR4_START_H, CDR3_END_H + 1);
        assert_eq!(FR4_START_KL, CDR3_END_KL + 1);
        // Sanity vs the public chain API.
        assert_eq!(cdr3_end(ChainType::Heavy), 137);
        assert_eq!(fr4_start(ChainType::Heavy), 138);
        assert_eq!(cdr3_end(ChainType::Kappa), 138);
        assert_eq!(fr4_start(ChainType::Kappa), 139);
        assert_eq!(cdr3_end(ChainType::Lambda), 138);
        assert_eq!(fr4_start(ChainType::Lambda), 139);
    }

    /// Golden test: CDR3 symmetric fill must match ANARCI's AHo numbering.
    /// Expected position sets were extracted from ANARCI (scheme="aho") over the
    /// human repertoire and are the Honegger–Plückthun convention (apex gap;
    /// odd residue N-side for heavy, C-side for K/L). See cdr3_aho_positions.
    #[test]
    fn test_cdr3_symmetric_fill_matches_anarci() {
        // Heavy (span 109..=137, odd -> N-side).
        assert_eq!(cdr3_aho_positions(ChainType::Heavy, 2), vec![109, 137]);
        assert_eq!(cdr3_aho_positions(ChainType::Heavy, 3), vec![109, 110, 137]);
        assert_eq!(
            cdr3_aho_positions(ChainType::Heavy, 4),
            vec![109, 110, 136, 137]
        );
        assert_eq!(
            cdr3_aho_positions(ChainType::Heavy, 5),
            vec![109, 110, 111, 136, 137]
        );
        assert_eq!(
            cdr3_aho_positions(ChainType::Heavy, 22),
            vec![
                109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 127, 128, 129, 130, 131,
                132, 133, 134, 135, 136, 137
            ]
        );
        // K/L (span 109..=138, odd -> C-side).
        assert_eq!(cdr3_aho_positions(ChainType::Kappa, 2), vec![109, 138]);
        assert_eq!(cdr3_aho_positions(ChainType::Kappa, 3), vec![109, 137, 138]);
        assert_eq!(
            cdr3_aho_positions(ChainType::Lambda, 5),
            vec![109, 110, 136, 137, 138]
        );
        assert_eq!(
            cdr3_aho_positions(ChainType::Lambda, 11),
            vec![109, 110, 111, 112, 113, 133, 134, 135, 136, 137, 138]
        );
        // Every returned position must lie in the chain's CDR3 span, ascending,
        // and never collide with the apex gap (no duplicates).
        for &chain in &[ChainType::Heavy, ChainType::Kappa, ChainType::Lambda] {
            let cap = (cdr3_end(chain) - CDR3_START + 1) as usize;
            for l in 0..=cap {
                let p = cdr3_aho_positions(chain, l);
                assert_eq!(p.len(), l);
                assert!(
                    p.windows(2).all(|w| w[0] < w[1]),
                    "must be strictly ascending"
                );
                assert!(p
                    .iter()
                    .all(|&x| (CDR3_START..=cdr3_end(chain)).contains(&x)));
            }
            // Overflow is capped at capacity, not panicking.
            assert_eq!(cdr3_aho_positions(chain, cap + 5).len(), cap);
        }
    }

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
            assert_eq!(
                positions.len(),
                ChainType::Heavy.max_aho_position() as usize * 3
            );
            // First position
            assert_eq!(positions[0], 1);
            // Each group of 3 consecutive positions (a codon) should share the same aho_position
            for (i, chunk) in positions.chunks(3).enumerate() {
                assert_eq!(chunk.len(), 3, "chunk {i} not length 3");
                assert_eq!(
                    chunk[0], chunk[1],
                    "codon {i}: aho_positions should be equal ({} != {})",
                    chunk[0], chunk[1]
                );
                assert_eq!(
                    chunk[1], chunk[2],
                    "codon {i}: aho_positions should be equal ({} != {})",
                    chunk[1], chunk[2]
                );
                // Aho positions should be monotonically increasing
                if i > 0 {
                    let prev = positions[(i - 1) * 3];
                    assert!(
                        chunk[0] > prev,
                        "codon {i} aho_position {} should exceed prev {}",
                        chunk[0],
                        prev
                    );
                }
            }
        }
    }
}
