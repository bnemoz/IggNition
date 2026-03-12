/// Needleman-Wunsch global pairwise alignment.
///
/// Uses BLOSUM62 for amino acid scoring with affine gap penalties.
/// Returns the alignment as two byte-slices of equal length, with
/// b'-' for gaps. Also returns the raw score.
///
/// # Arguments
/// * `query`    - Query amino acid sequence (ASCII uppercase)
/// * `target`   - Target amino acid sequence (ASCII uppercase)
/// * `gap_open` - Gap-open penalty (positive value, will be subtracted)
/// * `gap_ext`  - Gap-extension penalty (positive value)

pub struct Alignment {
    pub score: i32,
    /// query sequence with gaps inserted (same length as `target_aligned`)
    pub query_aligned: Vec<u8>,
    /// target sequence with gaps inserted
    pub target_aligned: Vec<u8>,
}

pub fn align(query: &[u8], target: &[u8], gap_open: i32, gap_ext: i32) -> Alignment {
    let m = query.len();
    let n = target.len();

    if m == 0 || n == 0 {
        return Alignment {
            score: 0,
            query_aligned: vec![b'-'; n],
            target_aligned: vec![b'-'; m],
        };
    }

    // Three matrices: M (match/mismatch), X (gap in target), Y (gap in query)
    const NEG_INF: i32 = i32::MIN / 2;

    let idx = |i: usize, j: usize| i * (n + 1) + j;
    let size = (m + 1) * (n + 1);

    let mut m_mat = vec![NEG_INF; size];
    let mut x_mat = vec![NEG_INF; size]; // gap in target (query has residue, target has gap)
    let mut y_mat = vec![NEG_INF; size]; // gap in query

    // Traceback: 0=M, 1=X, 2=Y
    let mut trace_mat = vec![0u8; size * 3]; // [m_trace, x_trace, y_trace] interleaved

    // Initialise
    m_mat[idx(0, 0)] = 0;
    for i in 1..=m {
        x_mat[idx(i, 0)] = -gap_open - (i as i32 - 1) * gap_ext;
        y_mat[idx(i, 0)] = NEG_INF;
        m_mat[idx(i, 0)] = NEG_INF;
    }
    for j in 1..=n {
        y_mat[idx(0, j)] = -gap_open - (j as i32 - 1) * gap_ext;
        x_mat[idx(0, j)] = NEG_INF;
        m_mat[idx(0, j)] = NEG_INF;
    }

    // Fill
    for i in 1..=m {
        let qi = query[i - 1];
        for j in 1..=n {
            let tj = target[j - 1];
            let sub = blosum62(qi, tj);

            // M[i][j]: extend from M, X, or Y at [i-1][j-1]
            let prev_m = m_mat[idx(i - 1, j - 1)];
            let prev_x = x_mat[idx(i - 1, j - 1)];
            let prev_y = y_mat[idx(i - 1, j - 1)];
            let (m_score, m_from) = max3(prev_m, prev_x, prev_y);
            m_mat[idx(i, j)] = m_score + sub;
            trace_mat[(idx(i, j)) * 3] = m_from;

            // X[i][j]: gap in target, query has residue at i (move from [i-1][j])
            let open_x = m_mat[idx(i - 1, j)].saturating_add(-gap_open);
            let ext_x = x_mat[idx(i - 1, j)].saturating_add(-gap_ext);
            let (x_score, x_from) = if open_x >= ext_x { (open_x, 0u8) } else { (ext_x, 1u8) };
            x_mat[idx(i, j)] = x_score;
            trace_mat[(idx(i, j)) * 3 + 1] = x_from;

            // Y[i][j]: gap in query, target has residue at j (move from [i][j-1])
            let open_y = m_mat[idx(i, j - 1)].saturating_add(-gap_open);
            let ext_y = y_mat[idx(i, j - 1)].saturating_add(-gap_ext);
            let (y_score, y_from) = if open_y >= ext_y { (open_y, 0u8) } else { (ext_y, 2u8) };
            y_mat[idx(i, j)] = y_score;
            trace_mat[(idx(i, j)) * 3 + 2] = y_from;
        }
    }

    // Find best ending state at (m, n)
    let sm = m_mat[idx(m, n)];
    let sx = x_mat[idx(m, n)];
    let sy = y_mat[idx(m, n)];
    let (best_score, mut state) = max3(sm, sx, sy);

    // Traceback
    let mut qa = Vec::with_capacity(m + n);
    let mut ta = Vec::with_capacity(m + n);
    let mut i = m;
    let mut j = n;

    while i > 0 || j > 0 {
        match state {
            0 => {
                // M: diagonal
                if i == 0 || j == 0 {
                    break;
                }
                qa.push(query[i - 1]);
                ta.push(target[j - 1]);
                state = trace_mat[idx(i, j) * 3];
                i -= 1;
                j -= 1;
            }
            1 => {
                // X: gap in target (query advances)
                if i == 0 {
                    break;
                }
                qa.push(query[i - 1]);
                ta.push(b'-');
                state = trace_mat[idx(i, j) * 3 + 1];
                i -= 1;
            }
            2 => {
                // Y: gap in query (target advances)
                if j == 0 {
                    break;
                }
                qa.push(b'-');
                ta.push(target[j - 1]);
                state = trace_mat[idx(i, j) * 3 + 2];
                j -= 1;
            }
            _ => break,
        }
    }

    // Handle remaining positions
    while i > 0 {
        qa.push(query[i - 1]);
        ta.push(b'-');
        i -= 1;
    }
    while j > 0 {
        qa.push(b'-');
        ta.push(target[j - 1]);
        j -= 1;
    }

    qa.reverse();
    ta.reverse();

    Alignment {
        score: best_score,
        query_aligned: qa,
        target_aligned: ta,
    }
}

#[inline]
fn max3(a: i32, b: i32, c: i32) -> (i32, u8) {
    if a >= b && a >= c {
        (a, 0)
    } else if b >= c {
        (b, 1)
    } else {
        (c, 2)
    }
}

/// BLOSUM62 substitution score for two uppercase amino acid bytes.
/// Returns -4 for unknown residues.
pub fn blosum62(a: u8, b: u8) -> i32 {
    const AA: &[u8] = b"ARNDCQEGHILKMFPSTWYV";
    // Flat row-major BLOSUM62 matrix (20x20)
    #[rustfmt::skip]
    const BLOSUM62: [i8; 400] = [
    //  A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V
        4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, // A
       -1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, // R
       -2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3, // N
       -2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3, // D
        0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, // C
       -1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2, // Q
       -1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2, // E
        0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, // G
       -2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3, // H
       -1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, // I
       -1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, // L
       -1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2, // K
       -1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, // M
       -2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, // F
       -1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, // P
        1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2, // S
        0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, // T
       -3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, // W
       -2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, // Y
        0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, // V
    ];

    let ai = AA.iter().position(|&c| c == a.to_ascii_uppercase());
    let bi = AA.iter().position(|&c| c == b.to_ascii_uppercase());
    match (ai, bi) {
        (Some(i), Some(j)) => BLOSUM62[i * 20 + j] as i32,
        _ => -4,
    }
}

/// Score an alignment of `query` aa against `target` aa without traceback.
/// Uses sum of BLOSUM62 scores + affine gap penalties.
/// This is a fast scoring function for candidate ranking.
pub fn score_ungapped(query: &[u8], target: &[u8]) -> i32 {
    query
        .iter()
        .zip(target.iter())
        .map(|(&a, &b)| blosum62(a, b))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blosum62_identity() {
        for &aa in b"ARNDCQEGHILKMFPSTWYV" {
            let s = blosum62(aa, aa);
            assert!(s > 0, "Self-score for {} should be positive, got {}", aa as char, s);
        }
    }

    #[test]
    fn test_align_identical() {
        let seq = b"QVQLVQSGA";
        let aln = align(seq, seq, 11, 1);
        assert_eq!(aln.query_aligned, seq);
        assert_eq!(aln.target_aligned, seq);
        assert!(aln.score > 0);
    }

    #[test]
    fn test_align_single_deletion() {
        // Query has one residue deleted relative to target
        let query  = b"ACGT";
        let target = b"ACXGT";
        let aln = align(query, target, 11, 1);
        // Should align with a gap somewhere
        assert_eq!(aln.query_aligned.len(), aln.target_aligned.len());
        assert!(aln.query_aligned.contains(&b'-') || aln.target_aligned.contains(&b'-'));
    }

    #[test]
    fn test_align_lengths_match() {
        let q = b"QVQLVQSGAEVKKPG";
        let t = b"QVQLVQSGAEVKKPGASVKVSCK";
        let aln = align(q, t, 11, 1);
        assert_eq!(aln.query_aligned.len(), aln.target_aligned.len());
    }
}
