/// Needleman-Wunsch global pairwise alignment.
///
/// Uses BLOSUM62 for amino acid scoring with affine gap penalties.
/// Returns the alignment as two byte-slices of equal length, with
/// b'-' for gaps. Also returns the raw score.
pub struct Alignment {
    pub score: i32,
    /// query sequence with gaps inserted (same length as `target_aligned`)
    pub query_aligned: Vec<u8>,
    /// target sequence with gaps inserted
    pub target_aligned: Vec<u8>,
}

/// Reusable scratch space for NW — avoids per-alignment heap allocation.
/// Create one per thread and pass to `align_with_workspace()`.
pub struct AlignWorkspace {
    m_mat: Vec<i32>,
    x_mat: Vec<i32>,
    y_mat: Vec<i32>,
    trace: Vec<u8>, // interleaved [m_from, x_from, y_from] per cell
}

impl AlignWorkspace {
    pub fn new() -> Self {
        // Pre-allocate for typical antibody sizes: ~150 aa query × ~110 aa target
        let size = 161 * 161;
        Self {
            m_mat: vec![0i32; size],
            x_mat: vec![0i32; size],
            y_mat: vec![0i32; size],
            trace: vec![0u8; size * 3],
        }
    }

    fn ensure(&mut self, size: usize) {
        if self.m_mat.len() < size {
            self.m_mat.resize(size, 0);
            self.x_mat.resize(size, 0);
            self.y_mat.resize(size, 0);
            self.trace.resize(size * 3, 0);
        }
    }
}

impl Default for AlignWorkspace {
    fn default() -> Self {
        Self::new()
    }
}

/// Align `query` against `target` using pre-allocated workspace (no heap alloc in hot path).
pub fn align_with_workspace(
    query: &[u8],
    target: &[u8],
    gap_open: i32,
    gap_ext: i32,
    ws: &mut AlignWorkspace,
) -> Alignment {
    let m = query.len();
    let n = target.len();

    if m == 0 || n == 0 {
        return Alignment {
            score: 0,
            query_aligned: vec![b'-'; n],
            target_aligned: vec![b'-'; m],
        };
    }

    const NEG_INF: i32 = i32::MIN / 2;
    let stride = n + 1;
    let size = (m + 1) * stride;
    ws.ensure(size);

    let idx = |i: usize, j: usize| i * stride + j;

    // --- Initialise ---
    ws.m_mat[idx(0, 0)] = 0;
    ws.x_mat[idx(0, 0)] = NEG_INF;
    ws.y_mat[idx(0, 0)] = NEG_INF;
    for i in 1..=m {
        ws.x_mat[idx(i, 0)] = -gap_open - (i as i32 - 1) * gap_ext;
        ws.y_mat[idx(i, 0)] = NEG_INF;
        ws.m_mat[idx(i, 0)] = NEG_INF;
    }
    for j in 1..=n {
        ws.y_mat[idx(0, j)] = -gap_open - (j as i32 - 1) * gap_ext;
        ws.x_mat[idx(0, j)] = NEG_INF;
        ws.m_mat[idx(0, j)] = NEG_INF;
    }

    // --- Fill ---
    let blt = blosum62_table(); // single table pointer for the inner loop
    for i in 1..=m {
        let qi = query[i - 1];
        let qi_row = qi as usize * 256; // pre-multiply row for table lookup
        for j in 1..=n {
            let tj = target[j - 1];
            let sub = blt[qi_row + tj as usize] as i32;

            // M: diagonal
            let (m_score, m_from) = max3(
                ws.m_mat[idx(i - 1, j - 1)],
                ws.x_mat[idx(i - 1, j - 1)],
                ws.y_mat[idx(i - 1, j - 1)],
            );
            ws.m_mat[idx(i, j)] = m_score + sub;
            ws.trace[idx(i, j) * 3] = m_from;

            // X: gap in target (query advances from row above)
            let open_x = ws.m_mat[idx(i - 1, j)].saturating_add(-gap_open);
            let ext_x = ws.x_mat[idx(i - 1, j)].saturating_add(-gap_ext);
            let (x_score, x_from) = if open_x >= ext_x { (open_x, 0u8) } else { (ext_x, 1u8) };
            ws.x_mat[idx(i, j)] = x_score;
            ws.trace[idx(i, j) * 3 + 1] = x_from;

            // Y: gap in query (target advances from column left)
            let open_y = ws.m_mat[idx(i, j - 1)].saturating_add(-gap_open);
            let ext_y = ws.y_mat[idx(i, j - 1)].saturating_add(-gap_ext);
            let (y_score, y_from) = if open_y >= ext_y { (open_y, 0u8) } else { (ext_y, 2u8) };
            ws.y_mat[idx(i, j)] = y_score;
            ws.trace[idx(i, j) * 3 + 2] = y_from;
        }
    }

    // --- Best ending state at (m, n) ---
    let (best_score, mut state) = max3(
        ws.m_mat[idx(m, n)],
        ws.x_mat[idx(m, n)],
        ws.y_mat[idx(m, n)],
    );

    // --- Traceback ---
    let mut qa = Vec::with_capacity(m + n);
    let mut ta = Vec::with_capacity(m + n);
    let mut i = m;
    let mut j = n;

    while i > 0 || j > 0 {
        match state {
            0 => {
                if i == 0 || j == 0 { break; }
                qa.push(query[i - 1]);
                ta.push(target[j - 1]);
                state = ws.trace[idx(i, j) * 3];
                i -= 1;
                j -= 1;
            }
            1 => {
                if i == 0 { break; }
                qa.push(query[i - 1]);
                ta.push(b'-');
                state = ws.trace[idx(i, j) * 3 + 1];
                i -= 1;
            }
            2 => {
                if j == 0 { break; }
                qa.push(b'-');
                ta.push(target[j - 1]);
                state = ws.trace[idx(i, j) * 3 + 2];
                j -= 1;
            }
            _ => break,
        }
    }
    while i > 0 { qa.push(query[i - 1]); ta.push(b'-'); i -= 1; }
    while j > 0 { qa.push(b'-'); ta.push(target[j - 1]); j -= 1; }

    qa.reverse();
    ta.reverse();

    Alignment { score: best_score, query_aligned: qa, target_aligned: ta }
}

/// Convenience wrapper (allocates its own workspace — fine for single-sequence use).
pub fn align(query: &[u8], target: &[u8], gap_open: i32, gap_ext: i32) -> Alignment {
    let mut ws = AlignWorkspace::new();
    align_with_workspace(query, target, gap_open, gap_ext, &mut ws)
}

/// Fast O(n+m) pre-filter score based on 2-gram (bigram) AA overlap.
///
/// Returns a count of shared 2-grams. Use this to shortlist candidates
/// before running the full NW alignment. Higher = more similar.
pub fn score_bigram(query: &[u8], target: &[u8]) -> u16 {
    const N: usize = 20;
    const SZ: usize = N * N;
    let mut counts = [0u8; SZ]; // target bigram frequencies (capped at 255)

    let aa_idx = |b: u8| -> usize {
        const AA: &[u8] = b"ACDEFGHIKLMNPQRSTVWY";
        AA.iter().position(|&c| c == b.to_ascii_uppercase()).unwrap_or(0)
    };

    for w in target.windows(2) {
        let k = aa_idx(w[0]) * N + aa_idx(w[1]);
        counts[k] = counts[k].saturating_add(1);
    }

    let mut score = 0u16;
    for w in query.windows(2) {
        let k = aa_idx(w[0]) * N + aa_idx(w[1]);
        if counts[k] > 0 {
            score += 1;
            counts[k] -= 1; // don't double-count
        }
    }
    score
}

#[inline]
fn max3(a: i32, b: i32, c: i32) -> (i32, u8) {
    if a >= b && a >= c { (a, 0) }
    else if b >= c       { (b, 1) }
    else                 { (c, 2) }
}

/// BLOSUM62 20×20 matrix (row-major, order: ARNDCQEGHILKMFPSTWYV)
#[rustfmt::skip]
const BLOSUM62_20: [i8; 400] = [
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

/// Precomputed 256×256 BLOSUM62 lookup table indexed by ASCII byte values.
/// Built once at startup via lazy_static equivalent (const fn not available for this size).
/// Use `blosum62_fast(a, b)` for hot-path access.
static BLOSUM62_256: std::sync::OnceLock<[i8; 65536]> = std::sync::OnceLock::new();

fn blosum62_table() -> &'static [i8; 65536] {
    BLOSUM62_256.get_or_init(|| {
        const AA: &[u8] = b"ARNDCQEGHILKMFPSTWYV";
        let mut table = [-4i8; 65536];
        for (i, &a) in AA.iter().enumerate() {
            for (j, &b) in AA.iter().enumerate() {
                let score = BLOSUM62_20[i * 20 + j];
                table[a as usize * 256 + b as usize] = score;
                // Also map lowercase
                table[a.to_ascii_lowercase() as usize * 256 + b as usize] = score;
                table[a as usize * 256 + b.to_ascii_lowercase() as usize] = score;
                table[a.to_ascii_lowercase() as usize * 256 + b.to_ascii_lowercase() as usize] = score;
            }
        }
        table
    })
}

/// O(1) BLOSUM62 lookup via precomputed 256×256 table.
#[inline(always)]
pub fn blosum62_fast(a: u8, b: u8) -> i32 {
    blosum62_table()[a as usize * 256 + b as usize] as i32
}

/// BLOSUM62 substitution score for two uppercase amino acid bytes.
/// For non-hot-path use. Use `blosum62_fast` in inner loops.
#[inline]
pub fn blosum62(a: u8, b: u8) -> i32 {
    blosum62_fast(a, b)
}

/// O(n) ungapped BLOSUM62 score on min(|q|, |t|) positions.
/// Use for fast candidate pre-ranking.
#[inline]
pub fn score_ungapped(query: &[u8], target: &[u8]) -> i32 {
    let table = blosum62_table();
    let n = query.len().min(target.len());
    (0..n).map(|i| table[query[i] as usize * 256 + target[i] as usize] as i32).sum()
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
        let query  = b"ACGT";
        let target = b"ACXGT";
        let aln = align(query, target, 11, 1);
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

    #[test]
    fn test_workspace_same_result_as_allocating() {
        let q = b"QVQLVQSGAEVKKPG";
        let t = b"QVQLVQSGAEVKKPGASVKVSCK";
        let aln1 = align(q, t, 11, 1);
        let mut ws = AlignWorkspace::new();
        let aln2 = align_with_workspace(q, t, 11, 1, &mut ws);
        assert_eq!(aln1.score, aln2.score);
        assert_eq!(aln1.query_aligned, aln2.query_aligned);
    }

    #[test]
    fn test_score_bigram_identical() {
        let q = b"QVQLVQSGA";
        let score = score_bigram(q, q);
        assert_eq!(score as usize, q.len() - 1, "Identical sequences: all bigrams should match");
    }

    #[test]
    fn test_score_bigram_unrelated() {
        let q = b"QVQLVQ";
        let t = b"XXXXXX";
        let score = score_bigram(q, t);
        assert_eq!(score, 0);
    }
}
