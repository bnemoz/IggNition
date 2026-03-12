use crate::core::types::{ChainType, GeneType};

// Auto-generated germline database (embedded at compile time)
mod data {
    include!("germline_data.rs");
}

pub use data::{GermlineEntry, GermlineResidue, GERMLINES};

/// Return all V-gene germlines for a given chain type
pub fn v_germlines(chain: ChainType) -> impl Iterator<Item = &'static GermlineEntry> {
    GERMLINES
        .iter()
        .filter(move |g| g.chain == chain && g.gene == GeneType::V)
}

/// Return all J-gene germlines for a given chain type
pub fn j_germlines(chain: ChainType) -> impl Iterator<Item = &'static GermlineEntry> {
    GERMLINES
        .iter()
        .filter(move |g| g.chain == chain && g.gene == GeneType::J)
}

/// Extract the amino acid sequence from a germline entry (occupied positions only, in order)
pub fn germline_aa_seq(entry: &GermlineEntry) -> Vec<u8> {
    entry.residues.iter().map(|&(_, _, aa)| aa).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_germlines_loaded() {
        let total: usize = GERMLINES.len();
        assert!(total > 0, "No germlines embedded");
        println!("Total germlines: {total}");
    }

    #[test]
    fn test_heavy_v_germlines() {
        let count = v_germlines(ChainType::Heavy).count();
        assert!(count > 50, "Expected >50 heavy V germlines, got {count}");
    }

    #[test]
    fn test_kappa_v_germlines() {
        let count = v_germlines(ChainType::Kappa).count();
        assert!(count > 20, "Expected >20 kappa V germlines, got {count}");
    }

    #[test]
    fn test_lambda_v_germlines() {
        let count = v_germlines(ChainType::Lambda).count();
        assert!(count > 20, "Expected >20 lambda V germlines, got {count}");
    }

    #[test]
    fn test_aho_positions_in_range() {
        for g in GERMLINES.iter() {
            for &(pos, _ins, _aa) in g.residues {
                let max = g.chain.max_aho_position();
                // Insertion positions can technically exceed max_aho but should be rare
                assert!(
                    pos <= max + 10,
                    "Germline {} has aho position {} > {} + 10",
                    g.id,
                    pos,
                    max
                );
            }
        }
    }

    #[test]
    fn test_germline_aa_seq_non_empty() {
        for g in GERMLINES.iter() {
            let seq = germline_aa_seq(g);
            assert!(!seq.is_empty(), "Germline {} has empty AA seq", g.id);
        }
    }
}
