use std::io::Write;

use crate::core::types::{ChainType, NumberingResult};
use crate::error::IgnitionError;

/// Output format selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputFormat {
    /// One row per nucleotide position (default)
    #[default]
    PerNucleotide,
    /// One row per codon (Aho position)
    PerCodon,
    /// One row per sequence, positional columns
    Wide,
}

// ─── Per-nucleotide ───────────────────────────────────────────────────────────

pub const PER_NT_HEADER: &str =
    "sequence_id\tchain\tnt_position\taho_position\tcodon_position\tnucleotide\tamino_acid\n";

/// Write per-nucleotide TSV rows for a single result.
pub fn write_per_nt_rows<W: Write>(
    w: &mut W,
    result: &NumberingResult,
) -> Result<(), IgnitionError> {
    for pos in &result.positions {
        writeln!(
            w,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}",
            result.sequence_id,
            result.chain,
            pos.nt_position,
            pos.aho_position,
            pos.codon_position,
            pos.nucleotide as char,
            pos.amino_acid as char,
        )
        .map_err(|e| IgnitionError::Io(e.to_string()))?;
    }
    Ok(())
}

// ─── Per-codon ────────────────────────────────────────────────────────────────

pub const PER_CODON_HEADER: &str = "sequence_id\tchain\taho_position\tcodon\tamino_acid\n";

/// Write per-codon TSV rows for a single result.
pub fn write_per_codon_rows<W: Write>(
    w: &mut W,
    result: &NumberingResult,
) -> Result<(), IgnitionError> {
    let positions = &result.positions;
    let n = positions.len();
    // positions is already sorted by aho_position, codon_position in groups of 3
    let mut i = 0;
    while i + 2 < n {
        let p1 = &positions[i];
        let p2 = &positions[i + 1];
        let p3 = &positions[i + 2];
        debug_assert_eq!(p1.aho_position, p2.aho_position);
        debug_assert_eq!(p2.aho_position, p3.aho_position);
        writeln!(
            w,
            "{}\t{}\t{}\t{}{}{}\t{}",
            result.sequence_id,
            result.chain,
            p1.aho_position,
            p1.nucleotide as char,
            p2.nucleotide as char,
            p3.nucleotide as char,
            p1.amino_acid as char,
        )
        .map_err(|e| IgnitionError::Io(e.to_string()))?;
        i += 3;
    }
    Ok(())
}

// ─── Wide format ─────────────────────────────────────────────────────────────

/// Generate the wide-format header for a given chain type.
///
/// Columns: ``sequence_id``, then ``H{n}`` for heavy-chain positions and ``L{n}``
/// for both Kappa and Lambda positions (Kappa and Lambda share the ``L`` prefix).
pub fn wide_header(chains: &[ChainType]) -> String {
    let mut cols = vec!["sequence_id".to_string()];
    for &chain in chains {
        let prefix = match chain {
            ChainType::Heavy => "H",
            ChainType::Kappa | ChainType::Lambda => "L",
        };
        let max_nt = chain.max_nt_positions();
        for nt_pos in 1..=max_nt {
            cols.push(format!("{}{}", prefix, nt_pos));
        }
    }
    cols.join("\t") + "\n"
}

/// Write wide-format TSV rows for a single result.
///
/// Assumes the header was already written (caller must ensure column ordering matches).
pub fn write_wide_row<W: Write>(
    w: &mut W,
    result: &NumberingResult,
    chain_order: &[ChainType],
) -> Result<(), IgnitionError> {
    let max_nt = result.chain.max_nt_positions() as usize;
    // Build a position → nucleotide lookup (1-based index)
    let mut nt_map = vec![b'-'; max_nt + 1];
    for pos in &result.positions {
        let idx = pos.nt_position as usize;
        if idx > 0 && idx <= max_nt {
            nt_map[idx] = pos.nucleotide;
        }
    }

    write!(w, "{}", result.sequence_id).map_err(|e| IgnitionError::Io(e.to_string()))?;

    for &chain in chain_order {
        if chain == result.chain {
            let max = chain.max_nt_positions() as usize;
            for &nt in &nt_map[1..=max] {
                write!(w, "\t{}", nt as char).map_err(|e| IgnitionError::Io(e.to_string()))?;
            }
        } else {
            // Fill with gaps for chains not present in this result
            let max = chain.max_nt_positions() as usize;
            for _ in 1..=max {
                write!(w, "\t-").map_err(|e| IgnitionError::Io(e.to_string()))?;
            }
        }
    }
    writeln!(w).map_err(|e| IgnitionError::Io(e.to_string()))?;
    Ok(())
}

// ─── Batch output ─────────────────────────────────────────────────────────────

/// Write all results to `writer` in the given format.
pub fn write_results<W: Write>(
    w: &mut W,
    results: &[NumberingResult],
    format: OutputFormat,
) -> Result<(), IgnitionError> {
    match format {
        OutputFormat::PerNucleotide => {
            write!(w, "{}", PER_NT_HEADER).map_err(|e| IgnitionError::Io(e.to_string()))?;
            for r in results {
                write_per_nt_rows(w, r)?;
            }
        }
        OutputFormat::PerCodon => {
            write!(w, "{}", PER_CODON_HEADER).map_err(|e| IgnitionError::Io(e.to_string()))?;
            for r in results {
                write_per_codon_rows(w, r)?;
            }
        }
        OutputFormat::Wide => {
            // Determine chain set from results
            let mut chains_seen = vec![];
            for r in results {
                if !chains_seen.contains(&r.chain) {
                    chains_seen.push(r.chain);
                }
            }
            chains_seen.sort_by_key(|c| c.as_str());
            write!(w, "{}", wide_header(&chains_seen))
                .map_err(|e| IgnitionError::Io(e.to_string()))?;
            for r in results {
                write_wide_row(w, r, &chains_seen)?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{number_chain, ChainType};

    const HEAVY_NT: &[u8] = b"CAGGTGCAGCTGGTGCAGTCTGGAGCTGAGGTGAAGAAGCCTGGGGCCTCAGTGAAGGTCTCCTGCAAGGCTTCTGGTTACACCTTTACCAGCTATGGTATCAGCTGGGTGCGACAGGCCCCTGGACAAGGGCTTGAGTGGATGGGGTGGATCAGCGCTTACAATGGTAACACAAACTATGCACAGAAGCTCCAGGGCAGAGTCACGATGACCACAGACACATCCACGAGCACAGCCTACATGGAGCTGAGGAGCCTGAGATCTGACGACACGGCCGTGTATTACTGTGCGAGA";
    const HEAVY_AA: &[u8] = b"QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYGISWVRQAPGQGLEWMGWISAYNGNTNYAQKLQGRVTMTTDTSTSTAYMELRSLRSDDTAVYYCAR";

    fn make_result() -> NumberingResult {
        number_chain(0, HEAVY_NT, Some(HEAVY_AA), ChainType::Heavy).unwrap()
    }

    #[test]
    fn test_per_nt_row_count() {
        let r = make_result();
        let mut buf = Vec::new();
        write_per_nt_rows(&mut buf, &r).unwrap();
        let text = std::str::from_utf8(&buf).unwrap();
        let lines: Vec<_> = text.lines().collect();
        assert_eq!(lines.len(), ChainType::Heavy.max_nt_positions() as usize);
    }

    #[test]
    fn test_per_nt_header_columns() {
        let header_cols: Vec<_> = PER_NT_HEADER.trim().split('\t').collect();
        assert_eq!(
            header_cols,
            [
                "sequence_id",
                "chain",
                "nt_position",
                "aho_position",
                "codon_position",
                "nucleotide",
                "amino_acid"
            ]
        );
    }

    #[test]
    fn test_per_codon_row_count() {
        let r = make_result();
        let mut buf = Vec::new();
        write_per_codon_rows(&mut buf, &r).unwrap();
        let text = std::str::from_utf8(&buf).unwrap();
        let lines: Vec<_> = text.lines().collect();
        assert_eq!(lines.len(), ChainType::Heavy.max_aho_position() as usize);
    }

    #[test]
    fn test_per_codon_codon_length() {
        let r = make_result();
        let mut buf = Vec::new();
        write_per_codon_rows(&mut buf, &r).unwrap();
        let text = std::str::from_utf8(&buf).unwrap();
        for line in text.lines() {
            let cols: Vec<_> = line.split('\t').collect();
            assert_eq!(cols.len(), 5, "Wrong column count: {}", line);
            assert_eq!(cols[3].len(), 3, "Codon not 3 chars: {}", cols[3]);
        }
    }

    #[test]
    fn test_write_results_per_nt() {
        let r = make_result();
        let mut buf = Vec::new();
        write_results(&mut buf, &[r], OutputFormat::PerNucleotide).unwrap();
        let text = std::str::from_utf8(&buf).unwrap();
        let mut lines = text.lines();
        assert_eq!(lines.next().unwrap(), PER_NT_HEADER.trim());
        assert_eq!(
            text.lines().count() - 1,
            ChainType::Heavy.max_nt_positions() as usize
        );
    }

    #[test]
    fn test_wide_header_column_count() {
        let header = wide_header(&[ChainType::Heavy]);
        let cols: Vec<_> = header.trim().split('\t').collect();
        // sequence_id + 447 nt positions
        assert_eq!(cols.len(), 1 + ChainType::Heavy.max_nt_positions() as usize);
        assert_eq!(cols[0], "sequence_id");
        assert_eq!(cols[1], "H1");
    }

    #[test]
    fn test_wide_row_column_count() {
        let r = make_result();
        let mut buf = Vec::new();
        write_wide_row(&mut buf, &r, &[ChainType::Heavy]).unwrap();
        let text = std::str::from_utf8(&buf).unwrap();
        let cols: Vec<_> = text.trim().split('\t').collect();
        assert_eq!(cols.len(), 1 + ChainType::Heavy.max_nt_positions() as usize);
    }
}
