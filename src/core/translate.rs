/// Standard genetic code: maps 3-byte codon (uppercase DNA) to single-letter AA.
/// Stop codons return b'*'. Unknown bases return b'X'.
pub fn translate_codon(codon: &[u8; 3]) -> u8 {
    // Normalise to uppercase
    let a = codon[0].to_ascii_uppercase();
    let b = codon[1].to_ascii_uppercase();
    let c = codon[2].to_ascii_uppercase();

    match (a, b, c) {
        // Phenylalanine (F)
        (b'T', b'T', b'T') | (b'T', b'T', b'C') => b'F',
        // Leucine (L)
        (b'T', b'T', b'A') | (b'T', b'T', b'G') => b'L',
        (b'C', b'T', _) => b'L',
        // Isoleucine (I)
        (b'A', b'T', b'T') | (b'A', b'T', b'C') | (b'A', b'T', b'A') => b'I',
        // Methionine (M) / Start
        (b'A', b'T', b'G') => b'M',
        // Valine (V)
        (b'G', b'T', _) => b'V',
        // Serine (S)
        (b'T', b'C', _) => b'S',
        (b'A', b'G', b'T') | (b'A', b'G', b'C') => b'S',
        // Proline (P)
        (b'C', b'C', _) => b'P',
        // Threonine (T)
        (b'A', b'C', _) => b'T',
        // Alanine (A)
        (b'G', b'C', _) => b'A',
        // Tyrosine (Y)
        (b'T', b'A', b'T') | (b'T', b'A', b'C') => b'Y',
        // Stop (*): TAA, TAG, TGA
        (b'T', b'A', b'A') | (b'T', b'A', b'G') => b'*',
        (b'T', b'G', b'A') => b'*',
        // Cysteine (C)
        (b'T', b'G', b'T') | (b'T', b'G', b'C') => b'C',
        // Tryptophan (W)
        (b'T', b'G', b'G') => b'W',
        // Histidine (H)
        (b'C', b'A', b'T') | (b'C', b'A', b'C') => b'H',
        // Glutamine (Q)
        (b'C', b'A', b'A') | (b'C', b'A', b'G') => b'Q',
        // Arginine (R)
        (b'C', b'G', _) => b'R',
        (b'A', b'G', b'A') | (b'A', b'G', b'G') => b'R',
        // Asparagine (N)
        (b'A', b'A', b'T') | (b'A', b'A', b'C') => b'N',
        // Lysine (K)
        (b'A', b'A', b'A') | (b'A', b'A', b'G') => b'K',
        // Aspartate (D)
        (b'G', b'A', b'T') | (b'G', b'A', b'C') => b'D',
        // Glutamate (E)
        (b'G', b'A', b'A') | (b'G', b'A', b'G') => b'E',
        // Glycine (G)
        (b'G', b'G', _) => b'G',
        // Unknown / ambiguous
        _ => b'X',
    }
}

/// Translate a nucleotide slice (starting at `offset`) into an amino acid Vec.
/// Stops at the first in-frame stop codon (exclusive) or when fewer than 3 nt remain.
/// Returns (aa_sequence, codon_count).
pub fn translate_frame(nt: &[u8], offset: usize) -> Vec<u8> {
    let mut aa = Vec::with_capacity((nt.len().saturating_sub(offset)) / 3 + 1);
    let mut i = offset;
    while i + 3 <= nt.len() {
        let codon: [u8; 3] = [nt[i], nt[i + 1], nt[i + 2]];
        let residue = translate_codon(&codon);
        if residue == b'*' {
            break;
        }
        aa.push(residue);
        i += 3;
    }
    aa
}

/// Try all three reading frames and return each translation (may include empty if < 3 nt).
pub fn translate_all_frames(nt: &[u8]) -> [Vec<u8>; 3] {
    [
        translate_frame(nt, 0),
        translate_frame(nt, 1),
        translate_frame(nt, 2),
    ]
}

/// Return the byte offset into `nt` where `aa_seq` is found as an in-frame translation.
/// Returns `None` if the AA sequence does not appear in any frame.
pub fn find_frame(nt: &[u8], aa_seq: &[u8]) -> Option<usize> {
    for offset in 0..3 {
        let translated = translate_frame(nt, offset);
        // Check if aa_seq is a subsequence (contiguous substring) of the translation
        if contains_subslice(&translated, aa_seq) {
            // Find the exact start position within the translated sequence
            if let Some(aa_start) = find_subslice_pos(&translated, aa_seq) {
                // nt start = offset + aa_start * 3
                return Some(offset + aa_start * 3);
            }
        }
    }
    None
}

fn contains_subslice(haystack: &[u8], needle: &[u8]) -> bool {
    if needle.is_empty() {
        return true;
    }
    if needle.len() > haystack.len() {
        return false;
    }
    haystack.windows(needle.len()).any(|w| w == needle)
}

fn find_subslice_pos(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    haystack
        .windows(needle.len())
        .position(|w| w == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_known_codons() {
        assert_eq!(translate_codon(b"ATG"), b'M');
        assert_eq!(translate_codon(b"TGG"), b'W');
        assert_eq!(translate_codon(b"TAA"), b'*');
        assert_eq!(translate_codon(b"TAG"), b'*');
        assert_eq!(translate_codon(b"TGA"), b'*');
        assert_eq!(translate_codon(b"GGG"), b'G');
        assert_eq!(translate_codon(b"TTT"), b'F');
        assert_eq!(translate_codon(b"TTC"), b'F');
    }

    #[test]
    fn test_translate_frame_stops_at_stop() {
        // ATG (M) TGG (W) TAA (*) GGG -> should give MW
        let nt = b"ATGTGGTAAGGG";
        let aa = translate_frame(nt, 0);
        assert_eq!(aa, b"MW");
    }

    #[test]
    fn test_translate_all_frames() {
        let nt = b"CATGGG"; // frame0: CAT(H) GGG(G), frame1: ATG(M) GG?, frame2: TGG(W) G?
        let frames = translate_all_frames(nt);
        assert_eq!(frames[0], b"HG");
        assert_eq!(frames[1], b"M");
        assert_eq!(frames[2], b"W");
    }

    #[test]
    fn test_find_frame() {
        // Encode QVQL in frame 0
        // Q=CAG, V=GTG, Q=CAG, L=CTG
        let nt = b"CAGGTGCAGCTG";
        let aa = b"QVQL";
        let offset = find_frame(nt, aa);
        assert_eq!(offset, Some(0));
    }

    #[test]
    fn test_find_frame_offset() {
        // One extra leading byte
        let nt = b"XCAGGTGCAGCTG";
        let aa = b"QVQL";
        let offset = find_frame(nt, aa);
        assert_eq!(offset, Some(1));
    }

    #[test]
    fn test_find_frame_not_found() {
        let nt = b"AAAAAAAAAA";
        let aa = b"QVQL";
        let offset = find_frame(nt, aa);
        assert_eq!(offset, None);
    }
}
