/// Chain type for heavy and light chains
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChainType {
    Heavy,  // H
    Kappa,  // K
    Lambda, // L
}

impl ChainType {
    pub fn as_str(self) -> &'static str {
        match self {
            ChainType::Heavy => "H",
            ChainType::Kappa => "K",
            ChainType::Lambda => "L",
        }
    }

    /// Maximum Aho amino acid position for this chain type
    pub fn max_aho_position(self) -> u16 {
        match self {
            ChainType::Heavy => 149,
            ChainType::Kappa => 148,
            ChainType::Lambda => 148,
        }
    }

    /// Maximum nucleotide positions (max_aho * 3)
    pub fn max_nt_positions(self) -> u16 {
        self.max_aho_position() * 3
    }
}

impl std::fmt::Display for ChainType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Gene segment type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeneType {
    V,
    D,
    J,
}

/// A single Aho-numbered amino acid position
#[derive(Debug, Clone, Copy)]
pub struct AhoResidue {
    pub position: u16, // 1-based Aho position
    pub amino_acid: u8, // single-byte AA code (uppercase ASCII)
}

/// A single numbered nucleotide position in the output
#[derive(Debug, Clone, Copy)]
pub struct NtPosition {
    pub nt_position: u16,   // absolute 1-based position in the numbered frame
    pub aho_position: u16,  // 1-based Aho position (same for all 3 nt in a codon)
    pub codon_position: u8, // 1, 2, or 3
    pub nucleotide: u8,     // ASCII byte: A/T/G/C or b'-'
    pub amino_acid: u8,     // ASCII byte: single-letter AA or b'-'
}

/// Result for a single chain of one sequence
#[derive(Debug, Clone)]
pub struct NumberingResult {
    pub sequence_id: u32,
    pub chain: ChainType,
    pub germline_id: String,
    pub positions: Vec<NtPosition>,
}

/// Result for a single sequence (may have heavy + light)
#[derive(Debug, Default)]
pub struct SequenceResult {
    pub results: Vec<NumberingResult>,
}

/// Batch result
#[derive(Debug, Default)]
pub struct BatchResult {
    pub results: Vec<NumberingResult>,
    pub errors: Vec<crate::error::NumberingError>,
}
