use thiserror::Error;

#[derive(Debug, Error, Clone)]
pub enum IgnitionError {
    #[error("Frame resolution failed: no valid reading frame found for sequence")]
    FrameResolutionFailed,

    #[error("Germline identification failed: no germline matched with sufficient score")]
    GermlineNotFound,

    #[error("Aho position transfer failed: {0}")]
    AhoTransferFailed(String),

    #[error("Invalid nucleotide sequence: {0}")]
    InvalidSequence(String),

    #[error("Sequence too short: length {0}, minimum required {1}")]
    SequenceTooShort(usize, usize),

    #[error("CDR3 too long: {0} residues, maximum Aho positions {1}")]
    Cdr3TooLong(usize, usize),

    #[error("IO error: {0}")]
    Io(String),
}

/// Error collected during batch processing (does not abort the batch)
#[derive(Debug, Clone)]
pub struct NumberingError {
    pub sequence_id: u32,
    pub chain: crate::core::types::ChainType,
    pub message: String,
}

impl NumberingError {
    pub fn new(sequence_id: u32, chain: crate::core::types::ChainType, err: IgnitionError) -> Self {
        Self {
            sequence_id,
            chain,
            message: err.to_string(),
        }
    }
}
