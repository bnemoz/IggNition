use rayon::prelude::*;
use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::core::aho::number_sequence_ws;
use crate::core::align::AlignWorkspace;
use crate::core::frame::{resolve_with_aa, resolve_without_aa};
use crate::core::types::{BatchResult, ChainType, NumberingResult};
use crate::error::{IgnitionError, NumberingError};

// Per-thread NW workspace — avoids per-sequence heap allocation
thread_local! {
    static WORKSPACE: RefCell<AlignWorkspace> = RefCell::new(AlignWorkspace::new());
}

/// A single sequence to number in a batch.
pub struct BatchInput {
    /// External row identifier (passed through to output unchanged)
    pub sequence_id: u32,
    /// Nucleotide sequence (ASCII)
    pub nt_seq: Vec<u8>,
    /// Amino acid sequence. If `None`, fallback frame detection is used.
    pub aa_seq: Option<Vec<u8>>,
    /// Chain type. If `None`, the best chain is auto-detected.
    pub chain: Option<ChainType>,
}

impl BatchInput {
    pub fn new(
        sequence_id: u32,
        nt_seq: Vec<u8>,
        aa_seq: Option<Vec<u8>>,
        chain: Option<ChainType>,
    ) -> Self {
        Self {
            sequence_id,
            nt_seq,
            aa_seq,
            chain,
        }
    }

    pub fn heavy(sequence_id: u32, nt_seq: Vec<u8>, aa_seq: Vec<u8>) -> Self {
        Self::new(sequence_id, nt_seq, Some(aa_seq), Some(ChainType::Heavy))
    }

    pub fn kappa(sequence_id: u32, nt_seq: Vec<u8>, aa_seq: Vec<u8>) -> Self {
        Self::new(sequence_id, nt_seq, Some(aa_seq), Some(ChainType::Kappa))
    }

    pub fn lambda(sequence_id: u32, nt_seq: Vec<u8>, aa_seq: Vec<u8>) -> Self {
        Self::new(sequence_id, nt_seq, Some(aa_seq), Some(ChainType::Lambda))
    }
}

/// Process a single `BatchInput`, returning either a `NumberingResult` or a
/// `NumberingError` (non-panicking).
fn process_one(input: &BatchInput) -> Result<NumberingResult, NumberingError> {
    let result: Result<NumberingResult, IgnitionError> = (|| {
        let aa_ref = input.aa_seq.as_deref();

        // Resolve reading frame
        let frame = match aa_ref {
            Some(aa) => resolve_with_aa(&input.nt_seq, aa)?,
            None => {
                // Fallback — warn once per sequence via atomic flag would be ideal,
                // but for batch mode we emit no per-seq eprintln to avoid log spam.
                resolve_without_aa(&input.nt_seq, input.chain)?
            }
        };

        let nt_trimmed = &input.nt_seq[frame.nt_start..];

        // Determine chain
        let chain = match input.chain {
            Some(c) => c,
            None => {
                use crate::core::aho::identify_v_germline_ws;
                let chains = [ChainType::Heavy, ChainType::Kappa, ChainType::Lambda];
                WORKSPACE
                    .with(|ws| {
                        let mut ws = ws.borrow_mut();
                        chains
                            .iter()
                            .filter_map(|&c| {
                                identify_v_germline_ws(&frame.aa_seq, c, &mut ws)
                                    .map(|h| (c, h.score))
                            })
                            .max_by_key(|&(_, s)| s)
                            .map(|(c, _)| c)
                    })
                    .ok_or(IgnitionError::GermlineNotFound)?
            }
        };

        WORKSPACE.with(|ws| {
            let mut ws = ws.borrow_mut();
            number_sequence_ws(input.sequence_id, nt_trimmed, &frame.aa_seq, chain, &mut ws)
        })
    })();

    result.map_err(|e| {
        NumberingError::new(
            input.sequence_id,
            input.chain.unwrap_or(ChainType::Heavy),
            e,
        )
    })
}

/// Configuration for batch processing.
pub struct BatchConfig {
    /// Number of threads. `None` uses Rayon's default (all logical cores).
    pub num_threads: Option<usize>,
    /// How many sequences to process between progress callbacks.
    /// `None` → callback called after every sequence.
    pub progress_interval: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            num_threads: None,
            progress_interval: 1000,
        }
    }
}

/// Run numbering on a batch of sequences in parallel using Rayon.
///
/// Progress is reported via `progress_fn(completed_count)` approximately every
/// `config.progress_interval` sequences. The callback is called from worker
/// threads and must be `Send + Sync`.
///
/// Errors are collected rather than propagated — see `BatchResult.errors`.
pub fn run_batch<F>(
    inputs: &[BatchInput],
    config: &BatchConfig,
    progress_fn: Option<&F>,
) -> BatchResult
where
    F: Fn(usize) + Send + Sync,
{
    if inputs.is_empty() {
        return BatchResult::default();
    }

    let total = inputs.len();
    let counter = AtomicUsize::new(0);
    let interval = config.progress_interval.max(1);

    // Build a thread pool if a specific thread count was requested
    let run = |inputs: &[BatchInput]| -> Vec<Result<NumberingResult, NumberingError>> {
        inputs
            .par_iter()
            .map(|input| {
                let result = process_one(input);
                // Progress reporting: every `interval` completions
                let prev = counter.fetch_add(1, Ordering::Relaxed);
                let done = prev + 1;
                if let Some(f) = progress_fn {
                    // Call at each interval boundary and at completion
                    if done % interval == 0 || done == total {
                        f(done);
                    }
                }
                result
            })
            .collect()
    };

    let raw: Vec<Result<NumberingResult, NumberingError>> = match config.num_threads {
        Some(n) => {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .expect("failed to build thread pool");
            pool.install(|| run(inputs))
        }
        None => run(inputs),
    };

    let mut results = Vec::with_capacity(raw.len());
    let mut errors = Vec::new();
    for r in raw {
        match r {
            Ok(nr) => results.push(nr),
            Err(e) => errors.push(e),
        }
    }

    BatchResult { results, errors }
}

/// Convenience wrapper: run a batch and report fallback-mode warning once if any
/// input lacks an AA sequence.
pub fn run_batch_with_fallback_warning<F>(
    inputs: &[BatchInput],
    config: &BatchConfig,
    progress_fn: Option<&F>,
) -> BatchResult
where
    F: Fn(usize) + Send + Sync,
{
    let has_fallback = inputs.iter().any(|i| i.aa_seq.is_none());
    if has_fallback {
        eprintln!(
            "WARNING [iggnition]: {} sequence(s) have no AA sequence; \
             auto-detecting reading frames. This is not the designed use case.",
            inputs.iter().filter(|i| i.aa_seq.is_none()).count()
        );
    }
    run_batch(inputs, config, progress_fn)
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEAVY_NT: &[u8] = b"CAGGTGCAGCTGGTGCAGTCTGGAGCTGAGGTGAAGAAGCCTGGGGCCTCAGTGAAGGTCTCCTGCAAGGCTTCTGGTTACACCTTTACCAGCTATGGTATCAGCTGGGTGCGACAGGCCCCTGGACAAGGGCTTGAGTGGATGGGGTGGATCAGCGCTTACAATGGTAACACAAACTATGCACAGAAGCTCCAGGGCAGAGTCACGATGACCACAGACACATCCACGAGCACAGCCTACATGGAGCTGAGGAGCCTGAGATCTGACGACACGGCCGTGTATTACTGTGCGAGA";
    const HEAVY_AA: &[u8] = b"QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYGISWVRQAPGQGLEWMGWISAYNGNTNYAQKLQGRVTMTTDTSTSTAYMELRSLRSDDTAVYYCAR";

    fn make_heavy_input(id: u32) -> BatchInput {
        BatchInput::heavy(id, HEAVY_NT.to_vec(), HEAVY_AA.to_vec())
    }

    #[test]
    fn test_batch_single() {
        let inputs = vec![make_heavy_input(0)];
        let result = run_batch::<fn(usize)>(&inputs, &BatchConfig::default(), None);
        assert_eq!(result.errors.len(), 0);
        assert_eq!(result.results.len(), 1);
        assert_eq!(result.results[0].chain, ChainType::Heavy);
    }

    #[test]
    fn test_batch_multiple() {
        let n = 20;
        let inputs: Vec<_> = (0..n as u32).map(make_heavy_input).collect();
        let result = run_batch::<fn(usize)>(&inputs, &BatchConfig::default(), None);
        assert_eq!(
            result.errors.len(),
            0,
            "Errors: {:?}",
            result.errors.iter().map(|e| &e.message).collect::<Vec<_>>()
        );
        assert_eq!(result.results.len(), n);
        // Each result has the right number of positions
        for r in &result.results {
            assert_eq!(
                r.positions.len(),
                ChainType::Heavy.max_aho_position() as usize * 3
            );
        }
    }

    #[test]
    fn test_batch_progress_callback() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let n = 10u32;
        let inputs: Vec<_> = (0..n).map(make_heavy_input).collect();
        let progress_calls = Arc::new(AtomicUsize::new(0));
        let total_reported = Arc::new(AtomicUsize::new(0));

        let pc = Arc::clone(&progress_calls);
        let tr = Arc::clone(&total_reported);
        let config = BatchConfig {
            progress_interval: 1,
            ..Default::default()
        };
        let callback = move |done: usize| {
            pc.fetch_add(1, Ordering::Relaxed);
            tr.store(done, Ordering::Relaxed);
        };

        let result = run_batch(&inputs, &config, Some(&callback));
        assert_eq!(result.errors.len(), 0);
        // With interval=1 and 10 sequences, should get exactly 10 calls
        assert_eq!(progress_calls.load(Ordering::Relaxed), n as usize);
        assert_eq!(total_reported.load(Ordering::Relaxed), n as usize);
    }

    #[test]
    fn test_batch_error_collection() {
        // Mix valid and intentionally bad sequences
        let mut inputs = vec![make_heavy_input(0)];
        // Bad input: empty NT sequence
        inputs.push(BatchInput::new(
            1,
            vec![],
            Some(HEAVY_AA.to_vec()),
            Some(ChainType::Heavy),
        ));
        inputs.push(make_heavy_input(2));

        let result = run_batch::<fn(usize)>(&inputs, &BatchConfig::default(), None);
        assert_eq!(result.results.len(), 2, "Expected 2 successful results");
        assert_eq!(result.errors.len(), 1, "Expected 1 error");
        assert_eq!(result.errors[0].sequence_id, 1);
    }

    #[test]
    fn test_batch_sequence_ids_preserved() {
        let inputs: Vec<_> = (0..5u32)
            .map(|id| BatchInput::heavy(id * 10, HEAVY_NT.to_vec(), HEAVY_AA.to_vec()))
            .collect();
        let result = run_batch::<fn(usize)>(&inputs, &BatchConfig::default(), None);
        assert_eq!(result.errors.len(), 0);
        let mut ids: Vec<u32> = result.results.iter().map(|r| r.sequence_id).collect();
        ids.sort();
        assert_eq!(ids, vec![0, 10, 20, 30, 40]);
    }

    #[test]
    fn test_batch_custom_thread_count() {
        let inputs: Vec<_> = (0..10u32).map(make_heavy_input).collect();
        let config = BatchConfig {
            num_threads: Some(2),
            ..Default::default()
        };
        let result = run_batch::<fn(usize)>(&inputs, &config, None);
        assert_eq!(result.errors.len(), 0);
        assert_eq!(result.results.len(), 10);
    }

    #[test]
    fn test_batch_empty() {
        let result = run_batch::<fn(usize)>(&[], &BatchConfig::default(), None);
        assert!(result.results.is_empty());
        assert!(result.errors.is_empty());
    }
}
