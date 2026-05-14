use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ignition::{
    batch::{BatchConfig, BatchInput},
    number_chain, number_chain_auto, run_batch, ChainType,
};

// Real IGHV1-18*01 VDJ sequence used as benchmark input
const HEAVY_NT: &[u8] = b"CAGGTGCAGCTGGTGCAGTCTGGAGCTGAGGTGAAGAAGCCTGGGGCCTCAGTGAAGGTCTCCTGCAAGGCTTCTGGTTACACCTTTACCAGCTATGGTATCAGCTGGGTGCGACAGGCCCCTGGACAAGGGCTTGAGTGGATGGGGTGGATCAGCGCTTACAATGGTAACACAAACTATGCACAGAAGCTCCAGGGCAGAGTCACGATGACCACAGACACATCCACGAGCACAGCCTACATGGAGCTGAGGAGCCTGAGATCTGACGACACGGCCGTGTATTACTGTGCGAGA";
const HEAVY_AA: &[u8] = b"QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYGISWVRQAPGQGLEWMGWISAYNGNTNYAQKLQGRVTMTTDTSTSTAYMELRSLRSDDTAVYYCAR";

fn bench_single_sequence(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_sequence");

    group.bench_function("number_chain_heavy_with_aa", |b| {
        b.iter(|| {
            number_chain(
                black_box(0),
                black_box(HEAVY_NT),
                black_box(Some(HEAVY_AA)),
                black_box(ChainType::Heavy),
            )
            .unwrap()
        })
    });

    group.bench_function("number_chain_auto_with_aa", |b| {
        b.iter(|| {
            number_chain_auto(black_box(0), black_box(HEAVY_NT), black_box(Some(HEAVY_AA))).unwrap()
        })
    });

    group.finish();
}

fn bench_batch_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_throughput");

    for &n in &[100usize, 1_000, 10_000] {
        let inputs: Vec<BatchInput> = (0..n as u32)
            .map(|id| BatchInput::heavy(id, HEAVY_NT.to_vec(), HEAVY_AA.to_vec()))
            .collect();

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::new("heavy_with_aa", n),
            &inputs,
            |b, inputs| {
                b.iter(|| run_batch::<fn(usize)>(black_box(inputs), &BatchConfig::default(), None))
            },
        );
    }

    group.finish();
}

fn bench_batch_thread_scaling(c: &mut Criterion) {
    let n = 1_000usize;
    let inputs: Vec<BatchInput> = (0..n as u32)
        .map(|id| BatchInput::heavy(id, HEAVY_NT.to_vec(), HEAVY_AA.to_vec()))
        .collect();

    let mut group = c.benchmark_group("thread_scaling");
    group.throughput(Throughput::Elements(n as u64));

    for &threads in &[1usize, 2, 4, 8] {
        let config = BatchConfig {
            num_threads: Some(threads),
            ..Default::default()
        };
        group.bench_with_input(
            BenchmarkId::new("threads", threads),
            &inputs,
            |b, inputs| b.iter(|| run_batch::<fn(usize)>(black_box(inputs), &config, None)),
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_single_sequence,
    bench_batch_throughput,
    bench_batch_thread_scaling,
);
criterion_main!(benches);
