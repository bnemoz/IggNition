# IgNition

## Project Overview

IgNition is a high-performance Rust library with Python bindings for antibody variable domain nucleotide numbering using the Aho scheme. It replaces ANARCI/HMMER entirely with a custom aligner against pre-numbered human germline V/D/J genes.

**Author:** Benjamin Nemoz (github.com/bnemoz/ignition)
**Target:** >50,000 sequences/second on a single core

## Core Algorithm

1. All human V/D/J germline genes for heavy (H) and light (K, L) chains are pre-numbered with Aho positions and embedded in the binary at compile time.
2. For each query: identify the closest germline via alignment (Needleman-Wunsch or Viterbi — choose the best fit), then transfer Aho positions from the germline to the query.
3. Aho positions map 1:1 to codons → each Aho position expands to 3 nucleotide positions: `(aho_pos - 1) * 3 + {1, 2, 3}`.
4. Unoccupied Aho positions are represented as gaps (`-` / `---`).

## Key Constants

- `AHO_MAX_POSITIONS`: H=149, K=148, L=148
- Species: human only
- Chains: H, K, L (heavy, kappa, lambda)
- Germline genes: V, D, and J segments

## Input Flexibility

### CLI (`ignition run`)
- FASTA files (paired nt + aa or nt-only)
- Parquet files
- TSV/CSV files (AIRR-annotated)

### Python API (`import ignition`)
- `ignition.run()` accepts:
  - Polars DataFrame
  - Pandas DataFrame
  - Single sequence (str for nt + str for aa)
  - File paths (str/Path)

### AA Sequence Handling
- Primary mode: user supplies both nt and aa sequences
- Fallback mode (with warning): aa omitted → auto-translate nt in 3 frames, select best frame. This is NOT the designed use case and must emit a warning.

## Output

### CLI
- Default output: TSV to stdout or file
- If input is parquet → output is parquet
- Explicit output path supported

### Python API
- Default: Polars DataFrame
- Output format configurable via args (DataFrame or file)

### Output Formats
- **Per-nucleotide (default):** columns = `sequence_id`, `chain`, `nt_position`, `aho_position`, `codon_position`, `nucleotide`, `amino_acid`
- **Per-codon:** columns = `sequence_id`, `chain`, `aho_position`, `codon`, `amino_acid`
- **Wide format:** one row per sequence, positional columns

## CLI Usage

```bash
ignition run input.fasta > numbered.tsv
ignition run input.parquet output.parquet
ignition run paired.tsv ./path/to/output/num.tsv
```

## Python Usage

```python
import ignition

# Single sequence
result = ignition.run(nt_seq="CAGGTG...", aa_seq="QV...")

# DataFrame
result_df = ignition.run(df, nt_col="sequence:0", aa_col="sequence_aa:0")

# File
ignition.run("input.fasta", output="numbered.tsv")
```

## Installation

```bash
pip install ignition
```

Package ships with pre-compiled Rust binary (maturin/PyO3 build).

## Architecture

```
ignition/
├── Cargo.toml
├── pyproject.toml
├── src/
│   ├── lib.rs              # Entry point, PyO3 module definition
│   ├── align.rs            # Core alignment engine (NW or Viterbi)
│   ├── aho.rs              # Aho numbering logic and position mapping
│   ├── germline.rs         # Embedded germline database + lookup
│   ├── translate.rs        # 3-frame translation and frame resolution
│   ├── io/
│   │   ├── mod.rs
│   │   ├── fasta.rs        # FASTA parser
│   │   ├── parquet.rs      # Parquet I/O
│   │   └── tsv.rs          # TSV/CSV I/O (AIRR format)
│   └── cli.rs              # CLI entry point
├── python/
│   └── ignition/
│       ├── __init__.py     # `ignition.run()` API
│       └── py.typed
├── data/
│   └── germlines/          # Human V/D/J germline sequences (Aho-numbered)
├── tests/
└── benches/
```

## Tech Stack

- **Rust** for core alignment and numbering
- **PyO3 + maturin** for Python bindings
- **clap** for CLI
- **rayon** for Rust-level parallelism
- **arrow2 or polars-core** for Parquet I/O in Rust

## Design Principles

- Zero Python in the hot path — all numbering logic in Rust
- Germline data embedded at compile time (include_bytes! or build.rs)
- Batch processing as the primary mode, single-sequence as convenience wrapper
- Errors collected, never crash the batch
- Progress reporting via callback to Python (tqdm compatible)

## Development Context

This project evolved from a Python prototype using ANARCI + Polars. ANARCI was the bottleneck (83x slower than frame resolution). At 350 seq/s on 87 cores with ANARCI, processing 5M sequences from PairPlex takes ~4 hours. IgNition targets >50k seq/s single-threaded, making the same job feasible in minutes with Rayon parallelism.
