# IgNition — Design Document

## 1. Problem Statement

Antibody variable domain sequences need to be numbered according to the Aho scheme so that structurally equivalent positions across different antibodies always have the same index. This enables:

- Aligned comparison of antibody sequences at the nucleotide level
- SHM (somatic hypermutation) analysis at specific codon positions
- ML feature extraction with fixed-dimension representations
- CDR/FR boundary identification at consistent positions

The existing tool (ANARCI) wraps HMMER, a general-purpose HMM search engine. It searches a full database of Ig/TCR profiles for every query — massive overkill when the use case is exclusively human antibody V/D/J genes. Benchmarking showed ANARCI at ~350 seq/s across 87 CPU cores, with HMMER accounting for 83x more compute than all other steps combined.

IgNition replaces this with a purpose-built aligner that knows upfront that all queries are human antibody variable domains.

---

## 2. Aho Numbering Scheme

The Aho scheme (Honegger & Plückthun, 2001) assigns structurally equivalent positions across all Ig/TCR variable domains using a fixed-length numbering derived from structural superposition.

### Position Ranges

| Chain | Max Aho Position | Max NT Positions |
|-------|-----------------|-----------------|
| H     | 149             | 447             |
| K     | 148             | 444             |
| L     | 148             | 444             |

### Nucleotide Mapping

For an occupied Aho position `N`, nucleotide positions are:
- `(N - 1) * 3 + 1` (codon position 1)
- `(N - 1) * 3 + 2` (codon position 2)
- `(N - 1) * 3 + 3` (codon position 3)

Unoccupied positions are filled with gap characters (`-` per nucleotide, `---` per codon).

### Regions (for future annotation)

Aho positions map to FR/CDR regions. The exact boundaries are fixed per chain type:

| Region | H (Aho)     | K/L (Aho)   |
|--------|-------------|-------------|
| FR1    | 1–26        | 1–26        |
| CDR1   | 27–40       | 27–40       |
| FR2    | 41–57       | 41–57       |
| CDR2   | 58–77       | 58–77       |
| FR3    | 78–108      | 78–108      |
| CDR3   | 109–138     | 109–137     |
| FR4    | 139–149     | 138–148     |

> Note: Verify these boundaries against the original Honegger & Plückthun paper during implementation. They may need adjustment.

---

## 3. Algorithm

### 3.1 Germline Database (Compile-Time)

All human immunoglobulin germline genes are obtained from IMGT/GENE-DB:

- **V genes:** IGHV, IGKV, IGLV (functional alleles)
- **D genes:** IGHD
- **J genes:** IGHJ, IGKJ, IGLJ

Each germline amino acid sequence is Aho-numbered once (using ANARCI as a one-time bootstrap, or manually from IMGT/Aho alignment tables). The result is a lookup table:

```
germline_id → Vec<(aho_position, amino_acid)>
```

This is embedded in the Rust binary at compile time.

### 3.2 Query Processing Pipeline

For each query sequence:

```
Input: nt_seq, aa_seq (or nt_seq only in fallback mode)
                    │
                    ▼
    ┌───────────────────────────────┐
    │  Frame Resolution             │
    │  - If aa_seq provided:        │
    │    translate nt in 3 frames,  │
    │    find frame containing aa   │
    │  - If aa_seq omitted:         │
    │    translate all 3 frames,    │
    │    align each to germlines,   │
    │    pick best-scoring frame    │
    │  Output: trimmed nt_seq,      │
    │          aa_seq               │
    └───────────────┬───────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │  Germline Identification      │
    │  - Align aa_seq against all   │
    │    germline V gene profiles   │
    │  - Select highest-scoring     │
    │    germline                   │
    │  Output: best germline,       │
    │          alignment            │
    └───────────────┬───────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │  Aho Position Transfer        │
    │  - Walk alignment: for each   │
    │    aligned pair (query_aa,    │
    │    germline_aa), assign the   │
    │    germline's Aho position    │
    │    to the query residue       │
    │  - Handle insertions in CDRs  │
    │    (query longer than         │
    │    germline)                  │
    │  Output: Vec<(aho_pos, aa)>   │
    └───────────────┬───────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │  Nucleotide Expansion         │
    │  - Map each occupied Aho pos  │
    │    to its codon from nt_seq   │
    │  - Fill unoccupied positions  │
    │    with gaps                  │
    │  Output: numbered nt records  │
    └───────────────────────────────┘
```

### 3.3 Alignment Strategy

Two candidate algorithms — decide during implementation based on performance benchmarks:

**Option A: Needleman-Wunsch (global pairwise alignment)**
- Simpler to implement
- Align query aa_seq against each candidate germline
- Use BLOSUM62 or identity matrix with affine gap penalties
- Score each germline, pick the best
- Transfer Aho positions from the germline side of the alignment

**Option B: Viterbi (profile HMM)**
- Build a profile HMM from the Aho scheme itself (one model per chain type)
- States correspond to Aho positions; insert states handle CDR variability
- Single Viterbi pass gives both the alignment and the Aho positions directly
- More elegant, potentially faster (one model vs N germlines)

**Recommendation:** Start with Needleman-Wunsch — faster to implement, easier to debug. Profile the result. If germline search is the bottleneck (unlikely — there are only ~300 germlines and queries are ~120 aa), switch to Viterbi.

### 3.4 CDR3 Handling

CDR3 is the critical challenge — it's not germline-encoded and varies dramatically in length. Strategy:

- V-gene alignment covers FR1 through FR3 (Aho 1–108)
- J-gene alignment covers FR4 (Aho 139–149 for H, 138–148 for K/L)
- CDR3 residues (between V and J matches) are assigned Aho positions sequentially starting from 109
- If CDR3 is longer than the available Aho positions (109–138 for H = 30 positions), excess residues get insertion positions
- D-gene identification is informational but not required for numbering

---

## 4. Input Specification

### 4.1 CLI Interface

```bash
# FASTA → TSV (stdout)
ignition run input.fasta > numbered.tsv

# FASTA → TSV (file)
ignition run input.fasta output.tsv

# Parquet → Parquet
ignition run input.parquet output.parquet

# TSV (AIRR) → TSV
ignition run paired.tsv ./path/to/output/num.tsv
```

**CLI arguments:**

| Argument        | Description                                      | Default           |
|-----------------|--------------------------------------------------|-------------------|
| `input`         | Input file path (positional)                     | required          |
| `output`        | Output file path (positional, optional)          | stdout            |
| `--per-codon`   | One row per codon instead of per nucleotide      | false             |
| `--wide`        | Pivot to wide format                             | false             |
| `--nt-col`      | NT column name (TSV/Parquet)                     | `sequence`        |
| `--aa-col`      | AA column name (TSV/Parquet)                     | `sequence_aa`     |
| `--nt-col-heavy`| Heavy chain NT column (paired data)              | `sequence:0`      |
| `--aa-col-heavy`| Heavy chain AA column (paired data)              | `sequence_aa:0`   |
| `--nt-col-light`| Light chain NT column (paired data)              | `sequence:1`      |
| `--aa-col-light`| Light chain AA column (paired data)              | `sequence_aa:1`   |
| `--no-aa`       | Fallback mode: no AA supplied, auto-translate    | false             |
| `--threads`     | Number of threads for Rayon                      | all cores         |
| `--chunk-size`  | Sequences per processing chunk                   | 10000             |

### 4.2 Python API

```python
import ignition

# --- Single sequence ---
result = ignition.run(
    nt_seq="CAGGTGCAGCTG...",
    aa_seq="QVQL...",
)
# Returns: pl.DataFrame

# --- Single sequence, no AA (fallback) ---
result = ignition.run(
    nt_seq="CAGGTGCAGCTG...",
)
# Emits UserWarning, returns: pl.DataFrame

# --- Polars DataFrame ---
result_df, errors_df = ignition.run(
    df,
    nt_col_heavy="sequence:0",
    aa_col_heavy="sequence_aa:0",
    nt_col_light="sequence:1",
    aa_col_light="sequence_aa:1",
    per_codon=False,
    wide=False,
)
# Returns: tuple[pl.DataFrame, pl.DataFrame]

# --- Pandas DataFrame ---
result_df, errors_df = ignition.run(
    pandas_df,
    nt_col_heavy="sequence:0",
    aa_col_heavy="sequence_aa:0",
)
# Accepts pandas, returns Polars (or pandas if input was pandas)

# --- File path ---
ignition.run(
    "input.parquet",
    output="output.parquet",
    nt_col_heavy="sequence:0",
    aa_col_heavy="sequence_aa:0",
)
```

### 4.3 Input File Formats

**FASTA:**
- Paired: alternating records with naming convention (e.g., `>seq1_H`, `>seq1_L`) or interleaved nt/aa
- Specification TBD based on common FASTA conventions in antibody field

**Parquet:**
- Columns specified via CLI args or Python kwargs
- Supports paired (heavy + light in same row) or single-chain

**TSV/CSV:**
- AIRR-annotated format (tab-separated)
- Column names match AIRR standard or user-specified

---

## 5. Output Specification

### 5.1 Per-Nucleotide (Default)

| Column          | Type  | Description                           |
|-----------------|-------|---------------------------------------|
| sequence_id     | i32   | Row index from input                  |
| chain           | str   | "H", "K", or "L"                     |
| nt_position     | i16   | Absolute nucleotide position (1-based)|
| aho_position    | i16   | Aho amino acid position (1-based)     |
| codon_position  | i8    | Position within codon (1, 2, 3)       |
| nucleotide      | str   | A/T/G/C or "-" for gaps              |
| amino_acid      | str   | Single-letter AA or "-" for gaps      |

### 5.2 Per-Codon

| Column          | Type  | Description                           |
|-----------------|-------|---------------------------------------|
| sequence_id     | i32   | Row index from input                  |
| chain           | str   | "H", "K", or "L"                     |
| aho_position    | i16   | Aho amino acid position (1-based)     |
| codon           | str   | 3-letter codon or "---" for gaps      |
| amino_acid      | str   | Single-letter AA or "-" for gaps      |

### 5.3 Wide Format

One row per sequence. Column names: `{chain}_nt_{position}` or `{chain}_{aho_position}` for per-codon.

### 5.4 Error DataFrame

| Column          | Type  | Description                           |
|-----------------|-------|---------------------------------------|
| sequence_id     | i64   | Row index from input                  |
| chain           | str   | "H" or "L"                           |
| error           | str   | Human-readable error description      |

---

## 6. Rust Architecture

```
src/
├── lib.rs                 # PyO3 module: exposes `run()` to Python
├── cli.rs                 # clap-based CLI: `ignition run`
├── core/
│   ├── mod.rs
│   ├── align.rs           # Needleman-Wunsch (or Viterbi) implementation
│   ├── aho.rs             # Aho position assignment and nt expansion
│   ├── frame.rs           # 3-frame translation and frame resolution
│   ├── translate.rs       # Codon table and fast translation
│   └── germline.rs        # Germline database (compiled-in)
├── io/
│   ├── mod.rs
│   ├── fasta.rs           # FASTA reader
│   ├── parquet.rs         # Parquet read/write (arrow2 or polars)
│   └── tsv.rs             # TSV/CSV read/write
├── batch.rs               # Batch orchestration + Rayon parallelism
└── error.rs               # Error types
```

### 6.1 Key Data Structures

```rust
/// A single Aho-numbered amino acid position
struct AhoPosition {
    position: u16,       // 1-based Aho position
    amino_acid: u8,      // single byte AA code
}

/// A germline gene with pre-computed Aho numbering
struct Germline {
    id: &'static str,
    chain_type: ChainType,  // H, K, L
    gene_type: GeneType,    // V, D, J
    sequence: &'static [u8],
    aho_positions: &'static [(u16, u8)],
}

/// Result for a single chain
struct NumberingResult {
    sequence_id: u32,
    chain: ChainType,
    germline_id: String,
    positions: Vec<NtPosition>,
}

struct NtPosition {
    nt_position: u16,
    aho_position: u16,
    codon_position: u8,
    nucleotide: u8,
    amino_acid: u8,
}

/// Batch result
struct BatchResult {
    results: Vec<NumberingResult>,
    errors: Vec<NumberingError>,
}

struct NumberingError {
    sequence_id: u32,
    chain: ChainType,
    message: String,
}
```

### 6.2 Performance Strategy

- **Rayon** for batch-level parallelism (par_iter over sequences)
- **No allocation in hot loop:** pre-allocate output vectors per thread
- **SIMD-friendly alignment:** scoring matrix as flat array, row-major
- **Germline pre-filtering:** quick k-mer hash to shortlist candidate germlines before full alignment (optional optimization — profile first)
- **Compile-time germline data:** `include!()` or `lazy_static!` from generated Rust source

### 6.3 PyO3 Bindings

```rust
#[pyfunction]
#[pyo3(signature = (
    input = None,
    nt_seq = None,
    aa_seq = None,
    nt_col_heavy = "sequence:0",
    aa_col_heavy = "sequence_aa:0",
    nt_col_light = "sequence:1",
    aa_col_light = "sequence_aa:1",
    per_codon = false,
    wide = false,
    output = None,
))]
fn run(
    py: Python,
    input: Option<PyObject>,  // DataFrame, str path, or None
    nt_seq: Option<&str>,
    aa_seq: Option<&str>,
    nt_col_heavy: &str,
    aa_col_heavy: &str,
    nt_col_light: &str,
    aa_col_light: &str,
    per_codon: bool,
    wide: bool,
    output: Option<&str>,
) -> PyResult<PyObject> {
    // dispatch based on input type
}
```

---

## 7. Build and Distribution

### 7.1 Build System

- **maturin** for building Python wheels with Rust extensions
- `pyproject.toml` with maturin backend
- `Cargo.toml` for Rust dependencies

### 7.2 Dependencies (Rust)

| Crate          | Purpose                           |
|----------------|-----------------------------------|
| pyo3           | Python bindings                   |
| rayon          | Data parallelism                  |
| clap           | CLI argument parsing              |
| arrow2/polars  | Parquet I/O                       |
| csv            | TSV/CSV I/O                       |
| bio            | Sequence utilities (optional)     |
| serde          | Serialization                     |
| thiserror      | Error handling                    |
| indicatif      | CLI progress bars                 |

### 7.3 Installation

```bash
# From PyPI (pre-compiled wheels)
pip install ignition

# From source
git clone https://github.com/bnemoz/ignition
cd ignition
maturin develop --release
```

### 7.4 Wheel Targets

- manylinux2014_x86_64 (primary — server/HPC)
- macosx_arm64 (Apple Silicon dev machines)
- macosx_x86_64
- win_amd64 (optional)

---

## 8. Testing Strategy

### 8.1 Correctness

- **Ground truth from ANARCI:** number ~10,000 sequences with ANARCI (Aho scheme), compare output position-by-position with IgNition
- **Edge cases:** very short CDR3, very long CDR3 (>30 aa), truncated sequences, sequences with unusual germlines
- **Frame resolution:** sequences with 1–3 extra leading/trailing nucleotides, all three reading frames represented

### 8.2 Performance

- **Benchmark suite** (criterion.rs):
  - Single sequence numbering latency
  - Throughput at 1k, 10k, 100k, 1M sequences
  - Scaling with thread count
- **Target:** >50,000 seq/s single-threaded
- **Regression:** CI runs benchmarks, alerts on >10% regression

### 8.3 Integration

- Python API returns correct dtypes
- CLI produces valid TSV/Parquet
- Round-trip: number → unnumber → same sequence

---

## 9. Development Phases

### Phase 1: Core Engine
- [ ] Germline database compilation (ANARCI bootstrap)
- [ ] Codon table and 3-frame translation
- [ ] Needleman-Wunsch aligner
- [ ] Aho position transfer logic
- [ ] Nucleotide expansion
- [ ] Single-sequence API in Rust
- [ ] Unit tests against ANARCI ground truth

### Phase 2: Batch Processing
- [ ] Rayon parallelism
- [ ] Error collection (non-panicking batch)
- [ ] Progress callback
- [ ] Benchmark suite

### Phase 3: I/O Layer
- [ ] TSV/CSV reader and writer
- [ ] Parquet reader and writer
- [ ] FASTA reader

### Phase 4: Python Bindings
- [ ] PyO3 module with `run()` function
- [ ] DataFrame input/output (Polars + Pandas)
- [ ] Single-sequence convenience API
- [ ] Fallback mode (no AA) with warning
- [ ] maturin build configuration

### Phase 5: CLI
- [ ] clap argument parsing
- [ ] Format detection (FASTA/Parquet/TSV)
- [ ] Output routing (stdout/file)
- [ ] Progress bar (indicatif)

### Phase 6: Distribution
- [ ] pyproject.toml
- [ ] GitHub Actions CI (test + build wheels)
- [ ] PyPI publishing
- [ ] README with usage examples

---

## 10. Prior Art and References

- **Aho numbering:** Honegger, A. & Plückthun, A. (2001). Yet another numbering scheme for immunoglobulin variable domains. J Mol Biol, 309(3), 657-670.
- **ANARCI:** Dunbar, J. & Deane, C.M. (2016). ANARCI: antigen receptor numbering and receptor classification. Bioinformatics, 32(2), 298-300.
- **IMGT:** Lefranc, M.-P. et al. IMGT/GENE-DB for human germline genes.
- **Prototype performance:** ANARCI/HMMER achieved 350 seq/s on 87 cores (50k test). Frame resolution was 83x faster than ANARCI. IgNition targets >50k seq/s single-threaded.
