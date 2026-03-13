# IggNition

<img src="assets/logo.svg" width="150" height="150" align="right" />

**Ultra-fast nucleotide-level Aho numbering for antibody variable domains.**

IggNition replaces ANARCI/HMMER with a purpose-built Rust aligner against pre-numbered human germline V/D/J genes. It assigns [Aho scheme](https://doi.org/10.1006/jmbi.2001.4662) positions to every nucleotide in an antibody variable domain sequence, producing a fixed-length coordinate frame suitable for repertoire analysis and AbLLM training.

| | ANARCI (Python/HMMER) | **IgNition** |
|---|---|---|
| Throughput (single core) | ~4 seq/s | **>50,000 seq/s** |
| Language | Python + HMMER | Rust |
| Output | amino-acid level | **nucleotide level** |
| Parallelism | external (shell) | built-in (Rayon) |

## Installation

```bash
pip install iggnition
```

Pre-compiled wheels are available for Linux (x86\_64, aarch64), macOS (Apple Silicon + Intel), and Windows.

---

## Quick Start

### Single sequence

```python
import iggnition

df = iggnition.run(
    nt_seq="CAGGTGCAGCTGGTGCAGTCTGGAGCT...",
    aa_seq="QVQLVQSGAE...",
)
print(df)
# shape: (447, 7)  ← 149 Aho positions × 3 nucleotides for a heavy chain
# ┌─────────────┬───────┬─────────────┬──────────────┬────────────────┬────────────┬───────────┐
# │ sequence_id ┆ chain ┆ nt_position ┆ aho_position ┆ codon_position ┆ nucleotide ┆ amino_acid│
# │ u32         ┆ str   ┆ u32         ┆ u32          ┆ u32            ┆ str        ┆ str       │
```

### AIRR-format DataFrame (single chain)

```python
import polars as pl
import iggnition

df = pl.read_csv("airr_table.tsv", separator="\t")

results, errors = iggnition.run(
    df,
    nt_col="sequence",
    aa_col="sequence_aa",
    locus_col="locus",   # e.g. "IGH", "IGK", "IGL"
)
```

### Paired heavy + light (PairPlex-style)

```python
results, errors = iggnition.run(
    df,
    paired=True,
    nt_col_heavy="sequence:0",
    aa_col_heavy="sequence_aa:0",
    nt_col_light="sequence:1",
    aa_col_light="sequence_aa:1",
)
```

### File paths

```python
# FASTA → DataFrame
results, errors = iggnition.run("input.fasta")

# Parquet → Parquet
iggnition.run("input.parquet", output="numbered.parquet")

# TSV → TSV
iggnition.run("input.tsv", output="numbered.tsv")
```

---

## Output Formats

`iggnition.run(df)` returns a `(results, errors)` tuple for batch input.

### Per-nucleotide (default)

One row per nucleotide position — the most detailed output:

```python
results, errors = iggnition.run(df)
# shape: (n_sequences × 447, 7) for heavy chains
# columns: sequence_id, chain, nt_position, aho_position, codon_position, nucleotide, amino_acid
```

| Column | Type | Description |
|--------|------|-------------|
| `sequence_id` | u32 | Row index from input |
| `chain` | str | `H`, `K`, or `L` |
| `nt_position` | u32 | Absolute nucleotide position (1-based) |
| `aho_position` | u32 | Aho amino acid position (1-based) |
| `codon_position` | u32 | Position within codon (1, 2, or 3) |
| `nucleotide` | str | `A`/`T`/`G`/`C` or `-` for gaps |
| `amino_acid` | str | Single-letter AA or `-` for gaps |

### Per-codon

One row per Aho position:

```python
results, errors = iggnition.run(df, per_codon=True)
# columns: sequence_id, chain, aho_position, codon, amino_acid
```

### Wide format — recommended for AI/ML

Wide format is the natural input shape for machine learning models: each sequence becomes a single fixed-length row with one column per nucleotide position (`H1`…`H447` for heavy chains, `L1`…`L444` for light chains). Both Kappa and Lambda light chains use the `L` prefix.

```python
results, errors = iggnition.run(df, wide=True)
```

Output shape: `(n_sequences, 1 + 447 + 444)` for paired heavy + light, with columns:

```
seq_name | H1 | H2 | … | H447 | L1 | L2 | … | L444
```

The first column is always `seq_name`. If you supply `name_col`, it takes the values from that column; otherwise it contains the integer row index.

#### Human-readable vs numeric encoding

By default (`human_readable=True`) nucleotide positions are stored as single-character strings — what you would expect:

```python
results, errors = iggnition.run(df, wide=True)
# H1 column: "C", "A", "G", "-", ...  (pl.Utf8)
```

For large-scale work (millions of sequences) switch to the raw ASCII byte encoding with `human_readable=False`. Each position becomes a `pl.UInt8` value (65=A, 84=T, 71=G, 67=C, 45=gap), which is the most memory-efficient representation and enables direct numerical operations without any decoding step:

```python
results, errors = iggnition.run(df, wide=True, human_readable=False)
# H1 column: 67, 65, 71, 45, ...  (pl.UInt8)

# Decode a column back to characters when needed:
results.with_columns(pl.col("H1").map_elements(chr, return_dtype=pl.Utf8))

# Or work directly with the numbers — e.g. one-hot encode, pass to PyTorch, etc.
```

> **Memory note:** The `UInt8` wide path uses a compact Rust byte-array representation (~2.5 GB for 2.4 M paired antibodies) vs. ~400+ GB peak via a naive pivot on per-nucleotide rows. Always prefer `wide=True` over post-hoc pivoting.

#### Per-chain wide format

By default paired results are merged so each antibody occupies a single row (H columns + L columns). Set `per_chain=True` to keep one row per chain:

```python
results, errors = iggnition.run(df, wide=True, per_chain=True)
# One row per chain; a "chain" column indicates H or L.
# H rows have L columns filled with null, and vice versa.
```

#### Propagating sequence names

Use `name_col` to carry an identifier from the input DataFrame into the results as `seq_name`:

```python
results, errors = iggnition.run(
    df,
    wide=True,
    name_col="clone_id",   # any column in df
)
# First column: seq_name (values from df["clone_id"])
# Then: H1…H447, L1…L444
```

---

## CLI

```bash
# FASTA → TSV (stdout)
iggnition run input.fasta

# FASTA → TSV (file)
iggnition run input.fasta output.tsv

# Parquet → Parquet
iggnition run input.parquet output.parquet

# TSV (AIRR) → TSV, per-codon
iggnition run input.tsv output.tsv --per-codon

# Wide format, 8 threads, with progress bar
iggnition run input.fasta output.tsv --wide --threads 8 --verbose
```

### CLI options

| Option | Default | Description |
|--------|---------|-------------|
| `--per-codon` | off | One row per codon instead of per nucleotide |
| `--wide` | off | Pivot to wide format |
| `--nt-col` | `sequence` | NT column name (TSV/Parquet) |
| `--aa-col` | `sequence_aa` | AA column name |
| `--locus-col` | `locus` | Chain/locus column name |
| `--nt-col-heavy` | `sequence:0` | Heavy chain NT column (paired mode) |
| `--aa-col-heavy` | `sequence_aa:0` | Heavy chain AA column (paired mode) |
| `--nt-col-light` | `sequence:1` | Light chain NT column (paired mode) |
| `--aa-col-light` | `sequence_aa:1` | Light chain AA column (paired mode) |
| `--no-aa` | off | Auto-translate NT (fallback, emits warning) |
| `--threads` | all cores | Rayon worker threads |
| `--verbose` / `-v` | off | Show progress bar and summary statistics |

---

## Aho Position Ranges

| Chain | Max Aho position | Max NT columns |
|-------|-----------------|----------------|
| H (heavy) | 149 | 447 (`H1`…`H447`) |
| K (kappa) | 148 | 444 (`L1`…`L444`) |
| L (lambda) | 148 | 444 (`L1`…`L444`) |

Kappa and Lambda share the `L` prefix in wide format — they map to the same coordinate frame.

---

## How It Works

1. All human V/D/J germline genes are pre-numbered with Aho positions and **embedded in the binary at compile time** — no external database files.
2. For each query: find the closest germline via Needleman-Wunsch alignment (amino acid level), then transfer Aho positions from the germline to the query.
3. Map each occupied Aho position to its codon from the nucleotide sequence: position `N` → nucleotides `(N-1)*3+1`, `(N-1)*3+2`, `(N-1)*3+3`. Unoccupied positions become gaps (`-`).
4. **Rayon** parallelises across sequences in the batch; the Python GIL is never held during alignment.

---

## Building from Source

```bash
git clone https://github.com/bnemoz/ignition
cd ignition
pip install .
```

---

## License

MIT. If you use IgNition in published research, please cite accordingly.

## References

- Honegger, A. & Plückthun, A. (2001). Yet another numbering scheme for immunoglobulin variable domains. *J Mol Biol*, 309(3), 657–670.
- Dunbar, J. & Deane, C.M. (2016). ANARCI: antigen receptor numbering and receptor classification. *Bioinformatics*, 32(2), 298–300.
