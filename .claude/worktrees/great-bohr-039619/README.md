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

## Region Annotations

IgNition ships a complete structural annotation layer accessible without running any alignment.  All positions are in the Aho coordinate system.

```python
import iggnition

# CDR / FR boundaries — Aho AA positions (1-based, inclusive)
iggnition.CDR_REGIONS["H"]["CDR3"]   # → (109, 137)
iggnition.CDR_REGIONS["L"]["CDR1"]   # → (26, 38)

# Same boundaries expressed as IgNition nt positions
iggnition.CDR_REGIONS_NT["H"]["CDR3"]  # → (325, 411)
iggnition.CDR_REGIONS_NT["L"]["FR4"]   # → (415, 444)
```

### Helper functions

```python
# Convert Aho AA position → 3 IgNition nt positions
iggnition.aho_to_nt(106)       # → (316, 317, 318) — the conserved FR3 Cys

# Convert an Aho range → (first_nt, last_nt)
iggnition.aho_range_to_nt(26, 38)  # → (76, 114)  — VH CDR1

# Which region does an Aho position fall in?
iggnition.region_of(43, "H")   # → "FR2"
iggnition.region_of(120, "H")  # → "CDR3"

# Convert a wide-format column name → (aho_pos, codon_pos)
iggnition.nt_col_to_aho("H76")   # → (26, 1) — first nt of Aho pos 26 (CDR1 start)
iggnition.nt_col_to_aho("L444")  # → (148, 3) — last nt of VL

# Build a list of column names for a given CDR/FR (for masking a wide DataFrame)
iggnition.cdr_mask("H", "CDR3")  # → ['H325', 'H326', ..., 'H411']
iggnition.cdr_mask("L", "CDR1")  # → ['L76', 'L77', ..., 'L114']
```

### Structural landmarks

Exact Aho positions for absolutely conserved residues:

```python
iggnition.LANDMARKS["H"]["Cys_disulfide_N"]["aho"]  # → 23  (IMGT 23 / Kabat ~22)
iggnition.LANDMARKS["H"]["Cys_disulfide_C"]["aho"]  # → 106 (IMGT 104 / Kabat 92)
iggnition.LANDMARKS["H"]["Trp_FR2"]["aho"]          # → 43  (IMGT 41 / Kabat 36)
```

These positions are derived directly from the germline database embedded in the binary — they are guaranteed to be consistent with the numbering IgNition assigns.

### Vernier zone and canonical positions

```python
# Vernier zone Aho positions (Foote & Winter 1992)
iggnition.VERNIER_ZONE_AHO["H"]   # → [2, 27, 29, 30, 43, 54, 55, 56, 66, 68, 70, 77, 92, 93]
iggnition.VERNIER_ZONE_AHO["L"]   # → [2, 25, 26, 27, 29, 33, 49, 51, 52, 73, 77, 78, 80]

# Full detail including Kabat cross-reference and notes
iggnition.VERNIER_ZONE["H"][71]   # → {"aho": 70, "kabat": 71, "region": "FR3", "note": "..."}

# Chothia canonical structural positions (Al-Lazikani et al. 1997)
iggnition.CHOTHIA_CANONICAL["H"]["CDR1_H1"]
iggnition.CHOTHIA_CANONICAL["L"]["CDR2_L2"]
```

### Numbering scheme cross-reference

```python
# Verified Aho ↔ IMGT ↔ Kabat correspondences at landmark positions
iggnition.NUMBERING_CROSSREF["H"]
# [{"aho": 23, "imgt": 23, "kabat_h": 22, "residue": "Cys (intrachain SS, N-terminal)"}, ...]
```

### CDR boundaries in Aho — quick reference

| Region | VH Aho | VH nt | VK/VL Aho | VK/VL nt |
|--------|--------|-------|-----------|----------|
| FR1 | 1–25 | 1–75 | 1–25 | 1–75 |
| CDR1 | 26–38 | 76–114 | 26–38 | 76–114 |
| FR2 | 39–49 | 115–147 | 39–49 | 115–147 |
| CDR2 | 50–64 | 148–192 | 50–66 | 148–198 |
| FR3 | 65–108 | 193–324 | 67–108 | 199–324 |
| CDR3 | 109–137 | 325–411 | 109–138 | 325–414 |
| FR4 | 138–149 | 412–447 | 139–148 | 415–444 |

VH CDR2 is shorter in the Aho frame (slots 62–64 for insertions) while VK/VL CDR2 has a larger insertion slot (59–66), reflecting the structural reality that H2 loops occupy different insertion positions than L2 loops.

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
git clone https://github.com/bnemoz/iggnition
cd iggnition
pip install .
```

---

## License

MIT. If you use IggNition in published research, please cite accordingly.

## References

- Honegger, A. & Plückthun, A. (2001). Yet another numbering scheme for immunoglobulin variable domains. *J Mol Biol*, 309(3), 657–670.
- Dunbar, J. & Deane, C.M. (2016). ANARCI: antigen receptor numbering and receptor classification. *Bioinformatics*, 32(2), 298–300.
