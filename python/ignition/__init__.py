"""ignition — ultra-fast antibody variable domain Aho numbering.

Public API: :func:`run`

Example usage::

    import ignition

    # Single sequence
    df = ignition.run(nt_seq="CAGGTG...", aa_seq="QV...")

    # Polars DataFrame (AIRR format)
    results_df, errors_df = ignition.run(
        df,
        nt_col="sequence",
        aa_col="sequence_aa",
        locus_col="locus",
    )

    # Paired heavy + light (PairPlex-style)
    results_df, errors_df = ignition.run(
        df,
        paired=True,
        nt_col_heavy="sequence:0",
        aa_col_heavy="sequence_aa:0",
        nt_col_light="sequence:1",
        aa_col_light="sequence_aa:1",
    )

    # File path (FASTA / TSV / Parquet)
    results_df, errors_df = ignition.run("input.fasta")
    ignition.run("input.parquet", output="numbered.parquet")
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Union

import polars as pl

from ignition._ignition import _run_batch as _rust_run_batch
from ignition._ignition import _run_fasta as _rust_run_fasta

try:
    import pandas as _pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False

__all__ = ["run"]

# ─── Schema helpers ────────────────────────────────────────────────────────────

_RESULTS_SCHEMA = {
    "sequence_id": pl.UInt32,
    "chain": pl.Utf8,
    "nt_position": pl.UInt32,
    "aho_position": pl.UInt32,
    "codon_position": pl.UInt32,
    "nucleotide": pl.Utf8,
    "amino_acid": pl.Utf8,
}

_ERRORS_SCHEMA = {
    "sequence_id": pl.UInt32,
    "chain": pl.Utf8,
    "error": pl.Utf8,
}


def _build_results_df(res: dict) -> pl.DataFrame:
    # nucleotide and amino_acid arrive as raw bytes — decode to char lists
    nt_bytes: bytes = res["nucleotide"]
    aa_bytes: bytes = res["amino_acid"]
    data = {
        "sequence_id": res["sequence_id"],
        "chain": res["chain"],
        "nt_position": res["nt_position"],
        "aho_position": res["aho_position"],
        "codon_position": res["codon_position"],
        "nucleotide": list(nt_bytes.decode("ascii")),
        "amino_acid": list(aa_bytes.decode("ascii")),
    }
    return pl.DataFrame(data, schema=_RESULTS_SCHEMA)


def _build_errors_df(err: dict) -> pl.DataFrame:
    return pl.DataFrame(err, schema=_ERRORS_SCHEMA)


# ─── Output format transforms ──────────────────────────────────────────────────

def _to_per_codon(df: pl.DataFrame) -> pl.DataFrame:
    """Collapse 3 nucleotide rows per Aho position into one codon row."""
    return (
        df.sort(["sequence_id", "chain", "aho_position", "codon_position"])
        .group_by(["sequence_id", "chain", "aho_position"], maintain_order=True)
        .agg(
            pl.col("nucleotide").str.join("").alias("codon"),
            pl.col("amino_acid").first(),
        )
    )


def _to_wide(df: pl.DataFrame) -> pl.DataFrame:
    """Pivot to one row per sequence with positional columns."""
    # Add a combined column name: chain_nt_position
    df = df.with_columns(
        (pl.col("chain") + "_nt_" + pl.col("nt_position").cast(pl.Utf8)).alias("_col")
    )
    return df.pivot(
        index="sequence_id",
        on="_col",
        values="nucleotide",
        aggregate_function="first",
    )


def _apply_format(df: pl.DataFrame, per_codon: bool, wide: bool) -> pl.DataFrame:
    if per_codon:
        df = _to_per_codon(df)
    if wide:
        df = _to_wide(df)
    return df


# ─── DataFrame dispatch helpers ────────────────────────────────────────────────

def _to_polars(df) -> tuple[pl.DataFrame, bool]:
    """Convert to Polars; return (polars_df, was_pandas)."""
    if _PANDAS_AVAILABLE and isinstance(df, _pd.DataFrame):
        return pl.from_pandas(df), True
    if isinstance(df, pl.DataFrame):
        return df, False
    raise TypeError(
        f"Expected a polars.DataFrame or pandas.DataFrame, got {type(df).__name__}"
    )


def _col_list(df: pl.DataFrame, col: Optional[str]) -> list:
    """Return column values as a Python list, or a list of None if absent."""
    if col and col in df.columns:
        return df[col].cast(pl.Utf8).to_list()
    return [None] * len(df)


def _run_generic(
    df: pl.DataFrame,
    nt_col: str,
    aa_col: Optional[str],
    locus_col: Optional[str],
    chain_override: Optional[str],
    num_threads: Optional[int],
) -> tuple[dict, dict]:
    n = len(df)
    seq_ids = list(range(n))
    nts = _col_list(df, nt_col)
    aas = _col_list(df, aa_col)
    if chain_override:
        chains: list = [chain_override] * n
    else:
        chains = _col_list(df, locus_col)
    return _rust_run_batch(seq_ids, nts, aas, chains, num_threads)


def _run_paired(
    df: pl.DataFrame,
    nt_col_heavy: str,
    aa_col_heavy: str,
    nt_col_light: str,
    aa_col_light: str,
    num_threads: Optional[int],
) -> tuple[dict, dict]:
    n = len(df)
    heavy_nts = _col_list(df, nt_col_heavy)
    heavy_aas = _col_list(df, aa_col_heavy)
    light_nts = _col_list(df, nt_col_light)
    light_aas = _col_list(df, aa_col_light)

    # Build inputs: heavy entries (chain="H") + light entries (chain=None, auto K/L)
    seq_ids, nts, aas, chains = [], [], [], []
    for i in range(n):
        if heavy_nts[i]:
            seq_ids.append(i); nts.append(heavy_nts[i])
            aas.append(heavy_aas[i]); chains.append("H")
        if light_nts[i]:
            seq_ids.append(i); nts.append(light_nts[i])
            aas.append(light_aas[i]); chains.append(None)

    if not seq_ids:
        return (
            {"sequence_id": [], "chain": [], "nt_position": [],
             "aho_position": [], "codon_position": [],
             "nucleotide": b"", "amino_acid": b""},
            {"sequence_id": [], "chain": [], "error": []},
        )

    return _rust_run_batch(seq_ids, nts, aas, chains, num_threads)


# ─── Output writer ─────────────────────────────────────────────────────────────

def _write_output(df: pl.DataFrame, path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix in (".parquet", ".pq"):
        df.write_parquet(path)
    elif suffix == ".csv":
        df.write_csv(path)
    else:
        # TSV / .txt / fallback
        df.write_csv(path, separator="\t")


# ─── Public API ────────────────────────────────────────────────────────────────

def run(
    input=None,
    *,
    nt_seq: Optional[str] = None,
    aa_seq: Optional[str] = None,
    nt_col: str = "sequence",
    aa_col: Optional[str] = "sequence_aa",
    locus_col: Optional[str] = "locus",
    chain: Optional[str] = None,
    paired: bool = False,
    nt_col_heavy: str = "sequence:0",
    aa_col_heavy: str = "sequence_aa:0",
    nt_col_light: str = "sequence:1",
    aa_col_light: str = "sequence_aa:1",
    per_codon: bool = False,
    wide: bool = False,
    output: Optional[Union[str, Path]] = None,
    num_threads: Optional[int] = None,
) -> Union[pl.DataFrame, tuple[pl.DataFrame, pl.DataFrame]]:
    """Number antibody variable domain sequences using the Aho scheme.

    Dispatch logic:

    * **Single-sequence** (``nt_seq`` provided): returns a :class:`~polars.DataFrame`
      of numbered positions for that one sequence.
    * **DataFrame** (``input`` is ``pl.DataFrame`` or ``pd.DataFrame``): returns
      ``(results_df, errors_df)``.  Pandas input returns Pandas output.
    * **File path** (``input`` is ``str``/``Path``): detects format from extension
      (``.fasta``, ``.tsv``, ``.csv``, ``.parquet``) and returns
      ``(results_df, errors_df)``.

    Args:
        input: DataFrame or file path.  Mutually exclusive with ``nt_seq``.
        nt_seq: Nucleotide sequence string (single-sequence mode).
        aa_seq: Amino acid sequence string.  ``None`` → fallback auto-translate
            (emits ``UserWarning``).
        nt_col: NT column name in the DataFrame / TSV / Parquet.
        aa_col: AA column name (``None`` to disable).
        locus_col: Chain/locus column (e.g. ``"locus"`` → ``"IGH"``/``"IGK"``/
            ``"IGL"``).  Ignored if ``chain`` is set.
        chain: Force a chain type for single-sequence mode (``"H"``, ``"K"``,
            ``"L"``).  ``None`` → auto-detect.
        paired: If ``True``, treat each DataFrame row as a paired H+L record and
            use ``nt_col_heavy`` / ``nt_col_light`` for extraction.  Auto-detected
            when both columns are present.
        nt_col_heavy: NT column for the heavy chain (paired mode).
        aa_col_heavy: AA column for the heavy chain (paired mode).
        nt_col_light: NT column for the light chain (paired mode).
        aa_col_light: AA column for the light chain (paired mode).
        per_codon: Return per-codon format (one row per Aho position) instead of
            per-nucleotide.
        wide: Pivot to wide format — one row per sequence, positional columns.
        output: Write results to this file path.  Format inferred from extension.
            When set, the function *still* returns the DataFrames.
        num_threads: Number of Rayon worker threads.  ``None`` → all cores.

    Returns:
        * Single-sequence mode: ``pl.DataFrame``
        * Batch mode: ``(results_df, errors_df)``
    """
    # ── Single sequence ───────────────────────────────────────────────────────
    if nt_seq is not None:
        if aa_seq is None:
            warnings.warn(
                "No AA sequence supplied; auto-detecting reading frame. "
                "This is not the designed use case.",
                UserWarning,
                stacklevel=2,
            )
        res_dict, _ = _rust_run_batch([0], [nt_seq], [aa_seq], [chain], num_threads)
        results_df = _build_results_df(res_dict)
        results_df = _apply_format(results_df, per_codon, wide)
        if output:
            _write_output(results_df, Path(output))
        return results_df

    # ── File path ─────────────────────────────────────────────────────────────
    if isinstance(input, (str, Path)):
        path = Path(input)
        suffix = path.suffix.lower()

        if suffix in (".fasta", ".fa", ".fna", ".fas"):
            res_dict, err_dict = _rust_run_fasta(str(path), paired, num_threads)
            results_df = _build_results_df(res_dict)
            errors_df = _build_errors_df(err_dict)

        elif suffix in (".parquet", ".pq"):
            df_in = pl.read_parquet(path)
            return run(
                df_in,
                nt_col=nt_col, aa_col=aa_col, locus_col=locus_col,
                chain=chain, paired=paired,
                nt_col_heavy=nt_col_heavy, aa_col_heavy=aa_col_heavy,
                nt_col_light=nt_col_light, aa_col_light=aa_col_light,
                per_codon=per_codon, wide=wide,
                output=output, num_threads=num_threads,
            )

        elif suffix in (".tsv", ".txt"):
            df_in = pl.read_csv(path, separator="\t")
            return run(
                df_in,
                nt_col=nt_col, aa_col=aa_col, locus_col=locus_col,
                chain=chain, paired=paired,
                nt_col_heavy=nt_col_heavy, aa_col_heavy=aa_col_heavy,
                nt_col_light=nt_col_light, aa_col_light=aa_col_light,
                per_codon=per_codon, wide=wide,
                output=output, num_threads=num_threads,
            )

        elif suffix == ".csv":
            df_in = pl.read_csv(path)
            return run(
                df_in,
                nt_col=nt_col, aa_col=aa_col, locus_col=locus_col,
                chain=chain, paired=paired,
                nt_col_heavy=nt_col_heavy, aa_col_heavy=aa_col_heavy,
                nt_col_light=nt_col_light, aa_col_light=aa_col_light,
                per_codon=per_codon, wide=wide,
                output=output, num_threads=num_threads,
            )

        else:
            raise ValueError(
                f"Unrecognised file extension '{path.suffix}'. "
                "Supported: .fasta/.fa/.fna, .tsv/.txt, .csv, .parquet/.pq"
            )

        results_df = _apply_format(results_df, per_codon, wide)
        if output:
            _write_output(results_df, Path(output))
        return results_df, errors_df

    # ── DataFrame input ───────────────────────────────────────────────────────
    if input is not None:
        df, was_pandas = _to_polars(input)

        use_paired = paired or (
            nt_col_heavy in df.columns and nt_col_light in df.columns
        )

        if use_paired:
            res_dict, err_dict = _run_paired(
                df, nt_col_heavy, aa_col_heavy, nt_col_light, aa_col_light,
                num_threads,
            )
        else:
            res_dict, err_dict = _run_generic(
                df, nt_col, aa_col, locus_col, chain, num_threads
            )

        results_df = _build_results_df(res_dict)
        errors_df = _build_errors_df(err_dict)
        results_df = _apply_format(results_df, per_codon, wide)

        if output:
            _write_output(results_df, Path(output))

        if was_pandas:
            return results_df.to_pandas(), errors_df.to_pandas()
        return results_df, errors_df

    raise ValueError(
        "Provide either `input` (DataFrame or file path) "
        "or `nt_seq` for single-sequence mode."
    )
