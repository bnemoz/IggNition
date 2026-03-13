"""iggnition — ultra-fast antibody variable domain Aho numbering.

Public API: :func:`run`

Example usage::

    import iggnition

    # Single sequence
    df = iggnition.run(nt_seq="CAGGTG...", aa_seq="QV...")

    # Polars DataFrame (AIRR format)
    results_df, errors_df = iggnition.run(
        df,
        nt_col="sequence",
        aa_col="sequence_aa",
        locus_col="locus",
    )

    # Paired heavy + light (PairPlex-style)
    results_df, errors_df = iggnition.run(
        df,
        paired=True,
        nt_col_heavy="sequence:0",
        aa_col_heavy="sequence_aa:0",
        nt_col_light="sequence:1",
        aa_col_light="sequence_aa:1",
    )

    # File path (FASTA / TSV / Parquet)
    results_df, errors_df = iggnition.run("input.fasta")
    iggnition.run("input.parquet", output="numbered.parquet")
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Union

import polars as pl

from iggnition._ignition import _run_batch as _rust_run_batch
from iggnition._ignition import _run_batch_wide as _rust_run_batch_wide
from iggnition._ignition import _run_fasta as _rust_run_fasta

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


def _to_wide(df: pl.DataFrame, per_chain: bool = False) -> pl.DataFrame:
    """Pivot to wide format with positional columns.

    Column naming: Heavy → ``H1``, ``H2``, …; Kappa or Lambda → ``L1``, ``L2``, …

    Args:
        per_chain: If ``True``, keep one row per chain (index by ``sequence_id``
            and ``chain``).  If ``False`` (default), merge H and L for the same
            ``sequence_id`` into a single row.
    """
    # Normalise chain label: K → L so both light-chain subtypes share the same
    # column prefix and (when per_chain=True) the same pivot-index value.
    df = df.with_columns(
        pl.when(pl.col("chain") == "K")
        .then(pl.lit("L"))
        .otherwise(pl.col("chain"))
        .alias("chain")
    )
    # Build positional column name: H1, L1, …
    df = df.with_columns(
        (pl.col("chain") + pl.col("nt_position").cast(pl.Utf8)).alias("_col")
    )
    index = ["sequence_id", "chain"] if per_chain else ["sequence_id"]
    result = df.pivot(
        index=index,
        on="_col",
        values="nucleotide",
        aggregate_function="first",
    )
    # Drop helper column if Polars carried it through
    if "_col" in result.columns:
        result = result.drop("_col")
    return result


# ─── Wide fast-path (Rust → compact bytes → Polars, bypasses per-nt pivot) ────

# These constants mirror ChainType::max_nt_positions() on the Rust side.
_WIDE_H_COLS = 447  # Heavy: 149 Aho positions × 3
_WIDE_L_COLS = 444  # Kappa/Lambda: 148 Aho positions × 3
_WIDE_H_NAMES = [f"H{i + 1}" for i in range(_WIDE_H_COLS)]
_WIDE_L_NAMES = [f"L{i + 1}" for i in range(_WIDE_L_COLS)]


def _build_wide_df(
    res: dict,
    per_chain: bool = False,
    human_readable: bool = True,
) -> pl.DataFrame:
    """Build a wide DataFrame from the compact Rust wide-format dict.

    By default nucleotide columns (``H1``…``H447``, ``L1``…``L444``) are stored
    as ``pl.UInt8`` (ASCII byte values: 65=A, 84=T, 71=G, 67=C, 45=gap), which
    is the most memory-efficient representation for downstream numerical work.

    When ``human_readable=True`` the same numpy byte arrays are decoded to
    single-character ``pl.Utf8`` strings (A/T/G/C/–) using a fully vectorised
    ``astype('U1')`` conversion — no Python loop per element.

    Memory (UInt8 path): ~2.5 GB for 2.4 M paired antibodies vs ~400+ GB via
    the per-nucleotide pivot path.
    """
    import numpy as np

    h_ids: list = res["H_sequence_id"]
    h_bytes: bytes = res["H_nucleotides"]
    l_ids: list = res["L_sequence_id"]
    l_bytes: bytes = res["L_nucleotides"]

    n_h = len(h_ids)
    n_l = len(l_ids)

    def _arr_to_df(ids, raw_bytes, n_rows, n_cols, col_names, uint8_schema):
        if n_rows == 0:
            return pl.DataFrame(
                schema={"sequence_id": pl.UInt32, **uint8_schema}
            )
        arr = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(n_rows, n_cols)
        if human_readable:
            # view as single-byte strings then promote to Unicode — fully vectorised
            arr = arr.view("S1").reshape(n_rows, n_cols).astype("U1")
        id_df = pl.DataFrame({"sequence_id": pl.Series(ids, dtype=pl.UInt32)})
        return pl.concat([id_df, pl.from_numpy(arr, schema=col_names)], how="horizontal")

    h_schema = {c: pl.UInt8 for c in _WIDE_H_NAMES}
    l_schema = {c: pl.UInt8 for c in _WIDE_L_NAMES}
    df_h = _arr_to_df(h_ids, h_bytes, n_h, _WIDE_H_COLS, _WIDE_H_NAMES, h_schema)
    df_l = _arr_to_df(l_ids, l_bytes, n_l, _WIDE_L_COLS, _WIDE_L_NAMES, l_schema)

    if per_chain:
        df_h = df_h.with_columns(pl.lit("H").alias("chain"))
        df_l = df_l.with_columns(pl.lit("L").alias("chain"))
        return pl.concat([df_h, df_l], how="diagonal_relaxed")
    else:
        return df_h.join(df_l, on="sequence_id", how="outer", coalesce=True)


def _apply_format(
    df: pl.DataFrame,
    per_codon: bool,
    wide: bool,
    per_chain: bool = False,
) -> pl.DataFrame:
    if per_codon:
        df = _to_per_codon(df)
    if wide:
        df = _to_wide(df, per_chain=per_chain)
    return df


# ─── Name propagation ─────────────────────────────────────────────────────────

def _attach_name(
    results_df: pl.DataFrame,
    source_df: pl.DataFrame,
    name_col: str,
) -> pl.DataFrame:
    """Left-join ``name_col`` from ``source_df`` onto ``results_df`` by ``sequence_id``."""
    name_map = pl.DataFrame(
        {
            "sequence_id": pl.Series(
                range(len(source_df)), dtype=pl.UInt32
            ),
            "name": source_df[name_col].cast(pl.Utf8),
        }
    )
    return results_df.join(name_map, on="sequence_id", how="left")


# ─── Progress bar ─────────────────────────────────────────────────────────────

def _make_progress_bar(total: int, verbose: bool):
    """Return ``(pbar, callback)`` when verbose and tqdm is available, else ``(None, None)``."""
    if not verbose or total == 0:
        return None, None
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return None, None
    pbar = tqdm(total=total, desc="numbering", unit=" seq")

    def _callback(done: int) -> None:
        delta = done - pbar.n
        if delta > 0:
            pbar.update(delta)

    return pbar, _callback


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
    progress_callback=None,
) -> tuple[dict, dict]:
    n = len(df)
    seq_ids = list(range(n))
    nts = _col_list(df, nt_col)
    aas = _col_list(df, aa_col)
    if chain_override:
        chains: list = [chain_override] * n
    else:
        chains = _col_list(df, locus_col)
    return _rust_run_batch(seq_ids, nts, aas, chains, num_threads, progress_callback)


def _run_paired(
    df: pl.DataFrame,
    nt_col_heavy: str,
    aa_col_heavy: str,
    nt_col_light: str,
    aa_col_light: str,
    num_threads: Optional[int],
    progress_callback=None,
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

    return _rust_run_batch(seq_ids, nts, aas, chains, num_threads, progress_callback)


def _run_generic_wide(
    df: pl.DataFrame,
    nt_col: str,
    aa_col: Optional[str],
    locus_col: Optional[str],
    chain_override: Optional[str],
    num_threads: Optional[int],
    progress_callback=None,
) -> tuple[dict, dict]:
    n = len(df)
    seq_ids = list(range(n))
    nts = _col_list(df, nt_col)
    aas = _col_list(df, aa_col)
    chains: list = [chain_override] * n if chain_override else _col_list(df, locus_col)
    return _rust_run_batch_wide(seq_ids, nts, aas, chains, num_threads, progress_callback)


def _run_paired_wide(
    df: pl.DataFrame,
    nt_col_heavy: str,
    aa_col_heavy: str,
    nt_col_light: str,
    aa_col_light: str,
    num_threads: Optional[int],
    progress_callback=None,
) -> tuple[dict, dict]:
    n = len(df)
    heavy_nts = _col_list(df, nt_col_heavy)
    heavy_aas = _col_list(df, aa_col_heavy)
    light_nts = _col_list(df, nt_col_light)
    light_aas = _col_list(df, aa_col_light)

    seq_ids, nts, aas, chains = [], [], [], []
    for i in range(n):
        if heavy_nts[i]:
            seq_ids.append(i); nts.append(heavy_nts[i])
            aas.append(heavy_aas[i]); chains.append("H")
        if light_nts[i]:
            seq_ids.append(i); nts.append(light_nts[i])
            aas.append(light_aas[i]); chains.append(None)

    if not seq_ids:
        empty = {
            "H_sequence_id": [], "H_nucleotides": b"",
            "L_sequence_id": [], "L_nucleotides": b"",
        }
        return empty, {"sequence_id": [], "chain": [], "error": []}

    return _rust_run_batch_wide(seq_ids, nts, aas, chains, num_threads, progress_callback)


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
    per_chain: bool = False,
    human_readable: bool = True,
    name_col: Optional[str] = None,
    output: Optional[Union[str, Path]] = None,
    num_threads: Optional[int] = None,
    verbose: bool = False,
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
        wide: Pivot to wide format — positional columns ``H1``…``H447``,
            ``L1``…``L444`` (Kappa and Lambda both use the ``L`` prefix).
            Columns are ``pl.UInt8`` (ASCII byte values: 65=A, 84=T, 71=G,
            67=C, 45=gap).  Use ``pl.col("H1").map_elements(chr)`` to decode
            to characters.  Implemented via a compact Rust path that is ~50×
            more memory-efficient than the per-nucleotide pivot.
        per_chain: When ``wide=True``, keep one row per chain instead of merging
            H and L for the same ``sequence_id`` into one row.
        human_readable: When ``wide=True``, decode positional columns from
            ``pl.UInt8`` ASCII bytes to single-character ``pl.Utf8`` strings
            (A/T/G/C/–).  Uses a fully vectorised numpy ``astype('U1')``
            conversion — no per-element Python loop.  Defaults to ``True``
            for readability.  Set to ``False`` when processing very large
            datasets (millions of sequences) where the ``UInt8`` representation
            saves significant memory and enables faster numerical operations.
        name_col: Column in the input DataFrame to use as the row identifier.
            In wide format, becomes the first column ``seq_name`` (replacing the
            integer ``sequence_id``).  Both H and L results for a paired row
            receive the same name.  When omitted, ``sequence_id`` is renamed to
            ``seq_name`` in wide format.
        output: Write results to this file path.  Format inferred from extension.
            When set, the function *still* returns the DataFrames.
        num_threads: Number of Rayon worker threads.  ``None`` → all cores.
        verbose: Reserved for future progress reporting.  Currently a no-op in
            the Python API.

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
        results_df = _apply_format(results_df, per_codon, wide, per_chain)
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
                per_codon=per_codon, wide=wide, per_chain=per_chain,
                name_col=name_col, output=output, num_threads=num_threads,
                verbose=verbose,
            )

        elif suffix in (".tsv", ".txt"):
            df_in = pl.read_csv(path, separator="\t")
            return run(
                df_in,
                nt_col=nt_col, aa_col=aa_col, locus_col=locus_col,
                chain=chain, paired=paired,
                nt_col_heavy=nt_col_heavy, aa_col_heavy=aa_col_heavy,
                nt_col_light=nt_col_light, aa_col_light=aa_col_light,
                per_codon=per_codon, wide=wide, per_chain=per_chain,
                name_col=name_col, output=output, num_threads=num_threads,
                verbose=verbose,
            )

        elif suffix == ".csv":
            df_in = pl.read_csv(path)
            return run(
                df_in,
                nt_col=nt_col, aa_col=aa_col, locus_col=locus_col,
                chain=chain, paired=paired,
                nt_col_heavy=nt_col_heavy, aa_col_heavy=aa_col_heavy,
                nt_col_light=nt_col_light, aa_col_light=aa_col_light,
                per_codon=per_codon, wide=wide, per_chain=per_chain,
                name_col=name_col, output=output, num_threads=num_threads,
                verbose=verbose,
            )

        else:
            raise ValueError(
                f"Unrecognised file extension '{path.suffix}'. "
                "Supported: .fasta/.fa/.fna, .tsv/.txt, .csv, .parquet/.pq"
            )

        results_df = _apply_format(results_df, per_codon, wide, per_chain)
        if output:
            _write_output(results_df, Path(output))
        return results_df, errors_df

    # ── DataFrame input ───────────────────────────────────────────────────────
    if input is not None:
        df, was_pandas = _to_polars(input)

        use_paired = paired or (
            nt_col_heavy in df.columns and nt_col_light in df.columns
        )

        n_total = 2 * len(df) if use_paired else len(df)
        _pbar, _progress_cb = _make_progress_bar(n_total, verbose)

        # ── Wide fast-path: bypass per-nucleotide intermediate ────────────────
        # For wide=True (and not per_codon), use the compact Rust wide API
        # which returns flat byte arrays instead of 2B per-nucleotide rows.
        use_wide_fast = wide and not per_codon
        try:
            if use_wide_fast:
                if use_paired:
                    raw_wide, err_dict = _run_paired_wide(
                        df, nt_col_heavy, aa_col_heavy, nt_col_light, aa_col_light,
                        num_threads, _progress_cb,
                    )
                else:
                    raw_wide, err_dict = _run_generic_wide(
                        df, nt_col, aa_col, locus_col, chain, num_threads, _progress_cb
                    )
            else:
                if use_paired:
                    res_dict, err_dict = _run_paired(
                        df, nt_col_heavy, aa_col_heavy, nt_col_light, aa_col_light,
                        num_threads, _progress_cb,
                    )
                else:
                    res_dict, err_dict = _run_generic(
                        df, nt_col, aa_col, locus_col, chain, num_threads, _progress_cb
                    )
        finally:
            if _pbar is not None:
                _pbar.close()

        errors_df = _build_errors_df(err_dict)

        if use_wide_fast:
            results_df = _build_wide_df(raw_wide, per_chain, human_readable)
            if name_col and name_col in df.columns:
                results_df = _attach_name(results_df, df, name_col)
                # Promote name to front as seq_name; drop integer sequence_id
                other_cols = [
                    c for c in results_df.columns if c not in ("sequence_id", "name")
                ]
                results_df = results_df.select(
                    pl.col("name").alias("seq_name"),
                    *[pl.col(c) for c in other_cols],
                )
            else:
                results_df = results_df.rename({"sequence_id": "seq_name"})
        else:
            results_df = _build_results_df(res_dict)
            if name_col and name_col in df.columns:
                results_df = _attach_name(results_df, df, name_col)
            results_df = _apply_format(results_df, per_codon, wide=False, per_chain=per_chain)

        if output:
            _write_output(results_df, Path(output))

        if was_pandas:
            return results_df.to_pandas(), errors_df.to_pandas()
        return results_df, errors_df

    raise ValueError(
        "Provide either `input` (DataFrame or file path) "
        "or `nt_seq` for single-sequence mode."
    )
