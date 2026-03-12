"""
Python API tests for ignition.run().

Run with:
    PYTHONPATH=/path/to/IgNition/python pytest tests/test_python_api.py
"""

import pytest
import polars as pl

import ignition

# ─── Test fixtures ─────────────────────────────────────────────────────────────

HEAVY_NT = (
    "CAGGTGCAGCTGGTGCAGTCTGGAGCTGAGGTGAAGAAGCCTGGGGCCTCAGTGAAGGTCTCCTGCAAGGCT"
    "TCTGGTTACACCTTTACCAGCTATGGTATCAGCTGGGTGCGACAGGCCCCTGGACAAGGGCTTGAGTGGATGG"
    "GGTGGATCAGCGCTTACAATGGTAACACAAACTATGCACAGAAGCTCCAGGGCAGAGTCACGATGACCACAGA"
    "CACATCCACGAGCACAGCCTACATGGAGCTGAGGAGCCTGAGATCTGACGACACGGCCGTGTATTACTGTGCGAGA"
)
HEAVY_AA = (
    "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYGISWVRQAPGQGLEWMGWISAYNGNTNYAQKLQGRVTMTT"
    "DTSTSTAYMELRSLRSDDTAVYYCAR"
)

# ─── Single-sequence tests ─────────────────────────────────────────────────────


def test_single_sequence_returns_dataframe():
    result = ignition.run(nt_seq=HEAVY_NT, aa_seq=HEAVY_AA)
    assert isinstance(result, pl.DataFrame)


def test_single_sequence_schema():
    result = ignition.run(nt_seq=HEAVY_NT, aa_seq=HEAVY_AA)
    assert result.columns == [
        "sequence_id", "chain", "nt_position", "aho_position",
        "codon_position", "nucleotide", "amino_acid",
    ]


def test_single_sequence_row_count():
    # Heavy chain: 149 Aho positions × 3 nucleotides = 447 rows
    result = ignition.run(nt_seq=HEAVY_NT, aa_seq=HEAVY_AA)
    assert result.shape[0] == 447


def test_single_sequence_chain_is_heavy():
    result = ignition.run(nt_seq=HEAVY_NT, aa_seq=HEAVY_AA)
    chains = result["chain"].unique().to_list()
    assert chains == ["H"]


def test_single_sequence_no_aa_warns():
    with pytest.warns(UserWarning, match="auto-detecting"):
        result = ignition.run(nt_seq=HEAVY_NT)
    assert isinstance(result, pl.DataFrame)


def test_single_sequence_per_codon():
    result = ignition.run(nt_seq=HEAVY_NT, aa_seq=HEAVY_AA, per_codon=True)
    # Per-codon: 149 Aho positions → 149 rows
    assert result.shape[0] == 149
    assert "codon" in result.columns
    assert "amino_acid" in result.columns
    assert "nt_position" not in result.columns


def test_single_sequence_codon_lengths():
    result = ignition.run(nt_seq=HEAVY_NT, aa_seq=HEAVY_AA, per_codon=True)
    codon_lengths = result["codon"].str.len_chars()
    # Every codon should be exactly 3 characters
    assert (codon_lengths == 3).all()


# ─── DataFrame input tests ─────────────────────────────────────────────────────


def _make_df(n: int = 2) -> pl.DataFrame:
    return pl.DataFrame({
        "sequence": [HEAVY_NT] * n,
        "sequence_aa": [HEAVY_AA] * n,
        "locus": ["IGH"] * n,
    })


def test_dataframe_input_returns_tuple():
    results, errors = ignition.run(_make_df())
    assert isinstance(results, pl.DataFrame)
    assert isinstance(errors, pl.DataFrame)


def test_dataframe_input_row_count():
    n = 3
    results, errors = ignition.run(_make_df(n))
    # n sequences × 447 positions each
    assert results.shape[0] == n * 447
    assert errors.shape[0] == 0


def test_dataframe_sequence_ids():
    n = 4
    results, errors = ignition.run(_make_df(n))
    ids = sorted(results["sequence_id"].unique().to_list())
    assert ids == list(range(n))


def test_dataframe_custom_nt_col():
    df = pl.DataFrame({
        "seq_nt": [HEAVY_NT],
        "seq_aa": [HEAVY_AA],
    })
    results, errors = ignition.run(df, nt_col="seq_nt", aa_col="seq_aa")
    assert results.shape[0] == 447
    assert errors.shape[0] == 0


def test_dataframe_locus_chain_detection():
    results, errors = ignition.run(_make_df())
    assert set(results["chain"].unique().to_list()) == {"H"}


def test_dataframe_error_collection():
    df = pl.DataFrame({
        "sequence": [HEAVY_NT, ""],          # second seq is empty → error
        "sequence_aa": [HEAVY_AA, HEAVY_AA],
        "locus": ["IGH", "IGH"],
    })
    results, errors = ignition.run(df)
    assert results.shape[0] == 447     # only first sequence
    assert errors.shape[0] == 1


def test_dataframe_errors_schema():
    df = pl.DataFrame({
        "sequence": [""],
        "sequence_aa": [HEAVY_AA],
        "locus": ["IGH"],
    })
    _, errors = ignition.run(df)
    assert errors.columns == ["sequence_id", "chain", "error"]


# ─── Paired mode tests ─────────────────────────────────────────────────────────


def test_paired_mode_auto_detection():
    df = pl.DataFrame({
        "sequence:0": [HEAVY_NT],
        "sequence_aa:0": [HEAVY_AA],
        "sequence:1": [HEAVY_NT],   # use heavy seq as a proxy for light
        "sequence_aa:1": [HEAVY_AA],
    })
    results, errors = ignition.run(df)
    # Should process both heavy and the light (which will pick best chain)
    assert results.shape[0] > 0
