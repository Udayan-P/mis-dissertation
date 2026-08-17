"""Regression tests for scripts/make_results_tables.py's table-building logic.

This is the file the report's numbers get quoted from, so a formatting or
aggregation bug here is a silent wrong-number-in-the-report risk, not just a
cosmetic one (md_table's int-vs-float handling already had one such bug,
caught by inspection before this test existed).
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from make_results_tables import coverage_section, md_table, ratio_by_n  # noqa: E402


def test_md_table_formats_n_and_count_as_int_not_float():
    df = pd.DataFrame({"n": [24.0, 26.0], "ratio": [0.886, 1.209], "n_instances": [15.0, 15.0]})
    out = md_table(df, ["n", "ratio", "n_instances"])
    assert "| 24 | 0.886 | 15 |" in out
    assert "24.000" not in out
    assert "15.000" not in out


def test_ratio_by_n_median_and_count():
    df = pd.DataFrame({
        "instance": ["i1", "i1", "i2", "i2"],
        "n": [24, 24, 24, 24],
        "algorithm": ["bk_pivot", "bk_tomita", "bk_pivot", "bk_tomita"],
        "median_seconds": [1.0, 2.0, 3.0, 2.0],
    })
    g = ratio_by_n(df, "bk_pivot", "bk_tomita", "median_seconds")
    # i1: 1.0/2.0 = 0.5 ; i2: 3.0/2.0 = 1.5 ; median of [0.5, 1.5] = 1.0
    assert g.loc[24, "median"] == 1.0
    assert g.loc[24, "n_instances"] == 2


def test_coverage_section_flags_short_arm_and_names_missing_instances():
    df = pd.DataFrame({
        "instance": ["a", "b", "a", "b"],
        "n": [40, 40, 40, 40],
        "algorithm": ["bk_pivot", "bk_pivot", "bk_tomita", "bk_tomita"],
        "mis_count": [5, 5, 5, 5],
    })
    # bk_tomita is missing instance "a" entirely at n=40
    df = df[~((df["algorithm"] == "bk_tomita") & (df["instance"] == "a"))]
    out = coverage_section(df)
    assert "bk_tomita" in out
    assert "1/2" in out
    assert "missing at n=40" in out
    assert "a" in out.split("missing at n=40:")[1]


def test_coverage_section_reports_all_agree_when_counts_match():
    df = pd.DataFrame({
        "instance": ["a", "a"],
        "n": [24, 24],
        "algorithm": ["bk_pivot", "bk_tomita"],
        "mis_count": [5, 5],
    })
    assert "ALL AGREE" in coverage_section(df)


def test_coverage_section_flags_mis_count_disagreement():
    df = pd.DataFrame({
        "instance": ["a", "a"],
        "n": [24, 24],
        "algorithm": ["bk_pivot", "bk_tomita"],
        "mis_count": [5, 6],
    })
    assert "DISAGREEMENT" in coverage_section(df)
