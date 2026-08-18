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
from make_results_tables import (  # noqa: E402
    coverage_section, md_table, node_and_cost_table, ratio_by_n)


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


def test_node_and_cost_table_unaffected_by_missing_third_algorithm():
    """A missing bk_basic row must not drop that instance from
    pivot_over_tomita_nodes or cost_ratio: those need only bk_pivot and
    bk_tomita. A single three-way dropna() did exactly that (final_v3.csv
    n=40: 15 instances have bk_pivot/bk_tomita, only 12 also have bk_basic),
    silently computing S5.3's headline node_ratio and cost_ratio off 12
    instances instead of 15.
    """
    # pivot/tomita node ratio differs per instance (3.0, 1.0, 2.0) so a
    # median over the wrong subset gives a different, wrong answer.
    pivot_nodes = {"a": 30, "b": 10, "c": 20}
    rows = []
    for inst, nodes in pivot_nodes.items():
        rows.append({"instance": inst, "n": 40, "algorithm": "bk_pivot",
                      "median_seconds": 2.0, "recursion_nodes": nodes})
        rows.append({"instance": inst, "n": 40, "algorithm": "bk_tomita",
                      "median_seconds": 1.0, "recursion_nodes": 10})
    # bk_basic present for only instance "a", whose pivot/tomita ratio (3.0)
    # is the outlier -- if the bug is present, node_ratio collapses to just
    # that instance instead of the median over all three (2.0).
    rows.append({"instance": "a", "n": 40, "algorithm": "bk_basic",
                  "median_seconds": 9.0, "recursion_nodes": 999})
    df = pd.DataFrame(rows)

    out = node_and_cost_table(df)
    lines = [l for l in out.splitlines() if l.startswith("| 40")]
    assert len(lines) == 1
    cells = [c.strip() for c in lines[0].strip("|").split("|")]
    assert cells[2] == "2.000"   # pivot_over_tomita_nodes: median over a, b, c
    assert cells[2] != "3.000"   # the buggy value: instance "a" only
