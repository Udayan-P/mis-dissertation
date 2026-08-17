"""Answer the two crossover questions from a sweep CSV.

  1. pivot vs tomita: ratio of median times against n. Above 1 means Tomita
     is winning. Where does it cross?
  2. set vs bitset: same ratio per variant against n.

Also reports time per recursion node, which separates "searches less" from
"is faster per unit of search".

Usage: python scripts/crossover_analysis.py results/crossover_xxx.csv
       (accepts several CSVs, they get concatenated)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = Path(__file__).resolve().parents[1] / "figures"


def load(paths):
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    return df.drop_duplicates(subset=["instance", "algorithm"])


def ratio_table(df, a, b, label):
    """median(time_a / time_b) per n, split by labelling when more than one is present.

    Pooling labellings together in one median per n hides the effect: a cheap
    pivot rule under degeneracy labelling behaves like an ordering heuristic and
    wins almost everywhere, while under random labelling it doesn't. Averaging
    the two into one line makes both look like noise.
    """
    has_lab = "labelling" in df.columns and df["labelling"].nunique() > 1
    idx = ["instance", "n"] + (["labelling"] if has_lab else [])
    w = df[df["algorithm"].isin([a, b])].pivot_table(
        index=idx, columns="algorithm",
        values="median_seconds").dropna().reset_index()
    if a not in w.columns or b not in w.columns:
        return None
    w["ratio"] = w[a] / w[b]
    group_cols = ["labelling", "n"] if has_lab else ["n"]
    g = w.groupby(group_cols)["ratio"].agg(["median", "count"])
    print(f"\n{label}  (>1 means {b} is faster)")
    if has_lab:
        print(f"{'labelling':>12}{'n':>5}{'ratio':>9}{'instances':>11}")
        for (lab, n), row in g.iterrows():
            flag = "  <-- crossover" if row["median"] > 1 else ""
            print(f"{lab:>12}{n:>5}{row['median']:>9.2f}{int(row['count']):>11}{flag}")
    else:
        print(f"{'n':>5}{'ratio':>9}{'instances':>11}")
        for n, row in g.iterrows():
            flag = "  <-- crossover" if row["median"] > 1 else ""
            print(f"{n:>5}{row['median']:>9.2f}{int(row['count']):>11}{flag}")
    return g


def per_node_cost(df):
    """Microseconds of wall clock per recursion node, split by labelling."""
    d = df[df["recursion_nodes"].notna() & (df["recursion_nodes"] > 0)].copy()
    if d.empty:
        return
    d["us_per_node"] = d["median_seconds"] * 1e6 / d["recursion_nodes"]
    has_lab = "labelling" in d.columns and d["labelling"].nunique() > 1
    print("\ntime per recursion node (microseconds, median over instances)")
    cols = [c for c in ["bk_basic", "bk_pivot", "bk_tomita"] if c in d["algorithm"].unique()]
    if has_lab:
        for lab, g in d.groupby("labelling"):
            print(f"\n  labelling = {lab}")
            piv = g.pivot_table(index="n", columns="algorithm", values="us_per_node")
            print(piv[[c for c in cols if c in piv.columns]].round(2).to_string())
    else:
        piv = d.pivot_table(index="n", columns="algorithm", values="us_per_node")
        print(piv[[c for c in cols if c in piv.columns]].round(2).to_string())
    print("\nIf Tomita's per-node cost stays flat while its node count keeps "
          "falling relative to the others, it wins eventually; if per-node "
          "cost grows with n, it may not.")


def node_ratios(df):
    has_lab = "labelling" in df.columns and df["labelling"].nunique() > 1
    idx = ["instance", "n"] + (["labelling"] if has_lab else [])
    w = df.pivot_table(index=idx, columns="algorithm",
                       values="recursion_nodes").dropna().reset_index()
    if not {"bk_basic", "bk_pivot", "bk_tomita"} <= set(w.columns):
        return
    w["basic_over_tomita"] = w["bk_basic"] / w["bk_tomita"]
    w["pivot_over_tomita"] = w["bk_pivot"] / w["bk_tomita"]
    group_cols = ["labelling", "n"] if has_lab else ["n"]
    print("\nsearch tree size ratios (>1 means tomita searches less)")
    print(w.groupby(group_cols)[["basic_over_tomita", "pivot_over_tomita"]]
          .median().round(2).to_string())


def plots(df):
    FIG.mkdir(exist_ok=True)

    # pivot vs tomita ratio against n, one line per family
    has_lab = "labelling" in df.columns and df["labelling"].nunique() > 1
    w = df[df["algorithm"].isin(["bk_pivot", "bk_tomita"])].pivot_table(
        index=["instance", "n", "family"]
              + (["labelling"] if has_lab else []),
        columns="algorithm", values="median_seconds").dropna().reset_index()
    if {"bk_pivot", "bk_tomita"} <= set(w.columns):
        w["ratio"] = w["bk_pivot"] / w["bk_tomita"]
        plt.figure(figsize=(7, 4.5))
        for fam, d in w.groupby("family"):
            g = d.groupby("n")["ratio"].median()
            plt.plot(g.index, g.values, marker="o", ms=4, label=fam)
        plt.axhline(1.0, color="k", ls="--", lw=1, label="equal")
        plt.xlabel("n")
        plt.ylabel("time(pivot) / time(tomita)")
        plt.title("Does Tomita's pivot rule overtake simple pivoting?")
        plt.legend(fontsize=8)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIG / "pivot_vs_tomita_crossover.png", dpi=150)
        plt.close()

    # pivot vs tomita ratio against n, one line per labelling
    if has_lab:
        plt.figure(figsize=(7, 4.5))
        for lab, d in w.groupby("labelling"):
            g = d.groupby("n")["ratio"].median()
            plt.plot(g.index, g.values, marker="o", ms=4, label=lab)
        plt.axhline(1.0, color="k", ls="--", lw=1, label="equal")
        plt.xlabel("n")
        plt.ylabel("time(pivot) / time(tomita)")
        plt.title("Does labelling change when Tomita's pivot rule pays off?")
        plt.legend(fontsize=8)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIG / "pivot_vs_tomita_by_labelling.png", dpi=150)
        plt.close()

    # bitset speedup against n, with the CPython 30-bit digit boundary marked
    plt.figure(figsize=(7, 4.5))
    any_line = False
    for base, bits in [("bk_basic", "bk_basic_bit"), ("bk_pivot", "bk_pivot_bit"),
                       ("bk_tomita", "bk_tomita_bit")]:
        w = df[df["algorithm"].isin([base, bits])].pivot_table(
            index=["instance", "n"], columns="algorithm",
            values="median_seconds").dropna().reset_index()
        if {base, bits} <= set(w.columns) and len(w):
            w["speedup"] = w[base] / w[bits]
            g = w.groupby("n")["speedup"].median()
            plt.plot(g.index, g.values, marker="o", ms=4, label=base)
            any_line = True
    if any_line:
        plt.axhline(1.0, color="k", ls="--", lw=1)
        # a vertex mask needs a second 30-bit digit from n = 31 (sys.int_info)
        plt.axvline(30.5, color="r", ls=":", lw=1.2,
                    label="30-bit digit boundary")
        plt.xlabel("n")
        plt.ylabel("speedup from bitsets")
        plt.title("Bitset speedup vs n")
        plt.legend(fontsize=8)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIG / "bitset_speedup_vs_n.png", dpi=150)
    plt.close()

    # node-count ratio and per-node cost ratio vs n, on one figure: the
    # interaction between the two is the decomposition that explains the
    # headline pivot/tomita crossover.
    w = df[df["algorithm"].isin(["bk_pivot", "bk_tomita"])].pivot_table(
        index=["instance", "n"], columns="algorithm",
        values=["median_seconds", "recursion_nodes"]).dropna().reset_index()
    if {"median_seconds", "recursion_nodes"} <= set(w.columns.get_level_values(0)):
        w.columns = ["_".join(c).strip("_") for c in w.columns]
        w["node_ratio"] = w["recursion_nodes_bk_pivot"] / w["recursion_nodes_bk_tomita"]
        w["us_pivot"] = (w["median_seconds_bk_pivot"] * 1e6
                          / w["recursion_nodes_bk_pivot"])
        w["us_tomita"] = (w["median_seconds_bk_tomita"] * 1e6
                           / w["recursion_nodes_bk_tomita"])
        w["cost_ratio"] = w["us_tomita"] / w["us_pivot"]
        g = w.groupby("n")[["node_ratio", "cost_ratio"]].median()

        fig, ax1 = plt.subplots(figsize=(7, 4.5))
        ax1.plot(g.index, g["node_ratio"], marker="o", ms=4, color="tab:blue",
                 label="tree-size ratio (nodes: pivot/tomita)")
        ax1.set_xlabel("n")
        ax1.set_ylabel("tree-size ratio", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax2 = ax1.twinx()
        ax2.plot(g.index, g["cost_ratio"], marker="s", ms=4, color="tab:orange",
                 label="per-node cost ratio (tomita/pivot)")
        ax2.set_ylabel("per-node cost ratio", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left")
        plt.title("Tomita wins once tree-size ratio exceeds per-node cost ratio")
        ax1.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG / "node_ratio_and_cost_vs_n.png", dpi=150)
        plt.close(fig)


def main(paths):
    df = load(paths)
    print(f"{len(df)} rows, n from {df['n'].min()} to {df['n'].max()}")
    bad = df.groupby("instance")["mis_count"].nunique()
    print("cross-validation:",
          "ALL AGREE" if (bad <= 1).all() else f"DISAGREEMENT: {bad[bad>1].index.tolist()}")

    ratio_table(df, "bk_pivot", "bk_tomita", "Q1  pivot vs tomita")
    for base, bits in [("bk_basic", "bk_basic_bit"), ("bk_pivot", "bk_pivot_bit"),
                       ("bk_tomita", "bk_tomita_bit")]:
        ratio_table(df, base, bits, f"Q2  {base} vs bitset")
    node_ratios(df)
    per_node_cost(df)
    plots(df)
    print(f"\nfigures -> {FIG}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        args = sorted((Path(__file__).resolve().parents[1] / "results")
                      .glob("crossover_*.csv"))
        if not args:
            sys.exit("no crossover CSV found; run scripts/crossover_sweep.py")
    main([Path(a) for a in args])
