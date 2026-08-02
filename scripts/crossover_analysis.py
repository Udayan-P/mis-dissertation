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
    """median(time_a / time_b) per n."""
    w = df[df["algorithm"].isin([a, b])].pivot_table(
        index=["instance", "n"], columns="algorithm",
        values="median_seconds").dropna().reset_index()
    if a not in w.columns or b not in w.columns:
        return None
    w["ratio"] = w[a] / w[b]
    g = w.groupby("n")["ratio"].agg(["median", "count"])
    print(f"\n{label}  (>1 means {b} is faster)")
    print(f"{'n':>5}{'ratio':>9}{'instances':>11}")
    for n, row in g.iterrows():
        flag = "  <-- crossover" if row["median"] > 1 else ""
        print(f"{n:>5}{row['median']:>9.2f}{int(row['count']):>11}{flag}")
    return g


def per_node_cost(df):
    """Microseconds of wall clock per recursion node."""
    d = df[df["recursion_nodes"].notna() & (df["recursion_nodes"] > 0)].copy()
    if d.empty:
        return
    d["us_per_node"] = d["median_seconds"] * 1e6 / d["recursion_nodes"]
    print("\ntime per recursion node (microseconds, median over instances)")
    piv = d.pivot_table(index="n", columns="algorithm", values="us_per_node")
    cols = [c for c in ["bk_basic", "bk_pivot", "bk_tomita"] if c in piv]
    print(piv[cols].round(2).to_string())
    print("\nIf Tomita's per-node cost stays flat while its node count keeps "
          "falling relative to the others, it wins eventually; if per-node "
          "cost grows with n, it may not.")


def node_ratios(df):
    w = df.pivot_table(index=["instance", "n"], columns="algorithm",
                       values="recursion_nodes").dropna().reset_index()
    if not {"bk_basic", "bk_pivot", "bk_tomita"} <= set(w.columns):
        return
    w["basic_over_tomita"] = w["bk_basic"] / w["bk_tomita"]
    w["pivot_over_tomita"] = w["bk_pivot"] / w["bk_tomita"]
    print("\nsearch tree size ratios (>1 means tomita searches less)")
    print(w.groupby("n")[["basic_over_tomita", "pivot_over_tomita"]]
          .median().round(2).to_string())


def plots(df):
    FIG.mkdir(exist_ok=True)

    # pivot vs tomita ratio against n
    w = df[df["algorithm"].isin(["bk_pivot", "bk_tomita"])].pivot_table(
        index=["instance", "n", "family"], columns="algorithm",
        values="median_seconds").dropna().reset_index()
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

    # bitset speedup against n
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
        plt.xlabel("n")
        plt.ylabel("speedup from bitsets")
        plt.title("Bitset speedup vs n")
        plt.legend(fontsize=8)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIG / "bitset_speedup_vs_n.png", dpi=150)
    plt.close()


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
