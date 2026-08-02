"""Analysis of a campaign CSV: growth rates, pivoting speedups, plots.

Main number is the empirical growth base c, fitting log(time) = a + n*log(c)
per family/density, compared against Moon-Moser 3^(1/3) = 1.4422.

Usage: python scripts/analyse.py results/campaign_xxx.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MOON_MOSER = 3 ** (1 / 3)
FIG = Path(__file__).resolve().parents[1] / "figures"


def growth_base(df, min_points=4):
    """Fit log(time) vs n and return the base c, plus the fit quality."""
    d = df[(df["median_seconds"] > 0) & (df["n"] > 0)]
    if len(d) < min_points or d["n"].nunique() < min_points:
        return None
    # average repeats at the same n first, so every n weighs equally
    g = d.groupby("n")["median_seconds"].median()
    slope, intercept = np.polyfit(g.index.values, np.log(g.values), 1)
    pred = intercept + slope * g.index.values
    resid = np.log(g.values) - pred
    ss_tot = ((np.log(g.values) - np.log(g.values).mean()) ** 2).sum()
    r2 = 1 - (resid ** 2).sum() / ss_tot if ss_tot > 0 else float("nan")
    return {"c": float(np.exp(slope)), "r2": float(r2), "points": int(len(g))}


def report(path):
    df = pd.read_csv(path)
    FIG.mkdir(exist_ok=True)
    print(f"{len(df)} rows, {df['instance'].nunique()} instances, "
          f"n from {df['n'].min()} to {df['n'].max()}\n")

    # correctness first: every algorithm must agree per instance
    disagree = (df.groupby("instance")["mis_count"].nunique() > 1)
    bad = disagree[disagree].index.tolist()
    print("cross-validation:", "ALL AGREE" if not bad
          else f"DISAGREEMENT on {bad}")

    print("\nempirical growth base c (time ~ c^n), by family/density/algorithm")
    print(f"{'family':<8}{'p':>6}{'algorithm':>16}{'c':>9}{'r2':>7}{'pts':>5}")
    rows = []
    for (fam, p, alg), sub in df.groupby(["family", "p", "algorithm"],
                                         dropna=False):
        fit = growth_base(sub)
        if fit:
            rows.append((fam, p, alg, fit))
            print(f"{fam:<8}{p:>6}{alg:>16}{fit['c']:>9.3f}"
                  f"{fit['r2']:>7.3f}{fit['points']:>5}")
    print(f"\nMoon-Moser worst case 3^(1/3) = {MOON_MOSER:.4f}")

    # how much does pivoting save, in tree size and in time?
    print("\npivoting effect (median over instances where all three ran)")
    piv = df[df["algorithm"].isin(["bk_basic", "bk_pivot", "bk_tomita"])]
    wide_n = piv.pivot_table(index="instance", columns="algorithm",
                             values="recursion_nodes")
    wide_t = piv.pivot_table(index="instance", columns="algorithm",
                             values="median_seconds")
    if {"bk_basic", "bk_pivot", "bk_tomita"} <= set(wide_n.columns):
        wide_n = wide_n.dropna()
        wide_t = wide_t.dropna()
        print(f"  nodes  basic/pivot  = {(wide_n['bk_basic']/wide_n['bk_pivot']).median():.2f}x")
        print(f"  nodes  basic/tomita = {(wide_n['bk_basic']/wide_n['bk_tomita']).median():.2f}x")
        print(f"  nodes  pivot/tomita = {(wide_n['bk_pivot']/wide_n['bk_tomita']).median():.2f}x")
        print(f"  time   basic/pivot  = {(wide_t['bk_basic']/wide_t['bk_pivot']).median():.2f}x")
        print(f"  time   basic/tomita = {(wide_t['bk_basic']/wide_t['bk_tomita']).median():.2f}x")
        print(f"  time   pivot/tomita = {(wide_t['bk_pivot']/wide_t['bk_tomita']).median():.2f}x")

    # what did the bitset representation buy?
    print("\nbitset speedup (same algorithm, per instance)")
    for base, bits in [("bk_basic", "bk_basic_bit"), ("bk_pivot", "bk_pivot_bit"),
                       ("bk_tomita", "bk_tomita_bit")]:
        w = df[df["algorithm"].isin([base, bits])].pivot_table(
            index="instance", columns="algorithm", values="median_seconds").dropna()
        if {base, bits} <= set(w.columns) and len(w):
            print(f"  {base:<12} -> {bits:<16} {(w[base]/w[bits]).median():.2f}x")

    make_plots(df)
    print(f"\nfigures written to {FIG}")


def make_plots(df):
    # runtime vs n, log scale, one line per algorithm, ER p=0.5
    sub = df[(df["family"] == "er") & (df["p"] == 0.5)]
    if len(sub):
        plt.figure(figsize=(7, 4.5))
        for alg, d in sub.groupby("algorithm"):
            g = d.groupby("n")["median_seconds"].median()
            plt.semilogy(g.index, g.values, marker="o", ms=3, label=alg)
        plt.xlabel("n")
        plt.ylabel("median runtime (s)")
        plt.title("Runtime vs n, G(n, 0.5)")
        plt.legend(fontsize=7)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIG / "runtime_vs_n_er05.png", dpi=150)
        plt.close()

    # effect of density on the pivoted algorithms
    sub = df[(df["family"] == "er") & df["algorithm"].isin(
        ["bk_basic", "bk_pivot", "bk_tomita"])]
    if len(sub) and sub["p"].nunique() > 1:
        plt.figure(figsize=(7, 4.5))
        for (alg, p), d in sub.groupby(["algorithm", "p"]):
            g = d.groupby("n")["median_seconds"].median()
            plt.semilogy(g.index, g.values, marker="o", ms=3,
                         label=f"{alg} p={p}")
        plt.xlabel("n")
        plt.ylabel("median runtime (s)")
        plt.title("Density effect, Erdos-Renyi")
        plt.legend(fontsize=6, ncol=2)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIG / "density_effect.png", dpi=150)
        plt.close()

    # number of maximal independent sets vs n against the Moon-Moser bound
    sub = df[df["family"] == "er"].drop_duplicates("instance")
    if len(sub):
        plt.figure(figsize=(7, 4.5))
        for p, d in sub.groupby("p"):
            g = d.groupby("n")["mis_count"].median()
            plt.semilogy(g.index, g.values, marker="o", ms=3, label=f"p={p}")
        ns = np.arange(sub["n"].min(), sub["n"].max() + 1)
        plt.semilogy(ns, 3 ** (ns / 3), "k--", lw=1, label="$3^{n/3}$ bound")
        plt.xlabel("n")
        plt.ylabel("number of maximal independent sets")
        plt.title("Output size vs the Moon-Moser bound")
        plt.legend(fontsize=7)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIG / "output_size_vs_bound.png", dpi=150)
        plt.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        results = sorted((Path(__file__).resolve().parents[1] / "results")
                         .glob("*.csv"))
        if not results:
            sys.exit("no results CSV found; run scripts/run_campaign.py first")
        target = results[-1]
    else:
        target = Path(sys.argv[1])
    report(target)
