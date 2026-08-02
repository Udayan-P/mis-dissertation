"""Does graph structure predict when Tomita's pivot rule pays off?

The family split (Tomita wins on BA and sparse ER, loses on WS and dense ER)
needs an explanation better than "family". Candidates, computed per instance:

  degeneracy of G          - the parameter Eppstein-Loffler-Strash bound on
  degeneracy of complement - the algorithms actually run on the complement
  mean clustering          - what makes WS different from ER at equal density
  density                  - the obvious baseline predictor
  output size              - number of maximal independent sets

Every instance is rebuilt from its recorded family/n/p/seed, so this joins
structure onto the existing results without re-running any benchmarks.

Usage: python scripts/structure_analysis.py results/crossover_v2.csv
"""

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mis import graphs

FIG = Path(__file__).resolve().parents[1] / "figures"


def spearman(a, b):
    """Rank correlation without pulling in scipy: Pearson on the ranks."""
    return a.rank().corr(b.rank())


def rebuild(row):
    fam, n, seed = row["family"], int(row["n"]), int(row["seed"])
    if fam == "er":
        return graphs.erdos_renyi(n, float(row["p"]), seed)
    if fam == "ba":
        return graphs.barabasi_albert(n, 3, seed)
    if fam == "ws":
        return graphs.watts_strogatz(n, min(6, n - 1), 0.1, seed)
    return None


def degeneracy(G):
    """Max core number = degeneracy."""
    if G.number_of_edges() == 0:
        return 0
    return max(nx.core_number(G).values())


def structural_features(df):
    rows = []
    for (inst, fam, n, p, seed), _ in df.groupby(
            ["instance", "family", "n", "p", "seed"], dropna=False):
        G = rebuild({"family": fam, "n": n, "p": p, "seed": seed})
        if G is None:
            continue
        C = nx.complement(G)
        rows.append({
            "instance": inst,
            "degeneracy": degeneracy(G),
            "comp_degeneracy": degeneracy(C),
            "clustering": nx.average_clustering(G),
            "comp_clustering": nx.average_clustering(C),
            "density": nx.density(G),
            "deg_ratio": degeneracy(C) / n,
        })
    return pd.DataFrame(rows)


def main(path):
    df = pd.read_csv(path)
    FIG.mkdir(exist_ok=True)

    ratios = df[df["algorithm"].isin(["bk_pivot", "bk_tomita"])].pivot_table(
        index=["instance", "family", "n"], columns="algorithm",
        values="median_seconds").dropna().reset_index()
    ratios["tomita_gain"] = ratios["bk_pivot"] / ratios["bk_tomita"]

    nodes = df[df["algorithm"].isin(["bk_pivot", "bk_tomita"])].pivot_table(
        index="instance", columns="algorithm",
        values="recursion_nodes").dropna().reset_index()
    nodes["node_gain"] = nodes["bk_pivot"] / nodes["bk_tomita"]

    feats = structural_features(df)
    m = ratios.merge(feats, on="instance").merge(
        nodes[["instance", "node_gain"]], on="instance")

    print(f"{len(m)} instances with structure computed\n")

    print("median structural properties by family")
    cols = ["degeneracy", "comp_degeneracy", "deg_ratio", "clustering",
            "density", "node_gain", "tomita_gain"]
    print(m.groupby("family")[cols].median().round(3).to_string())

    print("\ncorrelation with tomita_gain (time ratio pivot/tomita)")
    for c in ["degeneracy", "comp_degeneracy", "deg_ratio", "clustering",
              "density", "n", "node_gain"]:
        r = spearman(m[c], m["tomita_gain"])
        print(f"  {c:<18} spearman r = {r:+.3f}")

    print("\ncorrelation with node_gain (tree size ratio pivot/tomita)")
    for c in ["degeneracy", "comp_degeneracy", "deg_ratio", "clustering",
              "density"]:
        r = spearman(m[c], m["node_gain"])
        print(f"  {c:<18} spearman r = {r:+.3f}")

    # the key question: does structure explain the split better than family?
    print("\ntomita_gain by complement-degeneracy quartile")
    m["q"] = pd.qcut(m["comp_degeneracy"], 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"],
                     duplicates="drop")
    print(m.groupby("q", observed=True)[["comp_degeneracy", "tomita_gain",
                                         "node_gain"]].median().round(3).to_string())

    plt.figure(figsize=(7, 4.5))
    for fam, d in m.groupby("family"):
        plt.scatter(d["deg_ratio"], d["tomita_gain"], s=18, alpha=0.7, label=fam)
    plt.axhline(1.0, color="k", ls="--", lw=1)
    plt.xlabel("complement degeneracy / n")
    plt.ylabel("time(pivot) / time(tomita)")
    plt.title("Does complement degeneracy predict when Tomita wins?")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "degeneracy_vs_tomita_gain.png", dpi=150)
    plt.close()
    print(f"\nfigure -> {FIG / 'degeneracy_vs_tomita_gain.png'}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        c = sorted((Path(__file__).resolve().parents[1] / "results")
                   .glob("crossover*.csv"))
        if not c:
            sys.exit("no crossover CSV found")
        target = c[-1]
    else:
        target = Path(sys.argv[1])
    main(target)
