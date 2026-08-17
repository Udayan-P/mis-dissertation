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

import random
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


def clique_overlap(G, sample=200, seed=0):
    """Mean pairwise Jaccard between maximal independent sets.

    Abu-Khzam et al. (2005) predict that pivoting pays off on graphs with
    "a large number of highly overlapping cliques" and that un-pivoted BK
    should win "when there is little overlap among cliques". They build a
    clique intersection graph to look at this and then use it for biology
    rather than to test the prediction, so as far as I can tell the
    hypothesis has never been evaluated.

    Their CIG needs an arbitrary intersection threshold; mean pairwise
    Jaccard is the same idea as a single threshold-free scalar. Quadratic
    in the number of sets, so sample when there are many.
    """
    sets = [frozenset(s) for s in nx.find_cliques(nx.complement(G))]
    if len(sets) < 2:
        return 0.0
    if len(sets) > sample:
        sets = random.Random(seed).sample(sets, sample)
    tot = pairs = 0.0
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            u = len(sets[i] | sets[j])
            if u:
                tot += len(sets[i] & sets[j]) / u
            pairs += 1
    return tot / pairs if pairs else 0.0


def ck_ratio(G):
    """Cazals & Karande's mechanism as a per-instance number.

    RR-5615 / TCS 2008: pivoting from P alone degrades "whenever any node
    in P has a number of neighbors larger than the size of a clique to be
    discovered" - their K_n u K_{1,n} worst case. The algorithms work on
    the complement, so the quantity is the complement's maximum degree
    against the size of the largest maximal independent set. A ratio above
    1 means the configuration that breaks P-only pivoting is available;
    the larger it is, the more room there is for it to occur.
    """
    C = nx.complement(G)
    if C.number_of_edges() == 0:
        return 0.0
    omega = max((len(c) for c in nx.find_cliques(C)), default=1)
    return max(dict(C.degree()).values()) / max(omega, 1)


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
            # the two predictors the literature names and never tests
            "clique_overlap": clique_overlap(G),
            "ck_ratio": ck_ratio(G),
        })
    return pd.DataFrame(rows)


CORR_COLS = ["degeneracy", "comp_degeneracy", "deg_ratio", "clustering",
             "density", "clique_overlap", "ck_ratio", "n", "node_gain"]


def correlation_block(m, label_prefix=""):
    print(f"\n{label_prefix}correlation with tomita_gain (time ratio pivot/tomita)")
    for c in CORR_COLS:
        r = spearman(m[c], m["tomita_gain"])
        print(f"  {c:<18} spearman r = {r:+.3f}")

    print(f"\n{label_prefix}correlation with node_gain (tree size ratio pivot/tomita)")
    for c in CORR_COLS[:-1]:
        r = spearman(m[c], m["node_gain"])
        print(f"  {c:<18} spearman r = {r:+.3f}")

    print(f"\n{label_prefix}tomita_gain by complement-degeneracy quartile")
    q = pd.qcut(m["comp_degeneracy"], 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"],
                duplicates="drop")
    print(m.groupby(q, observed=True)[["comp_degeneracy", "tomita_gain",
                                        "node_gain"]].median().round(3).to_string())


def main(path):
    df = pd.read_csv(path)
    FIG.mkdir(exist_ok=True)

    has_lab = "labelling" in df.columns and df["labelling"].nunique() > 1
    idx = ["instance", "family", "n"] + (["labelling"] if has_lab else [])

    ratios = df[df["algorithm"].isin(["bk_pivot", "bk_tomita"])].pivot_table(
        index=idx, columns="algorithm",
        values="median_seconds").dropna().reset_index()
    ratios["tomita_gain"] = ratios["bk_pivot"] / ratios["bk_tomita"]

    nodes = df[df["algorithm"].isin(["bk_pivot", "bk_tomita"])].pivot_table(
        index="instance", columns="algorithm",
        values="recursion_nodes").dropna().reset_index()
    nodes["node_gain"] = nodes["bk_pivot"] / nodes["bk_tomita"]

    # structural features (degeneracy, clustering, density, clique overlap, ...)
    # are invariant to vertex labelling -- relabelling only permutes vertex
    # numbers, it does not change the graph. They get recomputed once per
    # instance name (which already encodes the labelling), so this is
    # redundant but not wrong.
    feats = structural_features(df)
    m = ratios.merge(feats, on="instance").merge(
        nodes[["instance", "node_gain"]], on="instance")

    print(f"{len(m)} instances with structure computed\n")

    print("median structural properties by family" + (" and labelling" if has_lab else ""))
    cols = ["degeneracy", "comp_degeneracy", "deg_ratio", "clustering",
            "density", "clique_overlap", "ck_ratio", "node_gain",
            "tomita_gain"]
    group_cols = ["family", "labelling"] if has_lab else "family"
    print(m.groupby(group_cols)[cols].median().round(3).to_string())

    if has_lab:
        # tomita_gain and node_gain are NOT labelling-invariant: the cheap
        # pivot rule takes the lowest-numbered vertex of its source, so
        # relabelling changes which vertex that is on every call. Pooling
        # labellings together correlates one fixed structural value against
        # three different tomita_gain regimes and the signal collapses.
        # Report the structural claim per labelling; the headline claim in
        # the report uses final_v3.csv, which is random-labelling only.
        for lab, g in m.groupby("labelling"):
            print(f"\n{'=' * 60}\nlabelling = {lab}  ({len(g)} instances)\n{'=' * 60}")
            correlation_block(g)
        print(f"\n{'=' * 60}\nall labellings pooled (misleading -- shown only for contrast)"
              f"\n{'=' * 60}")
        correlation_block(m)
    else:
        correlation_block(m)

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

    if has_lab:
        plt.figure(figsize=(7, 4.5))
        for lab, d in m.groupby("labelling"):
            plt.scatter(d["deg_ratio"], d["tomita_gain"], s=18, alpha=0.7, label=lab)
        plt.axhline(1.0, color="k", ls="--", lw=1)
        plt.xlabel("complement degeneracy / n")
        plt.ylabel("time(pivot) / time(tomita)")
        plt.title("Complement degeneracy vs Tomita gain, by labelling")
        plt.legend(fontsize=8)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIG / "degeneracy_vs_tomita_gain_by_labelling.png", dpi=150)
        plt.close()
        print(f"figure -> {FIG / 'degeneracy_vs_tomita_gain_by_labelling.png'}")


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
