"""Emit every numeric table needed to write S5 into one reference file.

Numbers only, no prose: the source CSV is named per table so a claim in the
report can be traced back to the data it came from. The structural
correlation table is deliberately built from final_v3.csv only; see CLAUDE.md
open item 1 on why pooled labellings collapse the correlation.

Usage: python scripts/make_results_tables.py
Writes results/RESULTS_TABLES.md.
"""

from pathlib import Path

import networkx as nx
import pandas as pd

from mis import graphs

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results" / "final_v3.csv"
LABELLING = ROOT / "results" / "labelling_v3.csv"
OUT = ROOT / "results" / "RESULTS_TABLES.md"


def spearman(a, b):
    return a.rank().corr(b.rank())


def ratio_by_n(df, a, b, value_col):
    w = df[df["algorithm"].isin([a, b])].pivot_table(
        index=["instance", "n"], columns="algorithm",
        values=value_col).dropna().reset_index()
    if a not in w.columns or b not in w.columns:
        return None
    w["ratio"] = w[a] / w[b]
    return w.groupby("n")["ratio"].agg(median="median", n_instances="count")


INT_COLS = {"n", "n_instances"}


def md_table(df, cols=None, floatfmt="{:.3f}"):
    if cols is None:
        cols = df.columns.tolist()
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    lines = [header, sep]
    for _, row in df[cols].iterrows():
        cells = []
        for name, v in zip(cols, row):
            if name in INT_COLS and isinstance(v, float):
                cells.append(str(int(v)))
            elif isinstance(v, float):
                cells.append(floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def coverage_section(df):
    algos = sorted(df["algorithm"].unique())
    counts = df.groupby("algorithm").size()
    expected = counts.max()
    short = counts[counts < expected]
    lines = ["## Data coverage", "", f"Source: `{FINAL.name}`", "",
              f"{len(df)} rows total, {df['instance'].nunique()} instances, "
              f"n = {df['n'].min()}..{df['n'].max()}, {len(algos)} arms "
              f"(expected {expected} rows/arm).", ""]
    if len(short):
        lines.append("Arms short of full coverage:")
        for alg, c in short.items():
            n40 = df[(df["n"] == df["n"].max())]
            present = set(n40[n40["algorithm"] == alg]["instance"])
            all_inst = set(n40["instance"])
            missing = sorted(all_inst - present)
            lines.append(f"- `{alg}`: {c}/{expected} rows; missing at n={df['n'].max()}: "
                         f"{', '.join(missing) if missing else '(see n<max too)'}")
    else:
        lines.append("All arms have full coverage.")
    bad = df.groupby("instance")["mis_count"].nunique()
    lines.append("")
    lines.append("Cross-validation (all arms agree on MIS count per instance): "
                 + ("ALL AGREE" if (bad <= 1).all()
                    else f"DISAGREEMENT: {bad[bad > 1].index.tolist()}"))
    return "\n".join(lines)


def pivot_tomita_table(df):
    g = ratio_by_n(df, "bk_pivot", "bk_tomita", "median_seconds")
    g = g.reset_index()
    g.columns = ["n", "time_pivot_over_tomita", "n_instances"]
    pooled = ("## S5.2 Headline comparison: time(pivot) / time(tomita)\n\n"
              f"Source: `{FINAL.name}`. Ratio above 1 means bk_tomita is "
              "faster. Pooled over family, this trend is non-monotone "
              "(rises n=24->28, falls n=28->32, then rises again), not a "
              "clean crossover -- see the by-family table below for why.\n\n"
              + md_table(g, ["n", "time_pivot_over_tomita", "n_instances"]))

    w = df[df["algorithm"].isin(["bk_pivot", "bk_tomita"])].pivot_table(
        index=["instance", "n", "family"], columns="algorithm",
        values="median_seconds").dropna().reset_index()
    w["ratio"] = w["bk_pivot"] / w["bk_tomita"]
    fam = w.groupby(["family", "n"]).agg(
        time_pivot_over_tomita=("ratio", "median"),
        n_instances=("ratio", "count")).reset_index()
    by_family = ("### time(pivot) / time(tomita) by family\n\n"
                 f"Source: `{FINAL.name}`. The pooled trend above is the "
                 "average of three different verdicts, not a shared trend "
                 "with noise on top: ba is above 1 at every n (tomita "
                 "wins, 3 instances/n), er is below 1 at every n (the "
                 "cheap rule wins, 9 instances/n), ws moves from below 1 "
                 "at n=24,26 to mostly above 1 from n=28 on, with one dip "
                 "back below 1 at n=32 (3 instances/n) -- a real crossover "
                 "in trend, not a single clean threshold.\n\n"
                 + md_table(fam, ["family", "n", "time_pivot_over_tomita", "n_instances"]))
    return pooled + "\n\n" + by_family


def node_and_cost_table(df):
    # Each column is built from only the algorithms it needs, dropna'd on
    # only those columns. A single three-way pivot_table().dropna() would
    # drop every instance missing ANY of the three algorithms from ALL
    # columns, even ones (node_ratio, cost_ratio) that only need two of
    # them -- that silently shrank n=40 from 15 to 12 instances for
    # quantities bk_basic's outage has nothing to do with.
    d = df.copy()
    d["us_per_node"] = d["median_seconds"] * 1e6 / d["recursion_nodes"]

    def per_algorithm(alg, col):
        return d[d["algorithm"] == alg].groupby("n")[col].median()

    def ratio(a, b, col):
        w = d[d["algorithm"].isin([a, b])].pivot_table(
            index=["instance", "n"], columns="algorithm",
            values=col).dropna().reset_index()
        w["ratio"] = w[a] / w[b]
        return w.groupby("n")["ratio"].median()

    g = pd.DataFrame({
        "basic_over_tomita_nodes": ratio("bk_basic", "bk_tomita", "recursion_nodes"),
        "pivot_over_tomita_nodes": ratio("bk_pivot", "bk_tomita", "recursion_nodes"),
        "us_per_node_basic": per_algorithm("bk_basic", "us_per_node"),
        "us_per_node_pivot": per_algorithm("bk_pivot", "us_per_node"),
        "us_per_node_tomita": per_algorithm("bk_tomita", "us_per_node"),
        "cost_ratio_tomita_over_pivot": ratio("bk_tomita", "bk_pivot", "us_per_node"),
    }).reset_index().rename(columns={"index": "n"})
    return ("## S5.3 Decomposition: tree-size ratio and per-node cost\n\n"
             f"Source: `{FINAL.name}`. Node ratios: >1 means tomita searches "
             "less. us_per_node is median wall-clock microseconds per "
             "recursion node. cost_ratio_tomita_over_pivot is Tomita's "
             "per-node cost divided by the cheap rule's. time_ratio "
             "(S5.2) equals node_ratio / cost_ratio by construction -- "
             "this is an accounting identity, not an independent "
             "prediction; it decomposes the observed ratio, it does not "
             "verify it.\n\n"
             + md_table(g))


def two_by_two_table(df):
    pairs = [
        ("bk_pivot_p", "bk_pivot", "cheap selection: source P vs source P union X"),
        ("bk_ikgp", "bk_tomita", "scanning selection: source P vs source P union X"),
        ("bk_pivot_p", "bk_ikgp", "source P: cheap selection vs scanning selection"),
        ("bk_pivot", "bk_tomita", "source P union X: cheap selection vs scanning selection"),
    ]
    sections = ["## S5.4 The 2x2: pivot source vs selection cost\n",
                f"Source: `{FINAL.name}`. Ratio above 1 means the second "
                "named arm is faster (time) or searches less (nodes).\n"]
    for a, b, desc in pairs:
        gt = ratio_by_n(df, a, b, "median_seconds")
        gn = ratio_by_n(df, a, b, "recursion_nodes")
        if gt is None or gn is None:
            continue
        g = gt[["median"]].rename(columns={"median": "time_ratio"}).join(
            gn[["median"]].rename(columns={"median": "node_ratio"})).reset_index()
        sections.append(f"### {a} / {b}  ({desc})\n")
        sections.append(md_table(g, ["n", "time_ratio", "node_ratio"]))
        sections.append("")
    return "\n".join(sections)


def labelling_table(path):
    if not path.exists():
        return f"## S5.5 Labelling breakdown\n\nSource `{path.name}` not found.\n"
    df = pd.read_csv(path)
    w = df[df["algorithm"].isin(["bk_pivot", "bk_tomita"])].pivot_table(
        index=["instance", "n", "labelling"], columns="algorithm",
        values="median_seconds").dropna().reset_index()
    w["ratio"] = w["bk_pivot"] / w["bk_tomita"]
    g = w.groupby(["labelling", "n"])["ratio"].agg(
        median="median", n_instances="count").reset_index()
    return ("## S5.5 Labelling: time(pivot) / time(tomita) by labelling\n\n"
             f"Source: `{path.name}` (random, native, degeneracy pooled in "
             "the file but never in a statistic below). Ratio above 1 means "
             "bk_tomita is faster.\n\n" + md_table(g))


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
    if G.number_of_edges() == 0:
        return 0
    return max(nx.core_number(G).values())


def structural_correlation_table(df):
    # local recompute, deliberately independent of scripts/structure_analysis.py's
    # clique_overlap sampling RNG state, so this table is reproducible standalone.
    import random as _random

    def clique_overlap(G, sample=200, seed=0):
        sets = [frozenset(s) for s in nx.find_cliques(nx.complement(G))]
        if len(sets) < 2:
            return 0.0
        if len(sets) > sample:
            sets = _random.Random(seed).sample(sets, sample)
        tot = pairs = 0.0
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                u = len(sets[i] | sets[j])
                if u:
                    tot += len(sets[i] & sets[j]) / u
                pairs += 1
        return tot / pairs if pairs else 0.0

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
            "density": nx.density(G),
            "clique_overlap": clique_overlap(G),
        })
    feats = pd.DataFrame(rows)

    ratios = df[df["algorithm"].isin(["bk_pivot", "bk_tomita"])].pivot_table(
        index=["instance"], columns="algorithm",
        values="median_seconds").dropna().reset_index()
    ratios["tomita_gain"] = ratios["bk_pivot"] / ratios["bk_tomita"]
    nodes = df[df["algorithm"].isin(["bk_pivot", "bk_tomita"])].pivot_table(
        index="instance", columns="algorithm",
        values="recursion_nodes").dropna().reset_index()
    nodes["node_gain"] = nodes["bk_pivot"] / nodes["bk_tomita"]
    m = ratios.merge(feats, on="instance").merge(
        nodes[["instance", "node_gain"]], on="instance")

    cols = ["degeneracy", "comp_degeneracy", "clustering", "density", "clique_overlap"]
    rows = []
    for c in cols:
        rows.append({
            "predictor": c,
            "spearman_r_tomita_gain": spearman(m[c], m["tomita_gain"]),
            "spearman_r_node_gain": spearman(m[c], m["node_gain"]),
        })
    tbl = pd.DataFrame(rows)
    return ("## S5.6 Structural correlation table\n\n"
             f"Source: `{FINAL.name}` ONLY (random labelling). {len(m)} "
             "instances. tomita_gain = time(pivot)/time(tomita); "
             "node_gain = nodes(pivot)/nodes(tomita).\n\n"
             + md_table(tbl, floatfmt="{:+.3f}"))


def bitset_table(df):
    sections = ["## S5.7 Bitset speedups\n", f"Source: `{FINAL.name}`. "
                "Ratio above 1 means the bitset variant is faster.\n"]
    for base, bits in [("bk_basic", "bk_basic_bit"), ("bk_pivot", "bk_pivot_bit"),
                       ("bk_tomita", "bk_tomita_bit")]:
        g = ratio_by_n(df, base, bits, "median_seconds")
        if g is None:
            continue
        g = g.reset_index()
        g.columns = ["n", f"speedup_{base}", "n_instances"]
        sections.append(f"### {base} vs {bits}\n")
        sections.append(md_table(g, ["n", f"speedup_{base}", "n_instances"]))
        sections.append("")
    return "\n".join(sections)


def main():
    df = pd.read_csv(FINAL)
    parts = [
        "# Results tables",
        "Generated by `scripts/make_results_tables.py`. Numbers only; the "
        "source CSV is named under each heading so every number here can be "
        "traced back to the data it came from. Do not hand-edit; re-run the "
        "script instead.\n",
        coverage_section(df),
        pivot_tomita_table(df),
        node_and_cost_table(df),
        two_by_two_table(df),
        labelling_table(LABELLING),
        structural_correlation_table(df),
        bitset_table(df),
    ]
    OUT.write_text("\n\n".join(parts) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
