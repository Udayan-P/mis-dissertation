"""Run all algorithms on some named graphs and check the counts agree.

Usage: python scripts/demo.py
"""

import time

import networkx as nx

from mis.naive import naive_mis
from mis.bron_kerbosch import bk_basic, bk_pivot

GRAPHS = [
    ("P4 path", nx.path_graph(4)),
    ("C5 cycle", nx.cycle_graph(5)),
    ("K5 complete", nx.complete_graph(5)),
    ("K33 bipartite", nx.complete_bipartite_graph(3, 3)),
    ("Petersen", nx.petersen_graph()),
    ("G(15, 0.3)", nx.gnp_random_graph(15, 0.3, seed=1)),
    ("G(18, 0.5)", nx.gnp_random_graph(18, 0.5, seed=1)),
]

ALGS = [("naive 2^n", naive_mis), ("BK basic", bk_basic), ("BK pivot", bk_pivot)]


def run():
    print(f"{'graph':<15} {'n':>3} " + "".join(f"{name:>14}" for name, _ in ALGS)
          + f"{'agree':>8}")
    print("-" * 65)
    for gname, G in GRAPHS:
        counts, times = [], []
        for _, alg in ALGS:
            t0 = time.perf_counter()
            found = set(alg(G))
            times.append(time.perf_counter() - t0)
            counts.append(len(found))
        agree = "yes" if len(set(counts)) == 1 else "NO!"
        cells = "".join(f"{c:>7} {t*1000:5.1f}ms" for c, t in zip(counts, times))
        print(f"{gname:<15} {G.number_of_nodes():>3} {cells}{agree:>6}")
    print("\ncounts = number of maximal independent sets, all three must agree")


if __name__ == "__main__":
    run()
