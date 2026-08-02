"""Correctness harness: hand-checked counts, cross-validation between
algorithms, and agreement with networkx as an outside reference.
"""

import networkx as nx
import pytest

from mis.naive import naive_mis
from mis.bron_kerbosch import bk_basic, bk_pivot, bk_tomita, count_nodes

ALGORITHMS = [naive_mis, bk_basic, bk_pivot, bk_tomita]

# counts worked out by hand:
# P2: {0},{1} = 2            P4: {0,2},{0,3},{1,3} = 3
# C5: {i,i+2} for each i = 5  K5: singletons = 5
# K33: the two sides = 2      star: centre or all leaves = 2
# empty graph: whole vertex set = 1
HAND_CHECKED = [
    ("P2", nx.path_graph(2), 2),
    ("P4", nx.path_graph(4), 3),
    ("C5", nx.cycle_graph(5), 5),
    ("K5", nx.complete_graph(5), 5),
    ("K33", nx.complete_bipartite_graph(3, 3), 2),
    ("star4", nx.star_graph(4), 2),
    ("empty4", nx.empty_graph(4), 1),
]


@pytest.mark.parametrize("alg", ALGORITHMS)
@pytest.mark.parametrize("name,G,expected", HAND_CHECKED)
def test_hand_checked_counts(alg, name, G, expected):
    assert len(set(alg(G))) == expected


@pytest.mark.parametrize("name,G", [
    ("petersen", nx.petersen_graph()),
    ("gnp_seeded", nx.gnp_random_graph(12, 0.4, seed=42)),
    ("barbell", nx.barbell_graph(4, 2)),
])
def test_cross_validation_identical_sets(name, G):
    # same actual sets, not just the same count
    results = [set(alg(G)) for alg in ALGORITHMS]
    assert all(r == results[0] for r in results[1:])
    # networkx find_cliques on the complement is an implementation
    # we didn't write, so it makes a good external check
    reference = {frozenset(c) for c in nx.find_cliques(nx.complement(G))}
    assert results[0] == reference


@pytest.mark.parametrize("alg", ALGORITHMS)
def test_every_output_is_maximal_independent(alg):
    from mis.core import is_maximal_independent_set
    G = nx.gnp_random_graph(10, 0.5, seed=7)
    for S in alg(G):
        assert is_maximal_independent_set(G, S)


@pytest.mark.parametrize("name,G", [
    ("petersen", nx.petersen_graph()),
    ("gnp20", nx.gnp_random_graph(20, 0.5, seed=3)),
])
def test_node_counts_consistent(name, G):
    # every variant must find the same sets, but pivoting should visit
    # fewer recursion nodes than no pivoting
    counts = {v: count_nodes(G, v) for v in ("basic", "pivot", "tomita")}
    found = {c[0] for c in counts.values()}
    assert len(found) == 1
    assert counts["pivot"][1] <= counts["basic"][1]
    assert counts["tomita"][1] <= counts["basic"][1]
