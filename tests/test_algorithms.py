"""Correctness harness: hand-checked counts, cross-validation between
algorithms, and agreement with networkx as an outside reference.
"""

import networkx as nx
import pytest

from mis.naive import naive_mis
from mis.bron_kerbosch import (bk_basic, bk_pivot, bk_tomita, bk_pivot_p,
                               bk_ikgp, count_nodes, count_stats)

ALGORITHMS = [naive_mis, bk_basic, bk_pivot, bk_tomita, bk_pivot_p, bk_ikgp]
VARIANTS = ("basic", "pivot", "tomita", "pivot_p", "ikgp")

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
    # every variant must find the same sets, but any pivot rule should visit
    # fewer recursion nodes than no pivoting
    counts = {v: count_nodes(G, v) for v in VARIANTS}
    found = {c[0] for c in counts.values()}
    assert len(found) == 1
    for v in VARIANTS:
        if v != "basic":
            assert counts[v][1] <= counts["basic"][1]


@pytest.mark.parametrize("name,G", [
    ("petersen", nx.petersen_graph()),
    ("gnp20", nx.gnp_random_graph(20, 0.5, seed=3)),
    ("gnp24", nx.gnp_random_graph(24, 0.35, seed=9)),
])
def test_pivot_work_separates_cost_from_source(name, G):
    """The 2x2 has to be an actual 2x2: the cheap arms must do no set
    intersections during pivot selection and the scanning arms must, or the
    cost axis isn't isolated. The P-only arms must examine no more candidates
    than their P|X counterparts, or the source axis isn't isolated either."""
    st = {v: count_stats(G, v)[1] for v in VARIANTS}

    assert st["basic"]["pivot_candidates"] == 0
    for cheap in ("pivot", "pivot_p"):
        assert st[cheap]["pivot_candidates"] > 0
        assert st[cheap]["pivot_intersections"] == 0
    for scan in ("tomita", "ikgp"):
        assert st[scan]["pivot_intersections"] == st[scan]["pivot_candidates"]

    assert st["pivot_p"]["pivot_candidates"] <= st["pivot"]["pivot_candidates"]
    assert st["ikgp"]["pivot_candidates"] <= st["tomita"]["pivot_candidates"]


@pytest.mark.parametrize("name,G", [
    ("petersen", nx.petersen_graph()),
    ("gnp20", nx.gnp_random_graph(20, 0.5, seed=3)),
])
def test_unproductive_calls_bounded_and_ordered(name, G):
    """Unproductive calls are the wasted descent a pivot rule exists to
    remove, so no pivot rule may have more of them than no pivoting at all,
    and they can never exceed the total call count."""
    stats = {v: count_stats(G, v)[1] for v in VARIANTS}
    for v, st in stats.items():
        assert 0 <= st["unproductive"] <= st["nodes"]
    for v in VARIANTS:
        if v != "basic":
            assert stats[v]["unproductive"] <= stats["basic"]["unproductive"]


def _cazals_karande(n):
    """K_n u K_{1,n}, the worst case from Cazals & Karande (RR-5615, Fig. 4).

    Their construction is stated in CLIQUE space, and these algorithms
    enumerate cliques of the complement, so the whole graph is complemented -
    not each component separately, which is a different graph.
    """
    return nx.complement(nx.disjoint_union(nx.complete_graph(n),
                                           nx.star_graph(n)))


@pytest.mark.parametrize("n", [5, 6, 7, 8, 10])
def test_ikgp_pathology_on_cazals_karande_graph(n):
    """Greedy-from-P degrades on K_n u K_{1,n} where greedy-from-P|X does not:
    a pivot drawn from P is its own non-neighbour, so the branch loop always
    runs at least once. Guards the claim in the bk_ikgp docstring."""
    G = _cazals_karande(n)
    assert set(bk_ikgp(G)) == set(bk_tomita(G))
    assert count_nodes(G, "ikgp")[1] > count_nodes(G, "tomita")[1]


def test_ikgp_pathology_is_quadratic_not_constant():
    """The point of the construction is the growth rate, not one data point:
    Cazals & Karande report Theta(n^2) for pivoting from P against O(n) for
    pivoting from P|X. The excess should grow superlinearly in n."""
    excess = {}
    for n in (5, 10, 20):
        G = _cazals_karande(n)
        excess[n] = count_nodes(G, "ikgp")[1] - count_nodes(G, "tomita")[1]
    # doubling n should more than double the excess if it is quadratic
    assert excess[10] > 2 * excess[5]
    assert excess[20] > 2 * excess[10]
