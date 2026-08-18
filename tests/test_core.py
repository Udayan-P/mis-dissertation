"""Sanity tests for core predicates: the seed of the W1 correctness harness."""

import networkx as nx

from mis.core import is_independent_set, is_maximal_independent_set


def test_independent_set_triangle():
    G = nx.complete_graph(3)
    assert is_independent_set(G, {0})
    assert not is_independent_set(G, {0, 1})
    assert is_independent_set(G, set())


def test_maximal_on_path4():
    # P4: 0-1-2-3. {0, 2} is maximal; {0} is independent but not maximal
    # (vertex 2 or 3 could still be added); {1, 3} is also maximal.
    G = nx.path_graph(4)
    assert is_maximal_independent_set(G, {0, 2})
    assert is_maximal_independent_set(G, {1, 3})
    assert not is_maximal_independent_set(G, {0})


def test_maximal_vs_maximum_star():
    # Star K_{1,3}: centre 0, leaves 1-3. {0} is maximal but not maximum;
    # {1, 2, 3} is both. The classic maximal != maximum example.
    G = nx.star_graph(3)
    assert is_maximal_independent_set(G, {0})
    assert is_maximal_independent_set(G, {1, 2, 3})
