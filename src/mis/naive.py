"""Naive 2^n baseline: check every subset of V for maximal independence.

Unusable beyond n ~ 20 but obviously correct, so it anchors the test
harness and gives the baseline for the empirical comparison.
"""

from itertools import combinations

import networkx as nx

from mis.core import is_maximal_independent_set


def naive_mis(G: nx.Graph):
    """Yield all maximal independent sets of G as frozensets."""
    nodes = list(G.nodes)
    # can't stop at the first size that works: maximal sets of different
    # sizes coexist (star graph: centre alone, or all the leaves)
    for k in range(len(nodes) + 1):
        for subset in combinations(nodes, k):
            S = frozenset(subset)
            if is_maximal_independent_set(G, S):
                yield S
