"""Shared definitions used by the algorithms and the test harness."""

from itertools import combinations

import networkx as nx


def is_independent_set(G: nx.Graph, S: set) -> bool:
    return all(not G.has_edge(u, v) for u, v in combinations(S, 2))


def is_maximal_independent_set(G: nx.Graph, S: set) -> bool:
    # maximal = independent, and every vertex outside S has a neighbour in S
    # (otherwise that vertex could be added)
    if not is_independent_set(G, S):
        return False
    S = set(S)
    return all(any(w in S for w in G.neighbors(v)) for v in G.nodes if v not in S)
