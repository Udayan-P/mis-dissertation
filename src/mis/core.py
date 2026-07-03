"""Core definitions used by every algorithm and by the correctness harness.

A set S of vertices is *independent* if no two vertices in S are adjacent.
S is *maximal* if it is independent and no vertex can be added while keeping
independence (contrast with *maximum*: an independent set of largest size).
"""

from itertools import combinations

import networkx as nx


def is_independent_set(G: nx.Graph, S: set) -> bool:
    """True iff S is an independent set in G."""
    return all(not G.has_edge(u, v) for u, v in combinations(S, 2))


def is_maximal_independent_set(G: nx.Graph, S: set) -> bool:
    """True iff S is a *maximal* independent set in G.

    S is maximal iff it is independent and every vertex outside S has at
    least one neighbour in S (otherwise it could be added).
    """
    if not is_independent_set(G, S):
        return False
    S = set(S)
    return all(any(w in S for w in G.neighbors(v)) for v in G.nodes if v not in S)
