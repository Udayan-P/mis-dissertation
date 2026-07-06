"""Bron-Kerbosch enumeration of maximal independent sets.

Based on Bron & Kerbosch, "Algorithm 457: Finding All Cliques of an
Undirected Graph", CACM 16(9), 1973 (basic version and their version 2
with pivoting). Tomita et al. (2006) pivot selection to be added later.

Uses the standard duality: S is a maximal independent set of G iff S is
a maximal clique of the complement graph. So we enumerate cliques over
the complement's adjacency, built once up front.

State carried by the recursion:
    R - clique built so far
    P - vertices that can still extend R
    X - vertices that could extend R but were finished on an earlier branch
R is maximal exactly when P and X are both empty. X is what stops
non-maximal and duplicate outputs.
"""

import networkx as nx


def _complement_adj(G: nx.Graph) -> dict:
    nodes = set(G.nodes)
    return {v: nodes - {v} - set(G.neighbors(v)) for v in nodes}


def bk_basic(G: nx.Graph):
    """Bron-Kerbosch without pivoting."""
    adj = _complement_adj(G)
    yield from _bk(frozenset(), set(G.nodes), set(), adj)


def _bk(R, P, X, adj):
    if not P and not X:
        yield R
        return
    for v in list(P):
        yield from _bk(R | {v}, P & adj[v], X & adj[v], adj)
        P.remove(v)
        X.add(v)


def bk_pivot(G: nx.Graph):
    """Bron-Kerbosch with pivoting, arbitrary pivot choice.

    Any maximal clique contains the pivot u or a non-neighbour of u
    (if not, u could be added and the clique wasn't maximal). So only
    the non-neighbours of u need to be branched on.
    """
    adj = _complement_adj(G)
    yield from _bk_pivot(frozenset(), set(G.nodes), set(), adj)


def _bk_pivot(R, P, X, adj):
    if not P and not X:
        yield R
        return
    u = next(iter(P | X))
    for v in list(P - adj[u]):
        yield from _bk_pivot(R | {v}, P & adj[v], X & adj[v], adj)
        P.remove(v)
        X.add(v)
