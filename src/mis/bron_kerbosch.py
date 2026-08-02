"""Bron-Kerbosch enumeration of maximal independent sets.

Based on Bron & Kerbosch, "Algorithm 457: Finding All Cliques of an
Undirected Graph", CACM 16(9), 1973 (basic version and their version 2
with pivoting), and Tomita, Tanaka & Takahashi, "The worst-case time
complexity for generating all maximal cliques and computational
experiments", TCS 363(1), 2006, for the pivot selection rule.

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


def _bk(R, P, X, adj, counter=None):
    if counter is not None:
        counter[0] += 1
    if not P and not X:
        yield R
        return
    for v in sorted(P):
        yield from _bk(R | {v}, P & adj[v], X & adj[v], adj, counter)
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


def _bk_pivot(R, P, X, adj, counter=None):
    if counter is not None:
        counter[0] += 1
    if not P and not X:
        yield R
        return
    # lowest-numbered candidate, not an arbitrary one: iteration order over a
    # Python set is not something to build reproducible experiments on, and a
    # fixed rule makes the search tree identical to the bitset implementation
    u = min(P | X)
    for v in sorted(P - adj[u]):
        yield from _bk_pivot(R | {v}, P & adj[v], X & adj[v], adj, counter)
        P.remove(v)
        X.add(v)


def bk_tomita(G: nx.Graph):
    """Tomita et al. (2006) pivot selection: pick u maximising |P & N(u)|,
    which minimises the branching set P - N(u). Gives O(3^(n/3)) worst case,
    optimal by Moon & Moser. Costs a scan per node though, so whether it
    wins in practice is one of the things I'm measuring.
    """
    adj = _complement_adj(G)
    yield from _bk_tomita(frozenset(), set(G.nodes), set(), adj)


def _bk_tomita(R, P, X, adj, counter=None):
    if counter is not None:
        counter[0] += 1
    if not P and not X:
        yield R
        return
    # ties broken by lowest vertex number, so the tree is reproducible
    u = max(sorted(P | X), key=lambda w: len(P & adj[w]))
    for v in sorted(P - adj[u]):
        yield from _bk_tomita(R | {v}, P & adj[v], X & adj[v], adj, counter)
        P.remove(v)
        X.add(v)


def count_nodes(G: nx.Graph, variant: str = "tomita"):
    """(number of maximal independent sets, recursion nodes visited).
    Node count is machine independent, unlike the timings."""
    adj = _complement_adj(G)
    counter = [0]
    recursions = {"basic": _bk, "pivot": _bk_pivot, "tomita": _bk_tomita}
    if variant not in recursions:
        raise ValueError(f"unknown variant: {variant}")
    found = sum(1 for _ in recursions[variant](
        frozenset(), set(G.nodes), set(), adj, counter))
    return found, counter[0]
