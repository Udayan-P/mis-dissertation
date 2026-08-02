"""Bitset versions of the Bron-Kerbosch variants.

Vertices are relabelled 0..n-1 and each vertex set is a Python integer
whose i-th bit says whether vertex i is in the set. Intersection is &,
union is |, difference is & ~, membership is a shift and test. Python's
integers are arbitrary precision, so this works for any n, and the set
operations happen in C over machine words instead of in the interpreter
over hash tables.

The idea (using bit-parallelism for maximal clique enumeration) follows
San Segundo, Artieda & Strash, "Efficiently enumerating all maximal
cliques with bit-parallelism", Computers & Operations Research 92, 2018.
Their implementation is in C++ with explicit word arrays; using Python
integers is the same trick at a coarser grain.

Popcount: int.bit_count() exists from Python 3.10, and is needed for the
Tomita pivot rule, which counts |P & N(u)|.
"""

import networkx as nx


def _bitset_adj(G: nx.Graph):
    """Return (order, adjacency masks of the COMPLEMENT graph).

    order maps bit position -> original vertex label, so results can be
    translated back. As elsewhere we work on the complement, because
    maximal independent sets of G are maximal cliques of the complement.
    """
    order = list(G.nodes)
    index = {v: i for i, v in enumerate(order)}
    n = len(order)
    full = (1 << n) - 1
    adj = [0] * n
    for v in order:
        i = index[v]
        # bits set for the true neighbours of v, then complement them
        nbrs = 0
        for w in G.neighbors(v):
            nbrs |= 1 << index[w]
        adj[i] = full & ~nbrs & ~(1 << i)
    return order, adj


def _bits(mask: int):
    """Yield the positions of the set bits of mask, lowest first."""
    while mask:
        low = mask & -mask          # isolate lowest set bit
        yield low.bit_length() - 1
        mask ^= low


def _decode(order, mask: int) -> frozenset:
    return frozenset(order[i] for i in _bits(mask))


def bk_basic_bitset(G: nx.Graph):
    order, adj = _bitset_adj(G)
    n = len(order)
    yield from (_decode(order, m)
                for m in _bk_bits(0, (1 << n) - 1, 0, adj))


def _bk_bits(R, P, X, adj, counter=None):
    if counter is not None:
        counter[0] += 1
    if not P and not X:
        yield R
        return
    for v in list(_bits(P)):
        bit = 1 << v
        yield from _bk_bits(R | bit, P & adj[v], X & adj[v], adj, counter)
        P &= ~bit
        X |= bit


def bk_pivot_bitset(G: nx.Graph):
    order, adj = _bitset_adj(G)
    n = len(order)
    yield from (_decode(order, m)
                for m in _bk_pivot_bits(0, (1 << n) - 1, 0, adj))


def _bk_pivot_bits(R, P, X, adj, counter=None):
    if counter is not None:
        counter[0] += 1
    if not P and not X:
        yield R
        return
    u = next(_bits(P | X))                 # arbitrary pivot: lowest set bit
    for v in list(_bits(P & ~adj[u])):
        bit = 1 << v
        yield from _bk_pivot_bits(R | bit, P & adj[v], X & adj[v], adj, counter)
        P &= ~bit
        X |= bit


def bk_tomita_bitset(G: nx.Graph):
    order, adj = _bitset_adj(G)
    n = len(order)
    yield from (_decode(order, m)
                for m in _bk_tomita_bits(0, (1 << n) - 1, 0, adj))


def _bk_tomita_bits(R, P, X, adj, counter=None):
    if counter is not None:
        counter[0] += 1
    if not P and not X:
        yield R
        return
    # Tomita pivot: maximise |P & N(u)|, which is a popcount here
    u = max(_bits(P | X), key=lambda w: (P & adj[w]).bit_count())
    for v in list(_bits(P & ~adj[u])):
        bit = 1 << v
        yield from _bk_tomita_bits(R | bit, P & adj[v], X & adj[v], adj, counter)
        P &= ~bit
        X |= bit


BITSET_VARIANTS = {
    "basic": _bk_bits,
    "pivot": _bk_pivot_bits,
    "tomita": _bk_tomita_bits,
}


def count_nodes_bitset(G: nx.Graph, variant: str = "tomita"):
    """(number of maximal independent sets, recursion nodes visited)."""
    order, adj = _bitset_adj(G)
    n = len(order)
    if variant not in BITSET_VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    counter = [0]
    found = sum(1 for _ in BITSET_VARIANTS[variant](0, (1 << n) - 1, 0, adj, counter))
    return found, counter[0]
