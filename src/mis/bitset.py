"""Bitset versions of the BK variants.

Vertices are relabelled 0..n-1 and each vertex set is a Python int, bit i
meaning vertex i is present. Bit-parallel clique enumeration follows
San Segundo, Artieda & Strash (Computers & OR 92, 2018), though they use
C++ word arrays and this is just Python's arbitrary-precision ints.

Needs int.bit_count() (3.10+) for the Tomita pivot popcount.
"""

import networkx as nx


def _bitset_adj(G: nx.Graph):
    """(order, complement adjacency masks). order maps bit position back to
    the original vertex label."""
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
