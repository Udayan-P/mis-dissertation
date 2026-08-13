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

FIVE VARIANTS, and they differ *only* in how the pivot is chosen. They
share one recursion (_bk_core), so any difference in node count, work or
time is attributable to pivot selection and nothing else. That is the
whole point of the ablation, and it is worth saying explicitly in the
write-up.

    variant   pivot source   selection cost   corresponds to
    basic     none           -                BK 1973 version 1
    pivot     P | X          cheap            no published counterpart
    tomita    P | X          scan             BK 1973 version 2 / Tomita 2006
    pivot_p   P              cheap            close to Koch's IK_RP
    ikgp      P              scan             Koch's IK_GP; Cazals & Karande's
                                              IK_GP; Abu-Khzam's "Improved BK"

That is a 2x2 over (source, cost) with `basic` as the no-pivot floor.
Source is the axis Cazals & Karande argue about (P vs P|X); cost is the
axis San Segundo et al. argue about. Varying both is the point.

Note on "cheap": the cheap rule picks the lowest-numbered vertex, which is
a property of the *labelling*, not of the graph. Under an arbitrary
labelling it behaves like a random pivot (Koch's IK_RP); under a
degeneracy or max-degree ordering it would behave like San Segundo's
ordering heuristic. Whatever the instance loader does, say so in the
write-up - it is a real confound otherwise.
"""

from itertools import chain
import time

import networkx as nx

_EMPTY = frozenset()


def _complement_adj(G: nx.Graph) -> dict:
    nodes = set(G.nodes)
    return {v: nodes - {v} - set(G.neighbors(v)) for v in nodes}


# --------------------------------------------------------------------------
# stats
# --------------------------------------------------------------------------

def new_stats(time_pivot: bool = False) -> dict:
    """Counters for one run.

    nodes                recursion calls entered
    unproductive         calls whose whole subtree yielded no maximal set.
                         Cazals & Karande (RR-5615) call these non-productive
                         calls and report them separately from total calls;
                         they isolate exactly the wasted descent a pivot rule
                         exists to remove, which raw node count does not.
    pivot_candidates     vertices examined during pivot selection
    pivot_intersections  set intersections performed during pivot selection.
                         Zero for the cheap rules, |P|+|X| per node for the
                         scan rules - this is the quantity that separates
                         "cost" from "source" and it is machine independent.
    pivot_seconds        wall-clock inside pivot selection, only if
                         time_pivot=True. See the warning on count_stats.
    """
    s = {"nodes": 0, "unproductive": 0,
         "pivot_candidates": 0, "pivot_intersections": 0}
    if time_pivot:
        s["pivot_seconds"] = 0.0
    return s


# --------------------------------------------------------------------------
# pivot rules
# --------------------------------------------------------------------------

def _pivot_cheap(P, X, adj, stats):
    """Lowest-numbered vertex of P | X.

    Does NOT materialise the union. The previous version computed
    `min(P | X)`, which allocates a set of size |P|+|X| on every node
    before scanning it - so the "cheap" arm was not cheap, and the
    per-node cost ratio against the scan arm was understated.
    """
    if stats is not None:
        stats["pivot_candidates"] += len(P) + len(X)
    if not X:
        return min(P)
    if not P:
        return min(X)
    return min(min(P), min(X))


def _pivot_greedy(P, X, adj, stats):
    """u in P | X maximising |P & N(u)|, ties broken by lowest vertex number.

    Same vertex as the previous `max(sorted(P | X), key=...)`, but without
    the sort. That sort was O(k log k) per node purely for tie-breaking and
    was paid by the scan arms only, so it inflated the very quantity the
    experiment is trying to measure. The tuple key reproduces the tie-break
    exactly in a single O(k) pass.
    """
    if stats is not None:
        k = len(P) + len(X)
        stats["pivot_candidates"] += k
        stats["pivot_intersections"] += k
    return min(chain(P, X), key=lambda w: (-len(P & adj[w]), w))


def _pivot_cheap_p(P, X, adj, stats):
    """Cheap rule restricted to P."""
    return _pivot_cheap(P, _EMPTY, adj, stats)


def _pivot_greedy_p(P, X, adj, stats):
    """Greedy rule restricted to P - Koch's IK_GP."""
    return _pivot_greedy(P, _EMPTY, adj, stats)


# --------------------------------------------------------------------------
# shared recursion
# --------------------------------------------------------------------------

def _bk_core(R, P, X, adj, pivot, stats=None, time_pivot=False):
    """One recursion for every variant. `pivot` is None (no pivoting) or a
    callable (P, X, adj, stats) -> vertex.

    `produced` tracks whether this call's subtree yielded anything, which is
    what makes the unproductive-call count exact rather than inferred.
    """
    if stats is not None:
        stats["nodes"] += 1
    produced = 0

    if not P and not X:
        produced = 1
        yield R
    elif not P:
        # Nothing to branch on and X is non-empty, so R is not maximal.
        # Returning here rather than after choosing a pivot matters for the
        # cost comparison: otherwise the scan arms pay a full |P|+|X| scan to
        # discover there is nothing to do, which biases the measurement
        # against them. The search tree is unchanged either way.
        pass
    else:
        if pivot is None:
            branch = sorted(P)
        else:
            if time_pivot and stats is not None:
                t0 = time.perf_counter()
                u = pivot(P, X, adj, stats)
                stats["pivot_seconds"] += time.perf_counter() - t0
            else:
                u = pivot(P, X, adj, stats)
            branch = sorted(P - adj[u])

        for v in branch:
            for S in _bk_core(R | {v}, P & adj[v], X & adj[v], adj,
                              pivot, stats, time_pivot):
                produced += 1
                yield S
            P.remove(v)
            X.add(v)

    if stats is not None and produced == 0:
        stats["unproductive"] += 1


PIVOTS = {
    "basic":   None,
    "pivot":   _pivot_cheap,
    "tomita":  _pivot_greedy,
    "pivot_p": _pivot_cheap_p,
    "ikgp":    _pivot_greedy_p,
}


def _run(G, variant, stats=None, time_pivot=False):
    adj = _complement_adj(G)
    return _bk_core(frozenset(), set(G.nodes), set(), adj,
                    PIVOTS[variant], stats, time_pivot)


# --------------------------------------------------------------------------
# public generators
# --------------------------------------------------------------------------

def bk_basic(G: nx.Graph):
    """Bron-Kerbosch without pivoting. The no-pivot floor of the design."""
    yield from _run(G, "basic")


def bk_pivot(G: nx.Graph):
    """Pivot from P | X, chosen cheaply (lowest vertex number).

    Any maximal clique contains the pivot u or a non-neighbour of u (if
    not, u could be added and the clique wasn't maximal), so only the
    non-neighbours of u need branching on. A fixed rule rather than an
    arbitrary one, because iteration order over a Python set is not
    something to build reproducible experiments on, and it keeps the search
    tree identical to the bitset implementation.
    """
    yield from _run(G, "pivot")


def bk_tomita(G: nx.Graph):
    """Tomita et al. (2006) pivot selection: u maximising |P & N(u)| over
    P | X, which minimises the branching set P - N(u). Gives O(3^(n/3))
    worst case, optimal by Moon & Moser. Costs a scan per node though, so
    whether it wins in practice is one of the things I'm measuring.

    Provenance worth recording: the same selection rule already appears in
    Bron & Kerbosch's version 2, where the fixed point is taken from
    `not` union `candidates` so as to minimise its count of disconnections
    to `candidates`. Tomita et al.'s contribution is the proof.
    """
    yield from _run(G, "tomita")


def bk_pivot_p(G: nx.Graph):
    """Pivot from P only, chosen cheaply. Completes the 2x2 against
    bk_pivot: same cost, different source."""
    yield from _run(G, "pivot_p")


def bk_ikgp(G: nx.Graph):
    """Koch's IK_GP: greedy pivot from P only.

    Same selection cost as bk_tomita, different source, so the pair
    isolates the effect of drawing the pivot from P rather than P | X.
    This is the arm that makes the results comparable with Cazals &
    Karande (RR-5615, IK_GP vs IK_GPX) and with Abu-Khzam et al.'s
    "Improved BK", both of which pivot from P.

    Expect it to be poor on graphs containing a high-degree vertex
    disconnected from the clique being built - a pivot in P is its own
    non-neighbour, so the branch loop always runs at least once. Cazals &
    Karande's K_n u K_{1,n} makes this Theta(n^2) where pivoting from
    P | X is O(n). That is a real property of the rule, not a bug.
    """
    yield from _run(G, "ikgp")


# --------------------------------------------------------------------------
# instrumentation
# --------------------------------------------------------------------------

def count_stats(G: nx.Graph, variant: str = "tomita", time_pivot: bool = False):
    """(number of maximal independent sets, stats dict). See new_stats.

    Counting is a separate pass from timing, as with the existing node
    counts: the bookkeeping is cheap but not free, and time_pivot in
    particular calls perf_counter twice per node. Never read pivot_seconds
    and a wall-clock figure off the same run.
    """
    if variant not in PIVOTS:
        raise ValueError(f"unknown variant: {variant}")
    stats = new_stats(time_pivot)
    found = sum(1 for _ in _run(G, variant, stats, time_pivot))
    return found, stats


def count_nodes(G: nx.Graph, variant: str = "tomita"):
    """(number of maximal independent sets, recursion nodes visited).
    Node count is machine independent, unlike the timings."""
    found, stats = count_stats(G, variant)
    return found, stats["nodes"]
