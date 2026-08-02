"""Bitset implementations must agree with the set-based ones, and the
DIMACS parser must round-trip.
"""

import networkx as nx
import pytest

from mis.naive import naive_mis
from mis.bron_kerbosch import bk_basic, bk_pivot, bk_tomita, count_nodes
from mis.bitset import (bk_basic_bitset, bk_pivot_bitset, bk_tomita_bitset,
                        count_nodes_bitset)
from mis import graphs

PAIRS = [
    (bk_basic, bk_basic_bitset),
    (bk_pivot, bk_pivot_bitset),
    (bk_tomita, bk_tomita_bitset),
]

CASES = [
    ("P4", nx.path_graph(4)),
    ("C5", nx.cycle_graph(5)),
    ("K5", nx.complete_graph(5)),
    ("K33", nx.complete_bipartite_graph(3, 3)),
    ("petersen", nx.petersen_graph()),
    ("gnp15", nx.gnp_random_graph(15, 0.4, seed=11)),
    ("ba15", nx.barabasi_albert_graph(15, 3, seed=11)),
]


@pytest.mark.parametrize("name,G", CASES)
@pytest.mark.parametrize("set_alg,bit_alg", PAIRS)
def test_bitset_matches_set_version(name, G, set_alg, bit_alg):
    assert set(bit_alg(G)) == set(set_alg(G))


@pytest.mark.parametrize("name,G", CASES)
def test_bitset_matches_naive(name, G):
    assert set(bk_tomita_bitset(G)) == set(naive_mis(G))


@pytest.mark.parametrize("variant", ["basic", "pivot", "tomita"])
def test_node_counts_match_between_implementations(variant):
    # the bitset version is the same algorithm, so it must explore the same
    # tree, not merely produce the same answer
    G = nx.gnp_random_graph(16, 0.4, seed=5)
    assert count_nodes(G, variant) == count_nodes_bitset(G, variant)


def test_dimacs_round_trip(tmp_path):
    G = nx.gnp_random_graph(12, 0.4, seed=2)
    path = tmp_path / "test.clq"
    graphs.write_dimacs(G, path)
    H = graphs.read_dimacs(path)
    assert H.number_of_nodes() == G.number_of_nodes()
    assert H.number_of_edges() == G.number_of_edges()
    # same structure up to the 1-based relabelling
    assert nx.is_isomorphic(G, H)


def test_generators_are_seeded():
    a = graphs.erdos_renyi(20, 0.3, seed=42)
    b = graphs.erdos_renyi(20, 0.3, seed=42)
    c = graphs.erdos_renyi(20, 0.3, seed=43)
    assert set(a.edges) == set(b.edges)
    assert set(a.edges) != set(c.edges)
