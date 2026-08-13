"""Graph instances for the experiments: random generators and DIMACS files.

Everything is seeded so a run can be reproduced exactly. The seed used is
recorded per instance and ends up as a column in the results CSV.
"""

import random
from pathlib import Path

import networkx as nx

DIMACS_DIR = Path(__file__).resolve().parents[2] / "instances"


def erdos_renyi(n: int, p: float, seed: int) -> nx.Graph:
    """G(n, p). NB the complement of G(n,p) is G(n,1-p) and the algorithms
    run on the complement, so a sparse input is a dense working graph."""
    return nx.gnp_random_graph(n, p, seed=seed)


def barabasi_albert(n: int, m: int, seed: int) -> nx.Graph:
    """Preferential attachment. Heavy-tailed degrees, low degeneracy, which
    is the parameter the Eppstein-Loffler-Strash bound depends on."""
    return nx.barabasi_albert_graph(n, m, seed=seed)


def watts_strogatz(n: int, k: int, p: float, seed: int) -> nx.Graph:
    """Small-world graphs: high clustering, short paths."""
    return nx.watts_strogatz_graph(n, k, p, seed=seed)


def structured(name: str, n: int) -> nx.Graph:
    """Deterministic families, used as sanity anchors in the results."""
    builders = {
        "path": nx.path_graph,
        "cycle": nx.cycle_graph,
        "complete": nx.complete_graph,
        "star": nx.star_graph,
        "empty": nx.empty_graph,
    }
    if name not in builders:
        raise ValueError(f"unknown family: {name}")
    return builders[name](n)


def relabel_random(G: nx.Graph, seed: int) -> nx.Graph:
    """Randomly permute the vertex numbering.

    This is a control, not a cosmetic change. The cheap pivot rule selects
    the lowest-numbered vertex, so it reads structure out of the labelling
    whenever the labelling carries any. Barabasi-Albert numbers the seed
    clique first and attaches later vertices preferentially, so low labels
    correlate with high degree; Watts-Strogatz labels are positions on a
    ring, so neighbours are label-adjacent; Erdos-Renyi labels mean nothing.
    Measured effect on the cheap arm's node count is 0.80x to 1.68x
    depending on family, against 1.00x for the scanning arms, which are
    label-invariant up to tie-breaking.

    Randomising makes the cheap rule an honestly arbitrary pivot on every
    family, so cross-family comparisons are of the rule rather than of the
    generator's numbering conventions.
    """
    nodes = list(G.nodes)
    shuffled = nodes[:]
    random.Random(seed).shuffle(shuffled)
    return nx.relabel_nodes(G, dict(zip(nodes, shuffled)), copy=True)


def relabel_degeneracy(G: nx.Graph) -> nx.Graph:
    """Number vertices by a degeneracy ordering, lowest first.

    The deliberate counterpart to relabel_random: instead of removing the
    labelling effect it makes the labelling maximally informative, so the
    cheap rule becomes an ordering heuristic in the sense of Eppstein,
    Loffler and Strash (2010) and San Segundo et al. (2018) rather than an
    arbitrary choice. Used only in the labelling sub-experiment.
    """
    H = G.copy()
    order = []
    while H.number_of_nodes():
        v = min(H.nodes, key=lambda w: H.degree(w))
        order.append(v)
        H.remove_node(v)
    return nx.relabel_nodes(G, {v: i for i, v in enumerate(order)}, copy=True)


LABELLINGS = {
    "native": lambda G, seed: G,
    "random": relabel_random,
    "degeneracy": lambda G, seed: relabel_degeneracy(G),
}


def read_dimacs(path) -> nx.Graph:
    """DIMACS .clq/.col reader (Johnson & Trick 1996 challenge format):
    'c' comment, 'p edge <n> <m>', 'e <u> <v>'. Vertices stay 1-based so
    they match the published instance descriptions.
    """
    G = nx.Graph()
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            tag = parts[0]
            if tag == "p":
                G.add_nodes_from(range(1, int(parts[2]) + 1))
            elif tag == "e":
                G.add_edge(int(parts[1]), int(parts[2]))
    return G


def load_dimacs_instances(directory=DIMACS_DIR, max_nodes=None):
    """Load every .clq file in a directory, optionally skipping big ones."""
    directory = Path(directory)
    out = {}
    if not directory.exists():
        return out
    for path in sorted(directory.glob("*.clq")):
        G = read_dimacs(path)
        if max_nodes is None or G.number_of_nodes() <= max_nodes:
            out[path.stem] = G
    return out


def write_dimacs(G: nx.Graph, path, name="generated"):
    """Write a graph in DIMACS format (used to check the parser round-trips)."""
    with open(path, "w") as fh:
        fh.write(f"c {name}\n")
        fh.write(f"p edge {G.number_of_nodes()} {G.number_of_edges()}\n")
        relabel = {v: i + 1 for i, v in enumerate(G.nodes)}
        for u, v in G.edges:
            fh.write(f"e {relabel[u]} {relabel[v]}\n")
