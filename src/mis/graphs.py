"""Graph instances for the experiments: random generators and DIMACS files.

Everything is seeded so a run can be reproduced exactly. The seed used is
recorded per instance and ends up as a column in the results CSV.
"""

from pathlib import Path

import networkx as nx

DIMACS_DIR = Path(__file__).resolve().parents[2] / "instances"


def erdos_renyi(n: int, p: float, seed: int) -> nx.Graph:
    """G(n, p): each edge present independently with probability p.

    Density is controlled directly, which is what the crossover analysis
    needs. Note the complement of G(n, p) is G(n, 1-p), so sparse inputs
    give dense complements and vice versa; since the algorithms run on the
    complement this matters for interpreting the results.
    """
    return nx.gnp_random_graph(n, p, seed=seed)


def barabasi_albert(n: int, m: int, seed: int) -> nx.Graph:
    """Preferential attachment: heavy-tailed degrees, low degeneracy.

    Included because real networks look more like this than like G(n, p),
    and because degeneracy is the parameter the Eppstein-Loffler-Strash
    bound depends on.
    """
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


def read_dimacs(path) -> nx.Graph:
    """Read a DIMACS .clq / .col file.

    Format (Johnson & Trick 1996, Second DIMACS Implementation Challenge):
        c comment line
        p edge <nodes> <edges>
        e <u> <v>
    Vertices are 1-based in the file; kept as-is so instance names and
    vertex numbers match the published benchmark descriptions.
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
