"""Experiment runner. One CSV row per (instance, algorithm).

Timings are medians over repeats (mean chases GC outliers). tracemalloc
only sees Python allocations, so peak_kib is comparative, not a real
process footprint.
"""

import csv
import platform
import statistics
import sys
import time
import tracemalloc
from dataclasses import dataclass, asdict, field

import networkx as nx

from mis.naive import naive_mis
from mis.bron_kerbosch import (bk_basic, bk_pivot, bk_tomita, bk_pivot_p,
                               bk_ikgp, count_nodes, count_stats)
from mis.bitset import (bk_basic_bitset, bk_pivot_bitset, bk_tomita_bitset,
                        count_nodes_bitset)
from mis import graphs

# name -> (callable, node-count variant or None, uses bitsets)
#
# The five set-based BK arms are a 2x2 over (pivot source, selection cost)
# with bk_basic as the no-pivot floor:
#
#                    cheap select      scanning select
#   source P | X     bk_pivot          bk_tomita
#   source P         bk_pivot_p        bk_ikgp
#
# Source is the axis Cazals & Karande argue about, cost is the axis San
# Segundo et al. argue about. bk_ikgp is the arm that makes these results
# comparable with both, and with Abu-Khzam et al.'s "Improved BK".
ALGORITHMS = {
    "naive":         (naive_mis,         None,      False),
    "bk_basic":      (bk_basic,          "basic",   False),
    "bk_pivot":      (bk_pivot,          "pivot",   False),
    "bk_tomita":     (bk_tomita,         "tomita",  False),
    "bk_pivot_p":    (bk_pivot_p,        "pivot_p", False),
    "bk_ikgp":       (bk_ikgp,           "ikgp",    False),
    "bk_basic_bit":  (bk_basic_bitset,   "basic",   True),
    "bk_pivot_bit":  (bk_pivot_bitset,   "pivot",   True),
    "bk_tomita_bit": (bk_tomita_bitset,  "tomita",  True),
}


@dataclass
class Instance:
    name: str
    family: str
    graph: nx.Graph = field(repr=False)
    n: int = 0
    m: int = 0
    p: float = float("nan")
    seed: int = -1
    # Vertex numbering is an experimental factor, not an incidental property:
    # the cheap pivot rule picks the lowest-numbered vertex, so it inherits
    # whatever structure the generator's labelling happens to carry. Headline
    # results use "random"; "native" and "degeneracy" appear only in the
    # labelling sub-experiment. See graphs.LABELLINGS.
    labelling: str = "random"

    def __post_init__(self):
        self.n = self.graph.number_of_nodes()
        self.m = self.graph.number_of_edges()


def timed_run(alg, G, repeats=5, time_limit=60.0, measure_memory=True):
    """(median seconds, count, peak KiB, all timings).

    Timing and memory are measured in SEPARATE passes. tracemalloc hooks
    every allocation, and the bitset implementations allocate a new int per
    set operation where the set-based ones mutate in place, so leaving it
    on during timing penalises the bitset variants specifically - it made
    them look about 2x worse than an uninstrumented run does.
    """
    timings = []
    count = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        found = sum(1 for _ in alg(G))
        elapsed = time.perf_counter() - t0
        timings.append(elapsed)
        count = found
        if elapsed > time_limit:
            break

    peak = float("nan")
    if measure_memory:
        tracemalloc.start()
        sum(1 for _ in alg(G))
        peak = tracemalloc.get_traced_memory()[1] / 1024
        tracemalloc.stop()

    return statistics.median(timings), count, peak, timings


def run_instance(inst: Instance, algorithms=None, repeats=5, time_limit=60.0,
                 skip_slow=None, measure_memory=True):
    """Benchmark every algorithm on one instance. skip_slow is a mutable set
    of algorithm names that have already blown the time limit and should not
    be run on anything larger."""
    algorithms = algorithms or list(ALGORITHMS)
    skip_slow = skip_slow if skip_slow is not None else set()
    rows = []
    for name in algorithms:
        if name in skip_slow:
            continue
        alg, variant, uses_bits = ALGORITHMS[name]
        median, count, peak, timings = timed_run(alg, inst.graph, repeats,
                                                 time_limit, measure_memory)
        # Counting is a separate pass from timing, as before. The set-based
        # arms also report unproductive calls and pivot-selection work; the
        # bitset arms only carry a node counter, so those columns stay blank.
        nodes = unproductive = pivot_cands = pivot_ints = ""
        if variant is not None:
            if uses_bits:
                nodes = count_nodes_bitset(inst.graph, variant)[1]
            else:
                _, st = count_stats(inst.graph, variant)
                nodes = st["nodes"]
                unproductive = st["unproductive"]
                pivot_cands = st["pivot_candidates"]
                pivot_ints = st["pivot_intersections"]
        rows.append({
            "instance": inst.name,
            "family": inst.family,
            "n": inst.n,
            "m": inst.m,
            "p": inst.p,
            "seed": inst.seed,
            "labelling": inst.labelling,
            "algorithm": name,
            "bitset": int(uses_bits),
            "mis_count": count,
            "median_seconds": median,
            "min_seconds": min(timings),
            "max_seconds": max(timings),
            "repeats": len(timings),
            "recursion_nodes": nodes,
            "unproductive_calls": unproductive,
            "pivot_candidates": pivot_cands,
            "pivot_intersections": pivot_ints,
            "peak_kib": round(peak, 1),
        })
        if median > time_limit:
            skip_slow.add(name)
    return rows


def build_sweep(ns, ps, seeds, families=("er",), ba_m=3,
                labellings=("random",)):
    """Instances for the main sweep.

    `labellings` is a tuple drawn from graphs.LABELLINGS. The default is
    ("random",) so that headline results are free of the generator's
    numbering conventions; pass ("native", "random", "degeneracy") to run
    the labelling sub-experiment, which multiplies the instance count.
    """
    out = []
    for family in families:
        for n in ns:
            for seed in seeds:
                if family == "er":
                    base = [(f"er_n{n}_p{p}_s{seed}", graphs.erdos_renyi(n, p, seed), p)
                            for p in ps]
                elif family == "ba":
                    if n <= ba_m:
                        continue
                    base = [(f"ba_n{n}_m{ba_m}_s{seed}",
                             graphs.barabasi_albert(n, ba_m, seed), float("nan"))]
                elif family == "ws":
                    base = [(f"ws_n{n}_s{seed}",
                             graphs.watts_strogatz(n, min(6, n - 1), 0.1, seed),
                             float("nan"))]
                else:
                    continue

                for stem, G, p in base:
                    for lab in labellings:
                        H = graphs.LABELLINGS[lab](G, seed)
                        # keep the stem stable when there is only one
                        # labelling, so existing result files still join
                        name = stem if labellings == ("random",) else f"{stem}_{lab}"
                        out.append(Instance(name, family, H, p=p, seed=seed,
                                            labelling=lab))
    return out


def write_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def environment():
    """Recorded with every campaign so results are interpretable later."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "networkx": nx.__version__,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
