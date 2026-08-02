"""Experiment runner.

Produces one CSV row per (instance, algorithm) with timing, output size,
search effort and memory, so the analysis is done on data rather than on
whatever was printed to the terminal at the time.

Design decisions worth defending in the report:
  * median of several repeats, not mean, because occasional GC pauses and
    OS scheduling produce outliers that a mean would follow;
  * recursion node counts recorded alongside wall-clock, because node
    counts are machine independent and comparable across implementations;
  * peak memory measured with tracemalloc, which counts Python allocations
    only, so it is a relative measure between algorithms rather than an
    absolute process footprint;
  * a per-instance time limit, so a slow algorithm on a large instance
    truncates the sweep for that algorithm instead of stalling the run;
  * every generated instance records its seed, so any row can be rebuilt.
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
from mis.bron_kerbosch import bk_basic, bk_pivot, bk_tomita, count_nodes
from mis.bitset import (bk_basic_bitset, bk_pivot_bitset, bk_tomita_bitset,
                        count_nodes_bitset)
from mis import graphs

# name -> (callable, node-count variant or None, uses bitsets)
ALGORITHMS = {
    "naive":         (naive_mis,         None,     False),
    "bk_basic":      (bk_basic,          "basic",  False),
    "bk_pivot":      (bk_pivot,          "pivot",  False),
    "bk_tomita":     (bk_tomita,         "tomita", False),
    "bk_basic_bit":  (bk_basic_bitset,   "basic",  True),
    "bk_pivot_bit":  (bk_pivot_bitset,   "pivot",  True),
    "bk_tomita_bit": (bk_tomita_bitset,  "tomita", True),
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

    def __post_init__(self):
        self.n = self.graph.number_of_nodes()
        self.m = self.graph.number_of_edges()


def timed_run(alg, G, repeats=5, time_limit=60.0):
    """Return (median seconds, count, peak KiB, all timings) or None if the
    first run already exceeds the time limit."""
    timings = []
    count = None
    tracemalloc.start()
    for i in range(repeats):
        t0 = time.perf_counter()
        found = sum(1 for _ in alg(G))
        elapsed = time.perf_counter() - t0
        timings.append(elapsed)
        count = found
        if elapsed > time_limit:
            break
    peak = tracemalloc.get_traced_memory()[1] / 1024
    tracemalloc.stop()
    return statistics.median(timings), count, peak, timings


def run_instance(inst: Instance, algorithms=None, repeats=5, time_limit=60.0,
                 skip_slow=None):
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
                                                 time_limit)
        nodes = ""
        if variant is not None:
            counter = count_nodes_bitset if uses_bits else count_nodes
            nodes = counter(inst.graph, variant)[1]
        rows.append({
            "instance": inst.name,
            "family": inst.family,
            "n": inst.n,
            "m": inst.m,
            "p": inst.p,
            "seed": inst.seed,
            "algorithm": name,
            "bitset": int(uses_bits),
            "mis_count": count,
            "median_seconds": median,
            "min_seconds": min(timings),
            "max_seconds": max(timings),
            "repeats": len(timings),
            "recursion_nodes": nodes,
            "peak_kib": round(peak, 1),
        })
        if median > time_limit:
            skip_slow.add(name)
    return rows


def build_sweep(ns, ps, seeds, families=("er",), ba_m=3):
    """Instances for the main sweep."""
    out = []
    for family in families:
        for n in ns:
            for seed in seeds:
                if family == "er":
                    for p in ps:
                        G = graphs.erdos_renyi(n, p, seed)
                        out.append(Instance(f"er_n{n}_p{p}_s{seed}", "er", G,
                                            p=p, seed=seed))
                elif family == "ba":
                    if n <= ba_m:
                        continue
                    G = graphs.barabasi_albert(n, ba_m, seed)
                    out.append(Instance(f"ba_n{n}_m{ba_m}_s{seed}", "ba", G,
                                        seed=seed))
                elif family == "ws":
                    G = graphs.watts_strogatz(n, min(6, n - 1), 0.1, seed)
                    out.append(Instance(f"ws_n{n}_s{seed}", "ws", G, seed=seed))
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
