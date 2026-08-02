"""Test whether the bitset speedup collapse is CPython's integer digit size.

The crossover sweep showed bitsets ~1.2-1.3x faster up to n=30 and ~0.5x
(i.e. 2x slower) from n=32 on. CPython stores ints in 30-bit digits
(sys.int_info.bits_per_digit), so a vertex-set mask fits in one digit while
n <= 30 and needs two from n = 31. If that's the cause, the speedup should
fall off a cliff between n=30 and n=31, not decline gradually.

This sweeps n one at a time across the boundary to see which it is.

Usage: python scripts/digit_boundary.py
"""

import statistics
import sys
import time

import networkx as nx

from mis.bron_kerbosch import bk_pivot, bk_tomita
from mis.bitset import bk_pivot_bitset, bk_tomita_bitset

PAIRS = [("bk_pivot", bk_pivot, bk_pivot_bitset),
         ("bk_tomita", bk_tomita, bk_tomita_bitset)]


def median_time(alg, G, repeats=5):
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        sum(1 for _ in alg(G))
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def raw_int_ops(bits_wide, iterations=200_000):
    """Time the bare integer operations the bitset code relies on, for a mask
    of a given width. Isolates the representation from the algorithms."""
    a = (1 << bits_wide) - 1
    b = a >> 1
    t0 = time.perf_counter()
    for _ in range(iterations):
        c = a & b
        c = a & ~b
        c = a | b
        c.bit_count()
    return (time.perf_counter() - t0) / iterations * 1e9   # ns per iteration


def microbenchmark():
    bits = sys.int_info.bits_per_digit
    print("raw integer op cost by mask width (ns per 4 ops)")
    print(f"{'width':>7}{'digits':>8}{'ns':>10}")
    prev = None
    for w in range(24, 40, 2):
        digits = (w + bits - 1) // bits
        ns = min(raw_int_ops(w) for _ in range(3))
        jump = ""
        if prev and ns > prev * 1.25:
            jump = "  <-- jump"
        print(f"{w:>7}{digits:>8}{ns:>10.1f}{jump}")
        prev = ns
    print()


def main():
    bits = sys.int_info.bits_per_digit
    print(f"CPython bits_per_digit = {bits}")
    print(f"so a mask needs 2 digits from n = {bits + 1}\n")
    microbenchmark()
    print(f"{'n':>4}{'digits':>8}", end="")
    for name, _, _ in PAIRS:
        print(f"{name + ' speedup':>20}", end="")
    print()

    for n in range(26, 37):
        digits = (n + bits - 1) // bits
        print(f"{n:>4}{digits:>8}", end="")
        for _, set_alg, bit_alg in PAIRS:
            ratios = []
            for seed in (1, 2, 3):
                G = nx.gnp_random_graph(n, 0.3, seed=seed)
                t_set = median_time(set_alg, G)
                t_bit = median_time(bit_alg, G)
                ratios.append(t_set / t_bit)
            print(f"{statistics.median(ratios):>20.2f}", end="")
        print(flush=True)

    print("\nA cliff between n=30 and n=31 supports the digit-size explanation.")
    print("A smooth decline means something else is going on.")


if __name__ == "__main__":
    main()
