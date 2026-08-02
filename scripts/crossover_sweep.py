"""Larger sweep aimed at two questions the first campaign left open:

  1. Tomita's pivot rule searches ~3.9x fewer nodes than basic BK but was
     slower in wall-clock than simple pivoting at n <= 24. Does it overtake?
  2. Bitsets gave only 1.2-1.3x at n <= 24. Does the gap widen with n?

Only the six BK variants run (no naive baseline), and only at the densities
where the algorithms are still tractable. Time limit per instance keeps the
whole thing finite; anything that blows it is dropped from larger n.

Usage:
    python scripts/crossover_sweep.py                    # n=24..36
    python scripts/crossover_sweep.py --max-n 40 --seeds 3
"""

import argparse
import json
import time
from pathlib import Path

from mis import benchmark
from mis.benchmark import build_sweep, run_instance, write_csv

RESULTS = Path(__file__).resolve().parents[1] / "results"

VARIANTS = ["bk_basic", "bk_pivot", "bk_tomita",
            "bk_basic_bit", "bk_pivot_bit", "bk_tomita_bit"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-n", type=int, default=24)
    ap.add_argument("--max-n", type=int, default=36)
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    out = Path(args.out) if args.out else RESULTS / (
        f"crossover_{time.strftime('%Y%m%d_%H%M')}.csv")

    ns = list(range(args.min_n, args.max_n + 1, args.step))
    seeds = list(range(1, args.seeds + 1))
    # p=0.1 is dropped: sparse inputs mean dense complements and the output
    # itself explodes, which measures output size rather than search quality
    instances = build_sweep(ns=ns, ps=[0.3, 0.5, 0.7], seeds=seeds,
                            families=("er", "ba", "ws"))

    print(f"{len(instances)} instances, n={ns[0]}..{ns[-1]} -> {out}")
    rows, skip_slow = [], set()
    t0 = time.perf_counter()
    for i, inst in enumerate(sorted(instances, key=lambda x: x.n), 1):
        new = run_instance(inst, VARIANTS, repeats=args.repeats,
                           time_limit=args.time_limit, skip_slow=skip_slow)
        if not new:
            continue
        rows.extend(new)
        by_time = sorted(new, key=lambda r: r["median_seconds"])
        piv = next((r for r in new if r["algorithm"] == "bk_pivot"), None)
        tom = next((r for r in new if r["algorithm"] == "bk_tomita"), None)
        ratio = ""
        if piv and tom and tom["median_seconds"]:
            ratio = f"pivot/tomita={piv['median_seconds']/tom['median_seconds']:.2f}"
        print(f"[{i}/{len(instances)}] {inst.name:<20} n={inst.n:<3} "
              f"sets={new[0]['mis_count']:<7} "
              f"fastest={by_time[0]['algorithm']:<15} {ratio}")
        write_csv(rows, out)

    meta = out.with_suffix(".meta.json")
    meta.write_text(json.dumps({
        "environment": benchmark.environment(),
        "args": vars(args),
        "rows": len(rows),
        "wall_clock_seconds": round(time.perf_counter() - t0, 1),
    }, indent=2))
    print(f"\n{len(rows)} rows -> {out}")
    print(f"then: python scripts/crossover_analysis.py {out}")


if __name__ == "__main__":
    main()
