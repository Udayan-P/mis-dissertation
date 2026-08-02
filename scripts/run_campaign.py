"""Run a benchmark campaign and write results to CSV.

Examples:
    python scripts/run_campaign.py --quick
    python scripts/run_campaign.py --out results/full.csv --max-n 34
    python scripts/run_campaign.py --dimacs
"""

import argparse
import json
import time
from pathlib import Path

from mis import benchmark
from mis.benchmark import Instance, build_sweep, run_instance, write_csv
from mis import graphs

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--quick", action="store_true",
                    help="small sweep for checking the pipeline works")
    ap.add_argument("--max-n", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--dimacs", action="store_true",
                    help="run on DIMACS instances instead of generated ones")
    ap.add_argument("--naive-max-n", type=int, default=20,
                    help="above this, skip the 2^n baseline")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    out = Path(args.out) if args.out else RESULTS / (
        f"{'quick' if args.quick else 'campaign'}_"
        f"{time.strftime('%Y%m%d_%H%M')}.csv")

    if args.dimacs:
        loaded = graphs.load_dimacs_instances(max_nodes=args.max_n)
        instances = [Instance(name, "dimacs", G) for name, G in loaded.items()]
        if not instances:
            print("no DIMACS instances found in instances/ ; skipping")
            return
    elif args.quick:
        instances = build_sweep(ns=[10, 14, 18], ps=[0.3, 0.5],
                                seeds=[1, 2], families=("er", "ba"))
    else:
        ns = [n for n in range(10, args.max_n + 1, 2)]
        instances = build_sweep(ns=ns, ps=[0.1, 0.3, 0.5, 0.7],
                                seeds=list(range(1, args.seeds + 1)),
                                families=("er", "ba", "ws"))

    # the naive algorithm is hopeless past about 20 vertices; excluding it
    # from bigger instances keeps the campaign finite
    algorithms = list(benchmark.ALGORITHMS)

    print(f"{len(instances)} instances -> {out}")
    rows, skip_slow = [], set()
    t0 = time.perf_counter()
    for i, inst in enumerate(sorted(instances, key=lambda x: x.n), 1):
        algs = [a for a in algorithms
                if not (a == "naive" and inst.n > args.naive_max_n)]
        new = run_instance(inst, algs, repeats=args.repeats,
                           time_limit=args.time_limit, skip_slow=skip_slow)
        rows.extend(new)
        best = min((r for r in new if r["algorithm"] != "naive"),
                   key=lambda r: r["median_seconds"], default=None)
        print(f"[{i}/{len(instances)}] {inst.name:<22} n={inst.n:<3} "
              f"sets={new[0]['mis_count']:<6} "
              f"fastest={best['algorithm'] if best else '-':<14} "
              f"{best['median_seconds']*1000:.2f}ms" if best else "")
        write_csv(rows, out)          # write as we go, so a crash keeps data

    meta = out.with_suffix(".meta.json")
    meta.write_text(json.dumps({
        "environment": benchmark.environment(),
        "args": vars(args),
        "instances": len(instances),
        "rows": len(rows),
        "wall_clock_seconds": round(time.perf_counter() - t0, 1),
    }, indent=2))
    print(f"\n{len(rows)} rows written to {out}")
    print(f"environment recorded in {meta}")


if __name__ == "__main__":
    main()
