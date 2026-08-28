# Empirical Study of Exact Exponential-Time Algorithms for Enumerating Maximal Independent Sets

MSc Dissertation — Udayan Purandare (rbfn02), supervised by Dr Igor Razgon.
MSc Advanced Computer Science, Durham University, 2025–26.

## The five pivot variants

The study is an ablation of the Bron-Kerbosch pivot rule. Five variants share a *single*
recursion and differ only in the function that picks the pivot, so any difference in node
count, pivot work or time comes from the pivot rule alone. They form a 2x2 of pivot *source*
against selection *cost*, with an un-pivoted floor outside it:

|                  | cheap selection (lowest-numbered vertex) | scanning selection (maximise \|P ∩ N(u)\|) |
|------------------|------------------------------------------|--------------------------------------------|
| source `P ∪ X`   | `bk_pivot`                               | `bk_tomita`                                |
| source `P`       | `bk_pivot_p`                             | `bk_ikgp`                                  |

- `bk_basic` — no pivoting at all; the floor the four above are measured against.
- `bk_tomita` is the rule of Tomita et al. (2006); `bk_ikgp` is Koch's rule, which Cazals and
  Karande call IK_GP and Abu-Khzam et al. implement as "Improved BK".
- Bitset counterparts `bk_basic_bit`, `bk_pivot_bit`, `bk_tomita_bit` hold vertex sets in
  Python integers rather than `set` objects. The P-only arms have no bitset counterpart.
- `naive` is a 2^n subset-enumeration baseline, used only for correctness checks at small n.

Implementations are in `src/mis/bron_kerbosch.py` and `src/mis/bitset.py`; the registry that
names all nine arms is `ALGORITHMS` in `src/mis/benchmark.py`.

## Reproducing a campaign

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
pytest                                    # correctness harness

# the headline campaign -> results/final_v3.csv (1,077 rows)
python scripts/crossover_sweep.py --min-n 24 --max-n 40 --step 2 --seeds 3 --repeats 5 \
    --time-limit 30 --labellings random --densities 0.3,0.5,0.7 \
    --out results/final_v3.csv

# the labelling sub-experiment -> results/labelling_v3.csv (1,200 rows)
python scripts/crossover_sweep.py --min-n 24 --max-n 32 --step 2 --seeds 2 --repeats 3 \
    --time-limit 30 --labellings random,native,degeneracy --densities 0.3,0.5,0.7 \
    --out results/labelling_v3.csv

# regenerate every numeric table quoted in the report
python scripts/make_results_tables.py    # writes results/RESULTS_TABLES.md
```

Each run writes a `.meta.json` beside its CSV recording the exact arguments, Python version,
platform and processor; the two committed `.meta.json` files are what the commands above were
reconstructed from. Instances are never stored, only regenerated from their recorded
`(family, n, p, seed)` tuple.

**Run one campaign at a time.** Two concurrent sweeps contaminate every timing in both.

## Which CSV backs which result

| Data | Backs |
|------|-------|
| `results/final_v3.csv` | Everything except the labelling sub-experiment: correctness and coverage, the headline `time(pivot)/time(tomita)` comparison and its by-family split, the tree-size / per-node-cost decomposition, the 2x2, the structural correlations, and the bitset speedups. Report §4.1-§4.6; `RESULTS_TABLES.md` sections S5.2, S5.3, S5.4, S5.6, S5.7. |
| `results/labelling_v3.csv` | The labelling sub-experiment only — the pivot/tomita ratio under random, native and degeneracy vertex numbering. Report §4.2; `RESULTS_TABLES.md` section S5.5. |

The structural correlation table (S5.6) is built from `final_v3.csv` **only**, and deliberately.
Pooling labellings collapses those correlations — clique overlap against node_gain falls from
+0.82 on random labelling alone to +0.44 pooled — so quoting them from the labelling campaign
would be quoting a weaker and genuinely different effect.

Six superseded pre-2026-08-11 campaign CSVs are excluded from this archive by `.gitignore`;
they predate three measurement fixes and their timings contradict the results above. Their
`.meta.json` records remain tracked, so the runs stay on the record.

## Layout

```
src/mis/          algorithms + graph utilities
tests/            correctness harness (pytest)
experiments/      benchmark runner, generators, results CSVs (added W2)
notebooks/        analysis notebooks (added W3)
```

## Setup (local)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest
```

## Algorithms (planned)

| ID | Algorithm | Status |
|----|-----------|--------|
| A0 | Naive 2^n subset enumeration | pending |
| A2 | Bron–Kerbosch, no pivot (1973) | pending |
| A3 | Bron–Kerbosch with pivot (1973) | pending |
| A4 | Tomita et al. pivot selection (2006) | pending |
| A5 | Eppstein–Löffler–Strash (2010) | only if on schedule at 20 Jul |

Correctness: all algorithms must emit identical collections of maximal independent
sets, cross-checked against hand-verified counts on small graphs (paths, cycles,
complete, complete bipartite, Petersen).

Code freeze: **Sunday 26 July 2026**.
