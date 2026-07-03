# Empirical Study of Exact Exponential-Time Algorithms for Enumerating Maximal Independent Sets

MSc Dissertation — Udayan Purandare (rbfn02), supervised by Dr Igor Razgon.
MSc Advanced Computer Science, Durham University, 2025–26.

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
