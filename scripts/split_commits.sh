#!/usr/bin/env bash
# Commit the current working tree as several logically scoped commits
# instead of one dump. Runs the tests before each commit so every commit in
# history is a state where the suite passed.
#
# Usage: bash scripts/split_commits.sh [--dry-run]

set -euo pipefail
cd "$(dirname "$0")/.."

DRY=${1:-}

commit_group() {
    local msg="$1"; shift
    local paths=("$@")
    local existing=()
    for p in "${paths[@]}"; do
        [ -e "$p" ] && existing+=("$p")
    done
    [ ${#existing[@]} -eq 0 ] && return 0

    git add -- "${existing[@]}"
    if git diff --cached --quiet; then
        echo "skip (nothing staged): $msg"
        return 0
    fi

    if [ "$DRY" = "--dry-run" ]; then
        echo "would commit: $msg"
        # only this group's paths, not whatever else was already staged
        git diff --cached --name-only -- "${existing[@]}" | sed 's/^/    /'
        git reset -q -- "${existing[@]}"
        return 0
    fi

    if ! pytest -q > /dev/null 2>&1; then
        echo "TESTS FAILING - refusing to commit: $msg" >&2
        git reset -q
        exit 1
    fi

    git commit -q -m "$msg"
    echo "committed: $msg"
}

commit_group "Bitset implementations of the BK variants" \
    src/mis/bitset.py

commit_group "Make pivot selection deterministic so trees are reproducible" \
    src/mis/bron_kerbosch.py

commit_group "Seeded graph generators and DIMACS reader" \
    src/mis/graphs.py instances

commit_group "Benchmark runner with timings, node counts and memory" \
    src/mis/benchmark.py scripts/run_campaign.py

commit_group "Analysis: growth-base fits, speedup tables, plots" \
    scripts/analyse.py

commit_group "Tests for bitset equivalence, DIMACS round-trip, seeding" \
    tests

commit_group "First campaign results and figures" \
    results figures

# note: notes/ lives in the parent Dissertation folder, outside this repo,
# so it isn't versioned here

# anything left over
if [ -n "$(git status --porcelain)" ] && [ "$DRY" != "--dry-run" ]; then
    echo
    echo "still uncommitted:"
    git status --short
fi
