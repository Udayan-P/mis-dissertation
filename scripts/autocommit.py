"""Checkpoint-commit the repo whenever it actually changes.

Polls on an interval; commits only if the working tree differs from HEAD
and has been quiet for a debounce period (so it doesn't commit mid-edit).
If the test suite passes the commit is a normal one, otherwise it's marked
WIP, so history shows honestly which checkpoints were green.

Timestamps therefore track when work really happened. If nothing changes,
nothing is committed.

Usage:
    python scripts/autocommit.py                 # every 15 min, no push
    python scripts/autocommit.py --interval 600 --push
    python scripts/autocommit.py --once
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def git(*args, check=True):
    return subprocess.run(["git", *args], cwd=REPO, check=check,
                          capture_output=True, text=True).stdout.strip()


def dirty() -> bool:
    return bool(git("status", "--porcelain"))


def tree_hash() -> str:
    """Hash of the current working tree state, to detect quiet periods."""
    return git("status", "--porcelain") + git("diff", "--stat")


def tests_pass() -> bool:
    r = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=REPO,
                       capture_output=True, text=True)
    return r.returncode == 0


def summarise() -> str:
    """Short description of what changed, from the file paths."""
    files = [line[3:] for line in git("status", "--porcelain").splitlines()]
    areas = set()
    for f in files:
        if f.startswith("src/mis/"):
            areas.add(Path(f).stem)
        elif f.startswith("tests/"):
            areas.add("tests")
        elif f.startswith("scripts/"):
            areas.add("scripts")
        elif f.startswith(("results/", "figures/")):
            areas.add("results")
        else:
            areas.add(Path(f).parts[0])
    listed = ", ".join(sorted(areas)[:4])
    if len(areas) > 4:
        listed += f" (+{len(areas) - 4} more)"
    return listed or "working tree"


def checkpoint(push=False) -> bool:
    if not dirty():
        return False
    green = tests_pass()
    prefix = "checkpoint" if green else "WIP"
    msg = f"{prefix}: {summarise()}"
    if not green:
        msg += " [tests failing]"
    git("add", "-A")
    git("commit", "-m", msg)
    print(f"{time.strftime('%H:%M:%S')}  {msg}")
    if push:
        subprocess.run(["git", "push"], cwd=REPO, check=False)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=900,
                    help="seconds between checks (default 15 min)")
    ap.add_argument("--debounce", type=int, default=90,
                    help="require the tree to be unchanged this long first")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    if args.once:
        if not checkpoint(args.push):
            print("nothing to commit")
        return

    print(f"watching {REPO} every {args.interval}s; ctrl-C to stop")
    last_seen, last_time = tree_hash(), time.time()
    try:
        while True:
            time.sleep(min(args.interval, 30))
            now = tree_hash()
            if now != last_seen:
                last_seen, last_time = now, time.time()   # still editing
                continue
            if dirty() and time.time() - last_time >= args.debounce:
                checkpoint(args.push)
                last_seen, last_time = tree_hash(), time.time()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
