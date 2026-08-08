"""
progress.py — how far along is an eval run, and is its output safe to read?

Written after getting this wrong twice.  Both times I counted
run_summary.json files by modification time inside a rolling window, and
both times the PREVIOUS run had finished inside that same window, so its
leftovers were counted as the current run's progress.  The second time
that produced a comparison over a MIXTURE of two configurations which I
then reported as a regression.

Two rules, learned the hard way:

  1. The run's own log is authoritative.  run_pipeline.py prints
     "[ n/50] <piece>" per completion; the highest n is the truth.  File
     timestamps describe the filesystem, not the run.

  2. compare.py --all reads every run directory, so it is only meaningful
     when ALL of them belong to the current run.  Anything less is a
     blend of configurations wearing the costume of a result.

Usage:
    python backend/eval/progress.py <logfile>
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional, Tuple

_LINE = re.compile(r"\[\s*(\d+)\s*/\s*(\d+)\s*\]")


def read_progress(log_path: Path) -> Tuple[Optional[int], Optional[int]]:
    """
    Highest (completed, total) seen in the log.  Note stdout is buffered
    through `tee`, so not every line reaches disk immediately — the highest
    index is a LOWER BOUND on real progress, never an over-count.
    """
    if not log_path.exists():
        return None, None
    best = total = None
    raw = log_path.read_bytes().decode("utf-8", errors="replace")
    for m in _LINE.finditer(raw.replace("\r", "\n")):
        n, t = int(m.group(1)), int(m.group(2))
        if best is None or n > best:
            best, total = n, t
    return best, total


def safe_to_compare(log_path: Path) -> bool:
    """True only when every piece has been re-run under the current code."""
    done, total = read_progress(log_path)
    return bool(done and total and done >= total)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    log = Path(sys.argv[1])
    done, total = read_progress(log)
    if done is None:
        print(f"no progress lines found in {log}")
        raise SystemExit(1)
    print(f"{done}/{total} pieces complete (lower bound; stdout is buffered)")
    if done >= total:
        print("SAFE to run compare.py --all")
    else:
        print(f"NOT safe to compare — {total - done} directories still hold "
              f"the previous run's output, and compare.py --all reads all of "
              f"them")


if __name__ == "__main__":
    main()
