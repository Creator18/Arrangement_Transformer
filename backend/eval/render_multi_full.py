"""Render ALL of MAESTRO through the multi-piano pool, in parallel.

1,276 pieces, 199 hours, ~34 GB.  Each piece is cut into consecutive 20 s
windows rendered with pre-roll (render.render_window), and the whole piece
uses one piano drawn deterministically from (split, index).

Resumable: a window already on disk is skipped, so the job can be stopped,
restarted, or split across machines (an Explorer job array can take the
remainder) and the result is identical to one uninterrupted run.

Workers: rendering is CPU-only (sfizz).  The limit here is RAM, not cores —
each worker holds sampled pianos in memory (~0.7 GB with all five loaded),
and the laptop has ~5 GB free.  Jobs are sorted by piano so a worker mostly
keeps one piano loaded, and the synth cache is dropped whenever its piano
changes.

    venv/Scripts/python.exe eval/render_multi_full.py --workers 6
Progress: <render dir>/progress.log  (written line by line, never buffered)
"""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SECONDS = 20.0
_current_piano = None


def _work(job):
    """Render every missing window of one piece; returns (piece key, windows written)."""
    global _current_piano
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np
    import pretty_midi
    from app.ml.spectral import render as R
    from app.services import sfz_sampler as Z

    split, i, midi, dur, piano = job
    out = R.CACHE_MULTI / split
    n_win = max(1, math.ceil(dur / SECONDS))
    todo = [w for w in range(n_win) if not (out / f"{i:04d}_{w:03d}.npz").exists()]
    if not todo:
        return f"{split}/{i}", 0, n_win
    if piano != _current_piano:          # keep one piano resident per worker
        Z._SYNTH_CACHE.clear()
        _current_piano = piano
    pm = pretty_midi.PrettyMIDI(str(R.MAESTRO / midi))
    for w in todo:
        y, on, fr = R.render_window(pm, w * SECONDS, SECONDS, piano)
        tmp = out / f"{i:04d}_{w:03d}.tmp.npz"
        np.savez_compressed(tmp, y=(np.clip(y, -1, 1) * 32767).astype(np.int16),
                            on=on, fr=fr, piano=piano)
        tmp.replace(out / f"{i:04d}_{w:03d}.npz")   # atomic: no half-written clips
    return f"{split}/{i}", len(todo), n_win


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--splits", default="test,validation,train")
    ap.add_argument("--limit", type=int, default=0, help="pieces per split, for a trial")
    ap.add_argument("--shard", default="",
                    help="i/N: render only every N-th piece starting at i, for a "
                         "Slurm job array (--shard $SLURM_ARRAY_TASK_ID/N)")
    args = ap.parse_args()

    import warnings
    warnings.filterwarnings("ignore")
    from app.ml.spectral import render as R

    R.latencies()                         # measured once, before workers start
    jobs = []
    for split in args.splits.split(","):
        (R.CACHE_MULTI / split).mkdir(parents=True, exist_ok=True)
        rows = R.pieces(split)
        if args.limit:
            rows = rows[: args.limit]
        for i, row in enumerate(rows):
            jobs.append((split, i, row["midi_filename"], float(row["duration"]),
                         R.piano_for(split, i)))
    log_name = "progress.log"
    if args.shard:
        # Strided over a stable (split, index) order, so long and short pieces
        # spread evenly across shards rather than one task drawing all the
        # long ones.  Every shard sees the same order, so together they cover
        # each piece exactly once.
        si, sn = (int(v) for v in args.shard.split("/"))
        jobs.sort(key=lambda j: (j[0], j[1]))
        jobs = jobs[si::sn]
        log_name = f"progress_{si:03d}.log"
    jobs.sort(key=lambda j: (j[4], j[0], j[1]))      # group by piano
    total_win = sum(max(1, math.ceil(j[3] / SECONDS)) for j in jobs)

    log = open(R.CACHE_MULTI / log_name, "a", buffering=1, encoding="utf-8")

    def say(msg):
        line = f"{time.strftime('%H:%M:%S')}  {msg}"
        print(line, flush=True)
        log.write(line + "\n")

    say(f"start: {len(jobs)} pieces, {total_win} windows, {args.workers} workers -> "
        f"{R.CACHE_MULTI}")
    t0 = time.time()
    done_win = written = 0
    with mp.Pool(args.workers) as pool:
        for k, (key, n_new, n_win) in enumerate(pool.imap_unordered(_work, jobs), 1):
            done_win += n_win
            written += n_new
            el = time.time() - t0
            rate = written / el if el > 0 else 0
            left = (total_win - done_win)
            eta = left / rate / 3600 if rate > 0 else float("nan")
            if k % 10 == 0 or k == len(jobs):
                say(f"{k}/{len(jobs)} pieces  {done_win}/{total_win} windows  "
                    f"{rate * 60:.0f} windows/min  ETA {eta:.1f} h")
    say(f"done in {(time.time() - t0) / 3600:.2f} h; {written} windows written")


if __name__ == "__main__":
    main()
