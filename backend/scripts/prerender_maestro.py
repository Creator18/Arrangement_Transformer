"""
Pre-render every MAESTRO MIDI to a packed-bool .npy piano roll on disk so the
training loop can mmap-slice in O(milliseconds) instead of re-parsing the
MIDI through pretty_midi every step (~200 ms per file).

The cache files live next to their source MIDIs:
    foo.midi  →  foo.roll.<frame_hz>.npy

Total disk impact: ~600 MB for the full MAESTRO MIDI-only split at 100 fps
(bool dtype = 1 byte per cell, no compression so mmap stays cheap).

Idempotent — re-running skips pieces whose cache already exists.  Pass
`--force` to re-render everything.

Usage:
    cd backend
    python -m scripts.prerender_maestro            # render any missing
    python -m scripts.prerender_maestro --force    # re-render all
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.ml.config import PATHS, FRAME_RATE_HZ
from app.ml.data.maestro import (
    list_maestro, load_piano_roll, cached_roll_path,
)


def main() -> int:
    p = argparse.ArgumentParser(description="Pre-render MAESTRO MIDIs to .npy piano rolls")
    p.add_argument("--root", type=Path, default=PATHS.maestro_root,
                   help=f"MAESTRO root (default: {PATHS.maestro_root})")
    p.add_argument("--frame-hz", type=int, default=FRAME_RATE_HZ,
                   help=f"Frame rate (default: {FRAME_RATE_HZ} fps)")
    p.add_argument("--force", action="store_true",
                   help="Re-render even if cache exists")
    args = p.parse_args()

    root = args.root
    if not (root / "maestro-v3.0.0.csv").exists():
        print(f"[prerender] manifest not found at {root}.  "
              f"Run scripts/download_maestro.py first.")
        return 2

    # split=None loads train + validation + test together — we want all
    # of them cached so the dataset can be reconfigured to any split later.
    entries = list_maestro(root, split=None)
    print(f"[prerender] {len(entries)} MAESTRO performances")

    rendered = skipped = failed = 0
    bytes_written = 0
    t0 = time.time()

    for i, entry in enumerate(entries, 1):
        midi_path = entry.midi_path(root)
        cache_path = cached_roll_path(midi_path, args.frame_hz)

        if cache_path.exists() and not args.force:
            skipped += 1
            continue

        try:
            # allow_cache=False so we always rebuild from the MIDI, even if a
            # stale cache file is present.
            roll = load_piano_roll(midi_path, args.frame_hz, allow_cache=False)
            packed = roll.astype(np.bool_)
            np.save(str(cache_path), packed)
            bytes_written += cache_path.stat().st_size
            rendered += 1
        except Exception as e:
            print(f"  [{i}/{len(entries)}] FAILED {entry.title[:60]}: {e}")
            failed += 1
            continue

        if i % 50 == 0 or i == len(entries):
            elapsed = time.time() - t0
            mb_done = bytes_written / 1e6
            print(
                f"  [{i:>4}/{len(entries)}] "
                f"rendered={rendered:>4}  skipped={skipped:>4}  "
                f"failed={failed:>2}  "
                f"{mb_done:.1f} MB written  ({elapsed:.0f}s elapsed)"
            )

    elapsed = time.time() - t0
    print(
        f"[prerender] done.  rendered={rendered}, skipped={skipped}, failed={failed} "
        f"({elapsed:.0f}s, {bytes_written/1e6:.1f} MB written)"
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
