"""Write velocity labels beside every rendered clip, from the MIDI alone.

The full render stored onsets and frames but not velocity, and the model's
velocity head needs a target — dynamics are a large part of what makes a
transcription sound played rather than typed.  Recomputing labels takes MIDI
parsing only, no audio, so it is minutes rather than a 45-minute re-render.

For each clip <i>_<w>.npz it writes <i>_<w>.vel.npy (88 x T, uint8).  Every
file is CHECKED against the clip's stored onset roll: velocity must be nonzero
exactly where an onset is labelled.  A mismatch means the two label passes
disagree about timing, and the run stops rather than train on it.

    python eval/make_velocity_rolls.py [--shard i/N]
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys
import warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

SECONDS = 20.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="")
    args = ap.parse_args()
    import pretty_midi
    from app.ml.spectral import render as R

    jobs = sorted((s, i) for s in ("test", "validation", "train")
                  for i in range(len(R.pieces(s))))
    if args.shard:
        si, sn = (int(v) for v in args.shard.split("/"))
        jobs = jobs[si::sn]
    rows = {s: R.pieces(s) for s in ("test", "validation", "train")}

    written = checked = 0
    for split, i in jobs:
        row = rows[split][i]
        pm = None
        n_win = max(1, math.ceil(float(row["duration"]) / SECONDS))
        for w in range(n_win):
            clip = R.CACHE_MULTI / split / f"{i:04d}_{w:03d}.npz"
            out = clip.with_name(clip.stem + ".vel.npy")
            if out.exists():
                continue
            pm = pm or pretty_midi.PrettyMIDI(str(R.MAESTRO / row["midi_filename"]))
            z = np.load(clip)
            piano = str(z["piano"])
            vel = R.velocity_roll(pm, w * SECONDS, SECONDS, piano)
            if not np.array_equal(vel > 0, z["on"] > 0):
                d = int(((vel > 0) != (z["on"] > 0)).sum())
                raise SystemExit(f"{clip}: velocity and onset labels disagree at {d} cells")
            np.save(out, vel)
            written += 1
            checked += 1
    print(f"wrote {written} velocity rolls, every one consistent with its onset roll")


if __name__ == "__main__":
    main()
