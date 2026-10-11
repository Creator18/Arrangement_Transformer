"""Render one fixed window per piano, so two machines can be compared.

Run on the laptop and on Explorer, then compare the outputs.  Rendering is
meant to be deterministic, so the two should match closely.  Two ways it can
quietly not:

  - the room processing skips itself on any import failure (it prints
    "[sampler] tone skipped" / "reverb skipped" and carries on), so a missing
    dependency gives drier audio rather than an error;
  - Linux file names are case-sensitive and Windows' are not, so a piano whose
    .sfz refers to a sample with different capitalisation plays on Windows
    and renders those notes as SILENCE on Linux.

    python eval/render_parity.py --out <dir>
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import pretty_midi
    from app.ml.spectral import render as R

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    row = R.pieces("test")[0]
    pm = pretty_midi.PrettyMIDI(str(R.MAESTRO / row["midi_filename"]))
    lat = R.latencies()
    for piano in sorted(R.pianos()):
        y, on, fr = R.render_window(pm, 20.0, 20.0, piano)
        np.savez(out / f"{piano}.npz", y=y, on=on, fr=fr)
        print(f"{piano:<11} rms {np.sqrt((y ** 2).mean()):.4f}  peak {np.abs(y).max():.3f}  "
              f"onsets {int(on.sum())}  latency {lat[piano] * 1000:.1f} ms", flush=True)


if __name__ == "__main__":
    main()
