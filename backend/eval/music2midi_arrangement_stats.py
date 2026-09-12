"""What does a human piano arrangement of a pop song actually look like?

Every reduction constant in transformation.py is hand-picked, and the one that
matters most says so out loud:

    MAX_SOUNDING = 4     # the human arrangement never exceeded about this

That is an empirical claim, and until now it rested on very little.  The
Music2MIDI dataset is 2,974 human piano covers of pop songs, scraped from
MuseScore and beat-aligned to their source recordings by the dataset's authors.
It can answer the claim directly, and it needs no audio to do it — the
arrangements are the measurement.

Note what the source implies.  These covers come from MuseScore, which is the
same corpus a transcription product would reach for if it wanted to retrieve an
arrangement rather than derive one.  Used as TARGETS with attribution it is a
training set; used as OUTPUT it is the shortcut.  Same data, and the difference
is entirely in which direction it flows.

Run:
    venv/Scripts/python.exe eval/music2midi_arrangement_stats.py [--n 900]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA = ROOT.parent / "data/music2midi/music2midi-dataset"
FPS = 25.0                  # frame grid for counting simultaneous notes

# What we currently do, for comparison.
MAX_SOUNDING = 4


def sounding_profile(notes: np.ndarray):
    """Per-frame count of simultaneously sounding notes, and the hand span."""
    end = float(notes[:, 1].max())
    nf = int(end * FPS) + 2
    count = np.zeros(nf, dtype=np.int16)
    lo = np.full(nf, 127, dtype=np.int16)
    hi = np.zeros(nf, dtype=np.int16)
    for on, off, p, _ in notes:
        a, b = int(on * FPS), max(int(off * FPS), int(on * FPS) + 1)
        b = min(b, nf)
        if b <= a:
            continue
        idx = np.arange(a, b)
        count[idx] += 1
        np.minimum.at(lo, idx, int(p))
        np.maximum.at(hi, idx, int(p))
    live = count > 0
    span = (hi - lo)[live]
    return count[live], span


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=900)
    args = ap.parse_args()

    files = sorted((DATA / "midi_numpy").glob("*.npy"))[:args.n]
    if not files:
        print(f"no data at {DATA}")
        return

    counts, spans, dens, pitches, nsongs = [], [], [], [], 0
    for f in files:
        try:
            a = np.load(f)
        except Exception:                                   # noqa: BLE001
            continue
        if len(a) < 50:
            continue
        c, s = sounding_profile(a)
        if not len(c):
            continue
        counts.append(c)
        spans.append(s)
        dens.append(len(a) / float(a[:, 1].max()))
        pitches.append((a[:, 2].min(), a[:, 2].max()))
        nsongs += 1

    c = np.concatenate(counts)
    s = np.concatenate(spans)
    p = np.array(pitches)

    print(f"\nMusic2MIDI — {nsongs} human piano arrangements of pop songs")
    print(f"{len(c) / FPS / 3600:.1f} h of sounding time\n")

    print("SIMULTANEOUS NOTES  (what cap_polyphony limits)")
    for q in (50, 75, 90, 95, 99, 99.9):
        print(f"  p{q:<6}{np.percentile(c, q):>6.1f}")
    print(f"  mean  {c.mean():>6.2f}     max {c.max():>4d}")
    over = float(np.mean(c > MAX_SOUNDING)) * 100
    print(f"\n  frames above our MAX_SOUNDING={MAX_SOUNDING}: {over:.1f}%")
    print(f"  => the constant's comment says humans 'never exceeded about this'")

    print("\nHAND SPAN, semitones lowest..highest sounding")
    for q in (50, 90, 99):
        print(f"  p{q:<6}{np.percentile(s, q):>6.1f}")
    print(f"  max   {s.max():>6.0f}")

    print("\nNOTE DENSITY, notes per second")
    d = np.array(dens)
    for q in (10, 50, 90):
        print(f"  p{q:<6}{np.percentile(d, q):>6.2f}")

    print("\nPITCH RANGE used")
    print(f"  lowest  median {np.median(p[:, 0]):.0f}   p1 {np.percentile(p[:, 0], 1):.0f}")
    print(f"  highest median {np.median(p[:, 1]):.0f}   p99 {np.percentile(p[:, 1], 99):.0f}")


if __name__ == "__main__":
    main()
