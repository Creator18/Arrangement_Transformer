"""
How often does each chord quality actually occur, and how often is it in key?

infer_chord scores 12 roots x 5 qualities with NO prior at all, so a diminished
triad competes on equal terms with a major one.  It should not: a diminished
triad is (0,3,6), a symmetric stack of minor thirds narrow enough to partially
explain a huge range of pitch-class sets, so with a flat prior it wins ties it
has no business winning.  Measured on a real track, 31% of bars were assigned a
diminished chord and only 47% were fully diatonic.

POP909 ships human chord labels (chord_midi.txt, Harte notation) and a key
(key_audio.txt) for all 909 songs, which is exactly the missing prior.  Two
distributions come out, both transposition-invariant so they transfer to any
key our pipeline detects:

  quality   P(maj), P(min), P(dom7), P(min7), P(dim)
  degree    P(root is n semitones above the key root), 12 values.  This is
            what lets a genuine secondary dominant still win — it is not
            "diatonic or nothing", it is how often each chromatic degree really
            gets used.

Labels are mapped down to the five templates the arpeggiator can actually
voice.  That mapping is lossy on purpose: sus4, maj7 and maj6 all function as
"major" for a left-hand figure, and pretending we can voice them when
build_figure only stacks the template's tones would be a fiction.
"""
from __future__ import annotations

import argparse, collections, pathlib, sys, warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
warnings.filterwarnings("ignore")

POP909 = ROOT / "data/POP909"
OUT = ROOT / "backend/app/services/_chord_priors.npz"

QUALITIES = ["maj", "min", "dom7", "min7", "dim"]

_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def parse_root(tok: str):
    if not tok or tok[0] not in _PC:
        return None
    pc = _PC[tok[0]]
    for ch in tok[1:]:
        if ch == "#":
            pc += 1
        elif ch == "b":
            pc -= 1
        else:
            break
    return pc % 12


def map_quality(q: str):
    """Harte quality -> one of the five templates we can voice."""
    q = q.split("/")[0]                    # inversions are the same chord
    if q.startswith("min7"):
        return "min7"
    if q.startswith("hdim") or q.startswith("dim"):
        return "dim"
    if q.startswith("min"):
        return "min"
    if q.startswith("maj") or q.startswith("sus") or q in ("6", "9", ""):
        return "maj"
    if q.startswith("7") or q.startswith("9"):
        return "dom7"
    if q.startswith("aug"):
        return None
    return "maj" if q[:1].isdigit() is False else "dom7"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=909)
    args = ap.parse_args()

    qual = collections.Counter()
    degree = np.zeros(12)
    raw = collections.Counter()
    songs = 0
    total = 0

    for folder in sorted([d for d in POP909.iterdir() if d.is_dir()])[: args.n]:
        cf, kf = folder / "chord_midi.txt", folder / "key_audio.txt"
        if not cf.exists() or not kf.exists():
            continue
        # key: take the first (songs here are single-key in practice)
        key_pc = None
        for line in kf.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 3 and ":" in parts[2]:
                key_pc = parse_root(parts[2].split(":")[0])
                break
        if key_pc is None:
            continue
        songs += 1

        for line in cf.read_text().splitlines():
            parts = line.split()
            if len(parts) < 3 or parts[2] == "N" or ":" not in parts[2]:
                continue
            root_tok, q = parts[2].split(":", 1)
            r = parse_root(root_tok)
            if r is None:
                continue
            raw[q.split("/")[0]] += 1
            m = map_quality(q)
            if m is None:
                continue
            # Weight by duration: a chord held for a bar is stronger evidence
            # than a passing one, and infer_chord scores duration-weighted
            # chroma too, so the prior should be measured the same way.
            try:
                dur = max(0.05, float(parts[1]) - float(parts[0]))
            except ValueError:
                dur = 1.0
            qual[m] += dur
            degree[(r - key_pc) % 12] += dur
            total += 1

    q = np.array([qual[k] for k in QUALITIES], dtype=float)
    q = q / q.sum()
    q = 0.97 * q + 0.03 / len(q)            # floor: never make a quality impossible
    d = degree / degree.sum()
    d = 0.95 * d + 0.05 / 12

    np.savez(OUT, qualities=np.array(QUALITIES), quality_p=q, degree_p=d,
             songs=songs, chords=total)

    print(f"POP909: {songs} songs, {total} chord labels -> {OUT.name}\n")
    print("QUALITY prior (duration-weighted, mapped to our five templates)")
    for k, v in zip(QUALITIES, q):
        print(f"  {k:<6} {v:.4f}")
    print(f"\n  -> dim is {q[QUALITIES.index('dim')]:.4f}, i.e. "
          f"{q[QUALITIES.index('maj')]/q[QUALITIES.index('dim')]:.0f}x rarer than maj."
          f"  A flat prior treated them as equal.")
    print("\nDEGREE prior (semitones above the key root)")
    names = ["I", "bII", "II", "bIII", "III", "IV", "bV", "V", "bVI", "VI", "bVII", "VII"]
    for i in range(12):
        bar = "#" * int(round(d[i] / d.max() * 32))
        print(f"  {names[i]:<4} {d[i]:.4f} {bar}")
    diat = sum(d[i] for i in (0, 2, 4, 5, 7, 9, 11))
    print(f"\n  diatonic degrees carry {100*diat:.1f}% of chord time")
    print("\nraw Harte labels, most common (before mapping)")
    for k, v in raw.most_common(12):
        print(f"  {k or '(bare)':<10} {v}")


if __name__ == "__main__":
    main()
