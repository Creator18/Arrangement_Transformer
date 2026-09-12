"""Which rule recovers a human's melody line?

`enforce_melody_on_top` is rule one of piano arranging: the tune must be the
highest sounding voice.  It needs to know WHICH notes are the tune, and it asks
for `role == "melody"`.  The transcriber emits `role: "all"` for everything, so
on the song path the function currently returns its input untouched — the rule
is not wrong, it is simply never firing.

Filling that gap needs a melody detector, and before building one it is worth
knowing what the cheap answers are actually worth.  POP909 can say, because it
ships a human's own split: a MELODY track (the sung line), a BRIDGE track
(secondary melodic material) and a PIANO track (accompaniment).  Flatten all
three into one undifferentiated note set — which is what the transcriber hands
us — and ask each rule to pick the melody back out.

This measures the rule in ISOLATION, on perfect input.  Our real input is
note F1 0.630, so every number here is an optimistic ceiling: a rule that
cannot find the melody in a human's own clean notes will certainly not find it
in ours.  That is the point — a ceiling is cheap to measure and rules out
candidates for free.

Run:
    venv/Scripts/python.exe eval/pop909_melody_role.py [--songs 909]
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

POP909 = ROOT.parent / "data/POP909"
FPS = 50.0                  # frame grid for "is this note the top voice"

# A note counts as melody if it is the highest sounding voice for at least this
# fraction of its length.  Swept below rather than assumed.
TOP_FRAC = 0.5


def load(song: pathlib.Path):
    """(pitch, start, end, is_melody) for every note in the human arrangement.

    BRIDGE counts as melody.  In POP909 it carries melodic material the singer
    is not on — instrumental answers and fills — and an arranger keeps those on
    top for the same reason they keep the sung line there.  Scoring it as
    accompaniment would punish a rule for being right.
    """
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(str(song / f"{song.name}.mid"))
    out = []
    for inst in pm.instruments:
        mel = inst.name.strip().upper() in ("MELODY", "BRIDGE")
        for n in inst.notes:
            out.append((int(n.pitch), float(n.start), float(n.end), mel))
    out.sort(key=lambda r: r[1])
    return out


def top_fraction(notes):
    """For each note, the fraction of its frames where it is the highest pitch."""
    if not notes:
        return np.zeros(0)
    end = max(n[2] for n in notes)
    nf = int(end * FPS) + 2
    top = np.zeros(nf, dtype=np.int16)
    spans = []
    for p, s, e, _ in notes:
        a, b = int(s * FPS), max(int(e * FPS), int(s * FPS) + 1)
        b = min(b, nf)
        spans.append((a, b, p))
        np.maximum.at(top, np.arange(a, b), p)
    frac = np.empty(len(notes))
    for i, (a, b, p) in enumerate(spans):
        frac[i] = float(np.mean(top[a:b] == p)) if b > a else 0.0
    return frac


def prf(pred: np.ndarray, truth: np.ndarray):
    tp = float(np.sum(pred & truth))
    if tp == 0:
        return 0.0, 0.0, 0.0
    p = tp / float(np.sum(pred))
    r = tp / float(np.sum(truth))
    return p, r, 2 * p * r / (p + r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--songs", type=int, default=909)
    args = ap.parse_args()

    songs = sorted(d for d in POP909.iterdir() if (d / f"{d.name}.mid").exists())
    songs = songs[:args.songs]

    rows: dict[str, list] = {}
    base_rates = []
    rng = np.random.default_rng(0)

    for si, song in enumerate(songs):
        try:
            notes = load(song)
        except Exception:                                   # noqa: BLE001
            continue
        if not notes:
            continue
        truth = np.array([n[3] for n in notes])
        pitch = np.array([n[0] for n in notes])
        if truth.sum() == 0 or truth.sum() == len(truth):
            continue
        base = float(truth.mean())
        base_rates.append(base)
        frac = top_fraction(notes)

        cands = {
            # The classic: melody is whatever is on top.  This is the rule
            # enforce_melody_on_top already ASSUMES holds after it runs, so if
            # it scores well the invariant is self-consistent.
            "skyline (top >=50% of its length)": frac >= 0.5,
            "skyline (top >=80%)":               frac >= 0.8,
            "skyline (top for any frame)":       frac > 0.0,
            # Register band, the assumption baked into the roll's display
            # colours (melody = 48..60) and into anything that splits by pitch.
            "register 48..60":        (pitch >= 48) & (pitch <= 60),
            "register >= 60":          pitch >= 60,
            # Floors.  Without these none of the above means anything.
            "ALL notes are melody":   np.ones(len(truth), bool),
            "random @ skyline rate":  rng.random(len(truth)) < float((frac >= 0.5).mean()),
        }
        for name, pred in cands.items():
            rows.setdefault(name, []).append(prf(pred, truth))
        if (si + 1) % 200 == 0:
            print(f"  ...{si + 1}/{len(songs)}", flush=True)

    n = len(base_rates)
    print(f"\nPOP909 melody-role recovery — {n} songs, clean human notes")
    print(f"melody is {np.mean(base_rates) * 100:.1f}% of notes "
          f"(so 'all notes' scores that as precision)\n")
    print(f"{'rule':<36}{'P':>7}{'R':>7}{'F1':>7}")
    print("-" * 57)
    for name, vals in rows.items():
        a = np.array(vals)
        print(f"{name:<36}{a[:, 0].mean():>7.3f}{a[:, 1].mean():>7.3f}"
              f"{a[:, 2].mean():>7.3f}")


if __name__ == "__main__":
    main()
