"""
Distil POP909's MELODY tracks into the priors a decoder needs.

melody_cleanup.py filters detected notes using four constants I set by hand.
melody_decode.py replaces it by DECODING a line through the activation matrix,
and a decode needs a model of what melodies do rather than thresholds for what
to delete.  This measures that model from 909 human-notated vocal lines and
writes it next to the service as _melody_priors.npz.

Four distributions, and what each one replaces
----------------------------------------------
  interval    P(next pitch - current pitch) over melodic moves.  This is what
              suppresses vibrato smear: a jump to a neighbouring semitone and
              straight back is cheap under a generic smoothness prior and
              expensive under a real one, because human melodies move by tone
              far more than by semitone (36.0% vs 8.2%).
  dwell       How long a sung note lasts, as a per-frame probability of
              staying on the same pitch.  Replaces MIN_DUR — duration stops
              being a threshold that deletes notes and becomes a cost that
              shapes them.
  metre       P(a note starts at subdivision k of the bar).  Replaces the
              snap-to-grid step: onsets are PULLED toward strong positions by
              evidence rather than moved there afterwards.
  silence     How much of a vocal line is rest.  Without this the decoder sings
              continuously, because there is always some activation somewhere.

Deliberately NOT measured here: absolute pitch distribution.  It would encode
POP909's key and register habits, and our input has its own key that the
pipeline already detects.  The priors have to be transposition-invariant to be
worth anything, so everything above is expressed in intervals and time.
"""
from __future__ import annotations

import argparse, pathlib, sys, warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
warnings.filterwarnings("ignore")

POP909 = ROOT / "data/POP909"
OUT = ROOT / "backend/app/services/_melody_priors.npz"

FPS = 100.0          # must match melody_decode.FPS
MAX_IV = 12          # intervals beyond an octave share one bucket
SLOTS = 16           # metrical resolution: sixteenths of a 4/4 bar


def bar_edges(folder: pathlib.Path):
    f = folder / "beat_midi.txt"
    if not f.exists():
        return None
    rows = np.loadtxt(f)
    if rows.ndim != 2 or rows.shape[1] < 3:
        return None
    downs = rows[rows[:, 2] > 0.5, 0]
    return downs if len(downs) >= 3 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=909)
    args = ap.parse_args()

    import pretty_midi

    folders = sorted([d for d in POP909.iterdir() if d.is_dir()])[: args.n]

    iv_counts = np.zeros(2 * MAX_IV + 3)      # [-12..+12] plus two overflow buckets
    metre = np.zeros(SLOTS)
    dwell_frames, n_notes = 0.0, 0
    sung_frames, total_frames = 0.0, 0.0
    used = 0

    for folder in folders:
        mid = folder / f"{folder.name}.mid"
        if not mid.exists():
            continue
        try:
            pm = pretty_midi.PrettyMIDI(str(mid))
        except Exception:                                   # noqa: BLE001
            continue
        mel = next((i for i in pm.instruments
                    if (i.name or "").strip().upper() == "MELODY"), None)
        if mel is None or len(mel.notes) < 8:
            continue
        used += 1
        notes = sorted(mel.notes, key=lambda n: n.start)

        span = pm.get_end_time() or 1.0
        total_frames += span * FPS
        for n in notes:
            d = max(0.0, n.end - n.start)
            dwell_frames += d * FPS
            sung_frames += d * FPS
            n_notes += 1

        for a, b in zip(notes[:-1], notes[1:]):
            iv = b.pitch - a.pitch
            if iv < -MAX_IV:
                iv_counts[0] += 1
            elif iv > MAX_IV:
                iv_counts[-1] += 1
            else:
                iv_counts[iv + MAX_IV + 1] += 1

        edges = bar_edges(folder)
        if edges is not None:
            for n in notes:
                k = int(np.searchsorted(edges, n.start)) - 1
                if k < 0 or k + 1 >= len(edges):
                    continue
                b0, b1 = edges[k], edges[k + 1]
                if b1 <= b0:
                    continue
                metre[int(np.clip((n.start - b0) / (b1 - b0) * SLOTS, 0, SLOTS - 1))] += 1

    # ── normalise, with a floor so nothing is impossible ────────────────────
    # A zero probability is a veto, and a decoder that can never make a move it
    # has not seen in 909 songs will refuse correct but unusual melodies.
    iv = iv_counts / iv_counts.sum()
    iv = 0.98 * iv + 0.02 / len(iv)

    metre = metre / metre.sum() if metre.sum() else np.ones(SLOTS) / SLOTS
    metre = 0.9 * metre + 0.1 / SLOTS

    mean_dwell = dwell_frames / max(n_notes, 1)          # frames per note
    p_stay = float(np.clip(1.0 - 1.0 / max(mean_dwell, 2.0), 0.5, 0.999))
    p_sing = float(np.clip(sung_frames / max(total_frames, 1.0), 0.05, 0.95))

    np.savez(OUT, interval=iv, metre=metre, p_stay=p_stay, p_sing=p_sing,
             max_iv=MAX_IV, slots=SLOTS, fps=FPS, songs=used, notes=n_notes)

    print(f"POP909: {used} songs, {n_notes} melody notes -> {OUT.name}\n")
    print(f"mean note length : {mean_dwell/FPS:.3f} s  ({mean_dwell:.1f} frames)")
    print(f"p_stay per frame : {p_stay:.4f}")
    print(f"voiced fraction  : {p_sing:.3f}")
    print("\ninterval prior (semitones -> probability)")
    for k in range(-5, 6):
        print(f"  {k:+3d}  {iv[k + MAX_IV + 1]:.4f}"
              + ("   <- unison" if k == 0 else
                 "   <- semitone" if abs(k) == 1 else
                 "   <- tone" if abs(k) == 2 else ""))
    print(f"  |iv|>12 combined  {iv[0] + iv[-1]:.4f}")
    print("\nmetrical prior (sixteenth slot -> probability of a note starting)")
    for i in range(SLOTS):
        bar = "#" * int(round(metre[i] / metre.max() * 34))
        tag = "beat" if i % 4 == 0 else ("8th" if i % 2 == 0 else "")
        print(f"  {i:>2} {metre[i]:.4f} {bar:<35}{tag}")


if __name__ == "__main__":
    main()
