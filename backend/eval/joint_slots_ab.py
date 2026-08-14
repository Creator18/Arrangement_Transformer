"""A/B the joint slot reconciliation on its own gate metric.

The gate, from overlay_containment_test: the two voices' best per-bar
alignment differed by more than 40 ms in 60.2% of bars.  If that does not fall
substantially the fix did not work, whatever the pooled numbers say — pooled
numbers were exactly what hid the problem, since the errors are inconsistent
rather than offset and therefore cancel.

Decodes ONCE and reconciles the same note list both ways, so the comparison
cannot be contaminated by NMF or decode variation.
"""
from __future__ import annotations

import argparse, json, pathlib, sys, warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "backend/eval"))
warnings.filterwarnings("ignore")
import scipy.signal, scipy.signal.windows                       # noqa: E402
if not hasattr(scipy.signal, "hann"):
    scipy.signal.hann = scipy.signal.windows.hann

from overlay_containment_test import (                           # noqa: E402
    piano_onset_map, song_masks, contain)

TEMP = ROOT / "backend/temp"


def per_bar_diff(notes, ons, downs, shifts, low, n_pitch, n_fr, fps):
    per = {}
    for role in ("melody", "bass"):
        sub = [n for n in notes if n.get("role") == role]
        P = piano_onset_map(sub, low, n_pitch, n_fr, fps)
        vals = []
        for a, b in zip(downs[:-1], downs[1:]):
            fa, fb = int(a * fps), int(b * fps)
            if fb - fa < 8 or not P[:, fa:fb].any():
                vals.append(np.nan)
                continue
            sc = [contain(P[:, fa:fb], ons[:, fa:fb], int(s)) for s in shifts]
            vals.append(np.nan if np.all(np.isnan(sc))
                        else shifts[int(np.nanargmax(sc))] / fps * 1000)
        per[role] = np.array(vals, dtype=float)
    d = per["melody"] - per["bass"]
    return d[~np.isnan(d)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    args = ap.parse_args()

    sess = TEMP / args.session
    from app.services.piano_projection import _activations, FPS, LOW
    from app.services import melody_decode as MD

    meta = json.loads((sess / "components.json").read_text())
    hop, msr = float(meta.get("hop_length", 512)), float(meta.get("sr", 22050))
    beats = np.asarray(meta.get("beat_frames", []), float) * hop / msr
    downs = np.asarray(meta.get("downbeat_frames", []), float) * hop / msr
    bpb = int(meta.get("beats_per_bar", 4))

    H = _activations(sess / "components/harmonic.wav", MD.BASS_LO, MD.VOCAL_HI)
    n_pitch, n_fr = H.shape

    # decode once with reconciliation OFF, then apply it explicitly
    MD.JOINT_SLOTS = False
    base = MD.decode_two_voices(H, LOW, beats, downs, bpb)
    fixed = MD.reconcile_slots(base, H, LOW, beats)

    _, ons = song_masks(H, 60, 60)
    step = max(1, int(0.01 * FPS))
    shifts = np.arange(-int(0.16 * FPS), int(0.16 * FPS) + 1, step)

    print(f"session {sess.name}   {len(base)} notes\n")
    moved = sum(1 for a, b in zip(base, fixed) if a["onset"] != b["onset"])
    print(f"{'':<10}{'|diff|>40ms':>13}{'median':>9}{'IQR':>16}"
          f"{'onset contain':>15}")
    for label, ns in (("before", base), ("after", fixed)):
        d = per_bar_diff(ns, ons, downs, shifts, LOW, n_pitch, n_fr, FPS)
        P = piano_onset_map(ns, LOW, n_pitch, n_fr, FPS)
        c = contain(P, ons, 0) * 100
        print(f"{label:<10}{100*np.mean(np.abs(d)>40):>12.1f}%"
              f"{np.median(d):>+8.0f}"
              f"{np.percentile(d,25):>+9.0f}..{np.percentile(d,75):<+6.0f}"
              f"{c:>13.1f}%")
    print(f"\nnotes whose onset moved: {moved} of {len(base)} "
          f"({100*moved/max(len(base),1):.1f}%)")
    print("The gate is the first column.  Onset containment is reported so a")
    print("drop there cannot be traded away for a better alignment number.")


if __name__ == "__main__":
    main()
