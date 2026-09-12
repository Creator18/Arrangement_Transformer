"""Underfitting or overfitting?  The two call for opposite fixes.

The training curve plateaus at note F1 ~0.50 from epoch 19 to 40 while the
training loss keeps falling.  That single fact is consistent with both:

    underfitting   the model has extracted everything its capacity allows;
                   the fix is a bigger model.
    overfitting    the model is memorising the training tracks;
                   the fix is more data or more regularisation, and a bigger
                   model makes it WORSE.

The discriminator is performance on TRAINING tracks.  If train and validation
score about the same, nothing is being memorised and capacity is the ceiling.
If train scores far higher, the gap IS the overfitting.

This project has been caught by exactly this once before: an "the model cannot
learn" reading where the real tell was frame F1 of 0.43 on training data, which
meant underpowered optimisation rather than an impossible task.  So it gets
measured rather than argued.

Run:
    venv/Scripts/python.exe eval/transcriber_fit_check.py [--tracks 6]
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


def score_track(track: pathlib.Path, onset_thr: float, frame_thr: float):
    """Note F1 for one Slakh track, whole track, same path as inference."""
    from app.ml.transcriber import config as C
    from app.ml.transcriber.data import audio_path, track_rolls
    from app.ml.transcriber.decode import decode_notes, note_scores
    from app.services import ml_transcription as ML
    import soundfile as sf

    est = ML.transcribe(audio_path(track, "mix"),
                        onset_thr=onset_thr, frame_thr=frame_thr)
    info = sf.info(str(audio_path(track, "mix")))
    nf = int(info.frames * C.SR / info.samplerate / C.HOP) + 1
    on, fr, _ = track_rolls(track, nf)
    # Reference notes from the ground-truth roll, decoded the same way the
    # estimate is, so the comparison cannot be biased by a different note
    # construction on either side.
    ref = decode_notes(on.astype(np.float32), fr.astype(np.float32),
                       np.zeros_like(on, dtype=np.float32), 0.5, 0.5)
    return note_scores(ref, est)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", type=int, default=6)
    ap.add_argument("--onset-thr", type=float, default=0.90)
    ap.add_argument("--frame-thr", type=float, default=0.5)
    args = ap.parse_args()

    from app.ml.transcriber.data import official_splits

    tr, va, how = official_splits()
    print(f"splits: {how}   train {len(tr)}  val {len(va)}")
    if "official" not in how:
        print("REFUSING: not the official splits — a fallback split makes this "
              "comparison meaningless.")
        return

    # Same count from each side, and the SAME tracks every run.
    sets = {"TRAIN (seen)": tr[:args.tracks], "VAL (held out)": va[:args.tracks]}
    out = {}
    for label, tracks in sets.items():
        rows = []
        for t in tracks:
            try:
                s = score_track(t, args.onset_thr, args.frame_thr)
            except Exception as e:                          # noqa: BLE001
                print(f"  skip {t.name}: {type(e).__name__}: {e}")
                continue
            rows.append(s)
            print(f"  {label:<16}{t.name}  F1 {s['conp_f1']:.3f} "
                  f"(P {s['conp_p']:.3f} R {s['conp_r']:.3f})  "
                  f"ref {s['n_ref']} est {s['n_est']}", flush=True)
        out[label] = rows

    print(f"\n{'set':<18}{'F1':>7}{'P':>7}{'R':>7}{'+off':>7}")
    print("-" * 46)
    means = {}
    for label, rows in out.items():
        if not rows:
            continue
        f = np.mean([r["conp_f1"] for r in rows])
        means[label] = f
        print(f"{label:<18}{f:>7.3f}"
              f"{np.mean([r['conp_p'] for r in rows]):>7.3f}"
              f"{np.mean([r['conp_r'] for r in rows]):>7.3f}"
              f"{np.mean([r['conpoff_f1'] for r in rows]):>7.3f}")

    if len(means) == 2:
        a, b = means["TRAIN (seen)"], means["VAL (held out)"]
        gap = a - b
        print(f"\ntrain - val = {gap:+.3f}")
        if gap < 0.05:
            print("=> UNDERFITTING.  The model does no better on tracks it was "
                  "trained on, so nothing is memorised and capacity is the "
                  "ceiling.  A bigger model is the indicated fix.")
        elif gap > 0.15:
            print("=> OVERFITTING.  The gap is the memorisation.  More data or "
                  "regularisation; a bigger model would make this worse.")
        else:
            print("=> MIXED.  Some memorisation but not the main story; "
                  "capacity and data are both plausible and need an A/B.")


if __name__ == "__main__":
    main()
