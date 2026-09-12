"""Re-sweep onset_thr.  A threshold tuned for one model does not carry to another.

onset_thr is the precision/recall dial: the model is trained with a heavy
positive weight on onsets (they are 0.28% of the target, so without it the model
would learn to predict silence), and the threshold is what undoes that bias at
inference.  It shipped at 0.90, swept on the 2.87M model whose precision topped
out at 0.36 — it had to be pushed that far because the model fired far too much.

The 19.38M model is calibrated differently: precision 0.484 against 0.361 at the
same training-time threshold.  A better-calibrated model needs LESS correction,
so carrying 0.90 across would throw away recall the new model no longer has to
pay for.  Constants tuned against a model belong to that model.

Runs inference ONCE per track and decodes at every threshold, which is what
makes sweeping cheap enough to actually redo.

    venv/Scripts/python.exe eval/onset_thr_sweep.py --tracks 8
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

THRS = [0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.90, 0.94, 0.98]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", type=int, default=8)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--frame-thr", type=float, default=0.5)
    args = ap.parse_args()

    from app.ml.transcriber import config as C
    from app.ml.transcriber.data import audio_path, official_splits, track_rolls
    from app.ml.transcriber.decode import decode_notes, note_scores
    from app.services import ml_transcription as ML
    import soundfile as sf

    ckpt = pathlib.Path(args.ckpt) if args.ckpt else None
    _, va, how = official_splits()
    if "official" not in how:
        raise SystemExit("not the official splits")
    tracks = va[: args.tracks]
    print(f"{len(tracks)} held-out Slakh tracks, "
          f"ckpt {ckpt.name if ckpt else 'transcriber.pt'}\n")

    rows = {t: [] for t in THRS}
    n_ref_tot = 0
    for t in tracks:
        wav = audio_path(t, "mix")
        on, fr, vl = ML.rolls(wav, ckpt)          # inference ONCE
        info = sf.info(str(wav))
        nf = int(info.frames * C.SR / info.samplerate / C.HOP) + 1
        gon, gfr, _ = track_rolls(t, nf)
        ref = decode_notes(gon.astype(np.float32), gfr.astype(np.float32),
                           np.zeros_like(gon, dtype=np.float32), 0.5, 0.5)
        n_ref_tot += len(ref)
        for thr in THRS:                           # ...decode many times
            est = decode_notes(on, fr, vl, thr, args.frame_thr)
            rows[thr].append(note_scores(ref, est))
        print(f"  {t.name}: {len(ref)} ref notes", flush=True)

    print(f"\nmean reference notes/track: {n_ref_tot / len(tracks):.0f}\n")
    print(f"{'thr':>6}{'notes':>8}{'F1':>8}{'P':>8}{'R':>8}{'+off':>8}")
    print("-" * 46)
    best = (None, -1.0)
    for thr in THRS:
        a = rows[thr]
        f1 = float(np.mean([r["conp_f1"] for r in a]))
        print(f"{thr:>6.2f}{np.mean([r['n_est'] for r in a]):>8.0f}{f1:>8.3f}"
              f"{np.mean([r['conp_p'] for r in a]):>8.3f}"
              f"{np.mean([r['conp_r'] for r in a]):>8.3f}"
              f"{np.mean([r['conpoff_f1'] for r in a]):>8.3f}")
        if f1 > best[1]:
            best = (thr, f1)
    print(f"\nbest: thr {best[0]} at F1 {best[1]:.3f}")
    if best[0] in (THRS[0], THRS[-1]):
        print("WARNING: the optimum is at the edge of the swept range, so this "
              "does not bracket a peak -- extend the range before trusting it.")


if __name__ == "__main__":
    main()
