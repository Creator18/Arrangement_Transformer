"""Why is the v2 arranger less confident than v1?  Trace it through every stage.

For each checkpoint, on held-out Music2MIDI validation songs (target: the human
arrangement), separating two things a single threshold conflates:

  RANKING     average precision of the onset probabilities against the
              arrangement's onsets (+-2 frames).  Threshold-free: does the
              model put the arrangement's notes above the rest?
  LEVEL       how high its probabilities are: the 99.9th percentile of onset
              probability, the mean probability at true arrangement onsets,
              and the threshold that maximises F1.

A model that ranks as well as v1 but sits lower is miscalibrated (a training
artefact); one that ranks worse genuinely knows less.  Also the share of the
arrangement's onsets that get ANY peak above 0.3 per register, since the
bass register vanished from v2's output on Complication.

    python eval/confidence_trace.py --m2m /scratch/.../m2m --ckpt name=path ... [--audio x.wav]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LEVELS = ["beginner", "intermediate", "advanced"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2m", required=True)
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--audio", default="")
    ap.add_argument("--cache", required=True,
                    help="roll cache: --stage gpu writes it (inference only), --stage cpu scores it")
    ap.add_argument("--stage", choices=["gpu", "cpu"], required=True)
    args = ap.parse_args()
    import torch
    from scipy.ndimage import maximum_filter1d
    from sklearn.metrics import average_precision_score
    from app.services import spectral_transcription as ST
    from app.ml.spectral import frontend as S
    from app.ml.transcriber.decode import decode_notes

    m2m = pathlib.Path(args.m2m)
    sp = json.loads((m2m / "m2m_splits.json").read_text())
    songs = sp["validation"][: args.n]
    print(f"{len(songs)} validation songs; target = human arrangement onsets\n")
    print(f"{'checkpoint':<12}{'AP':>7}{'p99.9':>8}{'p@true':>8}{'best thr':>9}{'best F1':>8}"
          f"{'  hit<C3':>8}{'  C3-C5':>8}{'  >C5':>7}")
    cache = pathlib.Path(args.cache); cache.mkdir(parents=True, exist_ok=True)

    def rolls(name, path, wav, cond, conditioned):
        f = cache / f"{name}__{pathlib.Path(wav).stem}.npz"
        if args.stage == "cpu":
            z = np.load(f); return z["on"].astype(np.float32), z["fr"].astype(np.float32), z["vl"].astype(np.float32)
        ST.CKPT = pathlib.Path(path)
        on, fr, vl = ST.rolls(pathlib.Path(wav), cond=cond if conditioned else None)
        np.savez(f, on=on.astype(np.float16), fr=fr.astype(np.float16), vl=vl.astype(np.float16))
        return None

    for spec in args.ckpt:
        name, path = spec.split("=", 1)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        conditioned = bool(ck["arch"].get("n_cond"))
        if args.stage == "gpu":                         # inference only: the GPU never waits on scoring
            for sid in songs:
                rolls(name, path, m2m / "audio" / f"{sid}.wav", LEVELS.index(sp["difficulty"][sid]), conditioned)
            if args.audio:
                rolls(name, path, args.audio, 2, conditioned)
            print(f"  cached {name}", flush=True)
            continue
        ys, ps, hits = [], [], {0: [], 1: [], 2: []}
        p999, ptrue = [], []
        f1_at = {t: [] for t in np.arange(0.1, 0.95, 0.05).round(2)}
        for sid in songs:
            on, fr, vl = rolls(name, path, m2m / "audio" / f"{sid}.wav", 0, conditioned)
            T = on.shape[1]
            tgt = np.zeros_like(on, dtype=bool)
            for s, e, p, v in np.load(m2m / "midi_numpy" / f"{sid}.npy"):
                k, f = int(p) - 21, int(round(s * S.FPS))
                if 0 <= k < 88 and 0 <= f < T:
                    tgt[k, f] = True
            tol = maximum_filter1d(tgt.astype(np.uint8), 5, axis=1) > 0          # +-2 frames
            peak = maximum_filter1d(on, 5, axis=1)                                 # best prob nearby
            rng = np.random.default_rng(0)
            neg = np.flatnonzero(~tol.ravel()); pos = np.flatnonzero(tgt.ravel())
            pick = np.concatenate([pos, rng.choice(neg, min(len(neg), 20 * len(pos)), replace=False)])
            ys.append(np.isin(pick, pos)); ps.append(np.where(np.isin(pick, pos), peak.ravel()[pick], on.ravel()[pick]))
            p999.append(np.percentile(on, 99.9)); ptrue.append(peak[tgt].mean())
            for reg, (lo, hi) in enumerate(((0, 27), (27, 51), (51, 88))):
                hits[reg].append((peak[lo:hi][tgt[lo:hi]] > 0.3).mean() if tgt[lo:hi].any() else np.nan)
            ref = np.load(m2m / "midi_numpy" / f"{sid}.npy")
            from mir_eval.transcription import precision_recall_f1_overlap as prf
            ri = np.c_[ref[:, 0], np.maximum(ref[:, 1], ref[:, 0] + 1e-3)]; rp = 440 * 2 ** ((ref[:, 2] - 69) / 12)
            for t in f1_at:
                ns = decode_notes(on, fr, vl, t, 0.3, min_frames=1, min_note_s=0.0, keep_restruck=True)
                if not ns:
                    f1_at[t].append(0.0); continue
                e = np.array([[n["onset"], n["offset"], n["pitch"]] for n in ns])
                f1_at[t].append(prf(ri, rp, np.c_[e[:, 0], np.maximum(e[:, 1], e[:, 0] + 1e-3)],
                                    440 * 2 ** ((e[:, 2] - 69) / 12), onset_tolerance=0.05, offset_ratio=None)[2])
        ap_ = average_precision_score(np.concatenate(ys), np.concatenate(ps))
        bt = max(f1_at, key=lambda t: np.mean(f1_at[t]))
        print(f"{name:<12}{ap_:>7.3f}{np.mean(p999):>8.3f}{np.mean(ptrue):>8.3f}{bt:>9.2f}"
              f"{np.mean(f1_at[bt]):>8.3f}" + "".join(f"{np.nanmean(hits[r]):>8.2f}" for r in range(3)), flush=True)
        if args.audio:
            on, fr, vl = rolls(name, path, args.audio, 2, conditioned)
            dur = on.shape[1] / S.FPS
            dens = {t: len(decode_notes(on, fr, vl, t, 0.3, min_frames=1, min_note_s=0.0, keep_restruck=True)) / dur
                    for t in (0.6, bt)}
            print(f"{'':<12}  on --audio: p99.9 {np.percentile(on, 99.9):.3f}, notes/s at 0.6 {dens[0.6]:.1f}, "
                  f"at its best thr {bt} {dens[bt]:.1f}", flush=True)
    print("\nAP: threshold-free ranking of arrangement onsets.  p@true: mean best-nearby onset probability "
          "at true arrangement onsets.  hit: share of arrangement onsets with a peak > 0.3, by register.")


if __name__ == "__main__":
    main()
