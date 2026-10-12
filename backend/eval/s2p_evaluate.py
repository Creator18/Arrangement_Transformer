"""Song-to-piano v1: does the fine-tuned arranger beat "just transcribe" and September's?

On the 106 held-out Music2MIDI test songs (pinned split), per difficulty:

  piano       the app's piano transcriber, untouched — "transcribe everything"
  september   app/ml/transcriber/checkpoints/arranger.pt (mel CRNN fine-tuned
              on the same split, September 2026)
  s2p-v1      spectral arranger, asked for the song's own difficulty
  s2p-v1 adv  the same model asked for "advanced" (what the app would request)

Each system's onset threshold is tuned on the VALIDATION songs (strict F1),
then fixed for test.  Scores against the one human arrangement per song:

  strict      note F1, 50 ms onsets, exact pitch
  tolerant    note F1, 150 ms onsets (absorbs score-to-audio alignment error)
  pc          tolerant, pitch class only (voicing/octave choices not penalised)

and the scorecard (eval/scorecard.py): each metric's value and whether it lands
inside the human p10-p90 range for the song's level.

    python eval/s2p_evaluate.py --m2m /scratch/.../m2m --ckpt s2p=<best.pt> \\
        --piano <run2 best.pt> --september <arranger.pt> --out results.json
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
from scorecard import score_arrangement      # noqa: E402

LEVELS = ["beginner", "intermediate", "advanced"]
THRS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
CARD = ["harmonic", "melody", "bass", "groove", "mud", "notes_per_s", "poly_mean", "jump_speed_p90"]


def as_arrays(notes):
    if not len(notes):
        return np.zeros((0, 2)), np.zeros(0)
    iv = np.array([[n[0], max(n[1], n[0] + 1e-3)] for n in notes])
    return iv, 440 * 2 ** ((np.array([n[2] for n in notes], float) - 69) / 12)


def f1s(ref, est):
    from mir_eval.transcription import precision_recall_f1_overlap as prf
    if not len(est):
        return 0.0, 0.0, 0.0, 0.0, 0.0
    ri, rp = as_arrays(ref); ei, ep = as_arrays(est)
    p, r, f, _ = prf(ri, rp, ei, ep, onset_tolerance=0.05, offset_ratio=None)
    ft = prf(ri, rp, ei, ep, onset_tolerance=0.15, offset_ratio=None)[2]
    pc = lambda ns: [(a, b, 60 + int(c) % 12) for a, b, c, *_ in ns]
    ri2, rp2 = as_arrays(pc(ref)); ei2, ep2 = as_arrays(pc(est))
    fp = prf(ri2, rp2, ei2, ep2, onset_tolerance=0.15, offset_ratio=None)[2]
    return f, p, r, ft, fp


def _decode(kind, r, thr):
    from app.ml.transcriber.decode import decode_notes
    on, fr, vl = r
    if kind == "spec":
        ns = decode_notes(on, fr, vl, thr, 0.5, min_frames=1, min_note_s=0.0, keep_restruck=True)
    else:
        ns = decode_notes(on, fr, vl, thr, 0.5)
    return np.array([[n["onset"], n["offset"], n["pitch"], n["velocity"]] for n in ns])         if ns else np.zeros((0, 4))


def _val_job(a):
    """Worker: strict F1 at every threshold for one (song, system)."""
    from mir_eval.transcription import precision_recall_f1_overlap as prf
    name, kind, r, ref = a
    ri, rp = as_arrays(ref)
    out = []
    for t in THRS:
        est = _decode(kind, r, t)
        f = prf(ri, rp, *as_arrays(est), onset_tolerance=0.05, offset_ratio=None)[2] if len(est) else 0.0
        out.append((name, t, f))
    return out


def _test_job(a):
    """Worker: all similarity scores and the scorecard for one (song, system)."""
    name, kind, r, thr, ref, song, lv, tgt, ranges = a
    est = ref if kind == "human" else _decode(kind, r, thr)
    f, p, rr, ft, fp = (1, 1, 1, 1, 1) if kind == "human" else f1s(ref, est)
    card = score_arrangement(est, song) if len(est) > 50 else {k: np.nan for k in CARD}
    inside = [ranges[k][tgt]["p10"] <= card[k] <= ranges[k][tgt]["p90"] for k in CARD
              if np.isfinite(card.get(k, np.nan))]
    return name, dict(level=lv, f1=f, p=p, r=rr, tol=ft, pc=fp,
                      inside=float(np.mean(inside)) if inside else 0.0,
                      **{k: card.get(k, np.nan) for k in CARD})


def main():
    import multiprocessing as mp
    pool = mp.Pool(7)                       # before CUDA is touched: workers fork clean
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2m", required=True)
    ap.add_argument("--ckpt", required=True, action="append",
                    help="s2p model: path, or label=path; repeat to compare versions")
    ap.add_argument("--piano", required=True)
    ap.add_argument("--september", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    from app.services import spectral_transcription as ST
    from app.services import ml_transcription as MT
    # ml_transcription reloads its checkpoint on every call; load the September
    # arranger once
    _ml, _orig = {}, MT._load
    def _load_once(ckpt=None):
        if str(ckpt) not in _ml:
            _ml[str(ckpt)] = _orig(ckpt)
        return _ml[str(ckpt)]
    MT._load = _load_once

    m2m = pathlib.Path(args.m2m)
    npy = m2m / "midi_numpy" if (m2m / "midi_numpy").is_dir() else m2m / "music2midi-dataset" / "midi_numpy"
    sp = json.loads((m2m / "m2m_splits.json").read_text())
    ranges = json.loads((ROOT / "app" / "ml" / "scorecard_ranges.json").read_text())
    level = sp["difficulty"]

    def systems(sid):
        """name -> (on, fr, vl) probability rolls, (88, T)."""
        wav = m2m / "audio" / f"{sid}.wav"
        out = {}
        ST.CKPT = pathlib.Path(args.piano); out["piano"] = ("spec", ST.rolls(wav))
        for spec in args.ckpt:
            label, path = spec.split("=", 1) if "=" in spec else ("s2p-v1", spec)
            ST.CKPT = pathlib.Path(path)
            out[label] = ("spec", ST.rolls(wav, cond=LEVELS.index(level[sid])))
            out[f"{label} adv"] = ("spec", ST.rolls(wav, cond=2))
        if args.september:
            out["september"] = ("mel", MT.rolls(wav, ckpt=pathlib.Path(args.september)))
        return out

    # 1. thresholds on validation
    va = sp["validation"][: args.limit or None]
    acc = collections.defaultdict(list)
    pend = []
    for sid in va:
        ref = np.load(npy / f"{sid}.npy")
        for name, (kind, r) in systems(sid).items():
            pend.append(pool.apply_async(_val_job, ((name, kind, r, ref),)))
        print(f"  val {sid}", flush=True)
    for job in pend:
        for name, t, f in job.get():
            acc[(name, t)].append(f)
    names = sorted({k[0] for k in acc})
    best = {n: max(THRS, key=lambda t: np.mean(acc[(n, t)])) for n in names}
    print("\nthresholds tuned on validation (strict F1):")
    for n in names:
        print(f"   {n:<12} thr {best[n]:.2f}  F1 {np.mean(acc[(n, best[n])]):.3f}")

    # 2. test
    te = sp["test"][: args.limit or None]
    res = collections.defaultdict(list)
    pend = []
    for sid in te:
        ref = np.load(npy / f"{sid}.npy")
        song = dict(np.load(m2m / "scorecard" / f"song_{sid}.npz"))
        lv = level[sid]
        for name, (kind, r) in systems(sid).items():
            tgt = "advanced" if name.endswith(" adv") else lv
            pend.append(pool.apply_async(_test_job, ((name, kind, r, best[name], ref, song, lv, tgt, ranges),)))
        pend.append(pool.apply_async(_test_job, (("human (reference)", "human", None, 0, ref, song, lv, lv, ranges),)))
        print(f"  test {sid} {lv}", flush=True)
    for job in pend:
        name, row = job.get()
        res[name].append(row)

    m = lambda rows, k: float(np.nanmean([r[k] for r in rows])) if rows else float("nan")
    print(f"\nTEST: {len(te)} songs\n")
    for lv in LEVELS + ["all"]:
        print(f"-- {lv} --")
        print(f"   {'system':<18}{'strict':>8}{'P':>7}{'R':>7}{'tol':>7}{'pc':>7}{'in-range':>10}"
              + "".join(f"{k[:9]:>10}" for k in CARD))
        for n in list(res):
            rows = [r for r in res[n] if lv == "all" or r["level"] == lv]
            print(f"   {n:<18}{m(rows, 'f1'):>8.3f}{m(rows, 'p'):>7.3f}{m(rows, 'r'):>7.3f}"
                  f"{m(rows, 'tol'):>7.3f}{m(rows, 'pc'):>7.3f}{m(rows, 'inside'):>10.2f}"
                  + "".join(f"{m(rows, k):>10.2f}" for k in CARD))
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps({"thresholds": best, "results": res}, indent=1))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
