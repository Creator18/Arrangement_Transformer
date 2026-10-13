"""Score cached arrangements of the Music2MIDI test songs, every model side by side (CPU).

Reads <notes>/<model>__<id>.npy (written by eval/group_fit.py --stage notes or
eval/audio_arranger_eval.py) and reports, per difficulty and overall, the same
measures as eval/s2p_evaluate.py: note F1 strict (50 ms), tolerant (150 ms),
pitch-class tolerant, precision / recall, and the scorecard metrics with the
share inside the human range for the song's level.  Reference = the original
human arrangement (as for every earlier model).

    python eval/score_notes.py --notes /scratch/.../group_fit --m2m /scratch/.../m2m
"""
from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
LEVELS = ["beginner", "intermediate", "advanced"]
CARD = ["harmonic", "melody", "bass", "groove", "mud", "notes_per_s", "poly_mean", "jump_speed_p90"]


def job(a):
    from s2p_evaluate import f1s
    from scorecard import score_arrangement
    model, sid, est, ref, song, lv, ranges = a
    f, p, r, ft, fp = f1s(ref, est)
    card = score_arrangement(est, song) if len(est) > 50 else {k: np.nan for k in CARD}
    inside = [ranges[k][lv]["p10"] <= card[k] <= ranges[k][lv]["p90"] for k in CARD if np.isfinite(card.get(k, np.nan))]
    return model, dict(level=lv, f1=f, p=p, r=r, tol=ft, pc=fp, inside=float(np.mean(inside)) if inside else 0.0,
                       **{k: card.get(k, np.nan) for k in CARD})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes", required=True)
    ap.add_argument("--m2m", required=True)
    ap.add_argument("--procs", type=int, default=4)
    args = ap.parse_args()
    nd, m2m = pathlib.Path(args.notes), pathlib.Path(args.m2m)
    sp = json.loads((m2m / "m2m_splits.json").read_text())
    ranges = json.loads((ROOT / "app" / "ml" / "scorecard_ranges.json").read_text())
    models = sorted({f.name.split("__")[0] for f in nd.glob("*__*.npy")})
    jobs = []
    for sid in sp["test"]:
        ref = np.load(m2m / "midi_numpy" / f"{sid}.npy")
        sc = m2m / "scorecard" / f"song_{sid}.npz"
        if not sc.exists():
            continue
        song = dict(np.load(sc))
        for m in models:
            f = nd / f"{m}__{sid}.npy"
            if f.exists():
                est = np.load(f)
                jobs.append((m, sid, est.astype(np.float64), ref, song, sp["difficulty"][sid], ranges))
    res = collections.defaultdict(list)
    with mp.Pool(args.procs) as pool:
        for m, row in pool.imap_unordered(job, jobs, chunksize=4):
            res[m].append(row)
    mean = lambda rows, k: float(np.nanmean([r[k] for r in rows])) if rows else float("nan")
    for lv in LEVELS + ["all"]:
        print(f"-- {lv} --")
        print(f"   {'model':<8}{'songs':>6}{'strict':>8}{'P':>7}{'R':>7}{'tol':>7}{'pc':>7}{'in-range':>10}"
              + "".join(f"{k[:9]:>10}" for k in CARD))
        for m in models:
            rows = [r for r in res[m] if lv == "all" or r["level"] == lv]
            print(f"   {m:<8}{len(rows):>6}{mean(rows, 'f1'):>8.3f}{mean(rows, 'p'):>7.3f}{mean(rows, 'r'):>7.3f}"
                  f"{mean(rows, 'tol'):>7.3f}{mean(rows, 'pc'):>7.3f}{mean(rows, 'inside'):>10.2f}"
                  + "".join(f"{mean(rows, k):>10.2f}" for k in CARD))


if __name__ == "__main__":
    main()
