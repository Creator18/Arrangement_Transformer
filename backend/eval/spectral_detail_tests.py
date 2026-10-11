"""Why does the transcription lose fine detail?  Each suspect, against ground truth.

Every earlier score was computed against references decoded from the label
rolls by app.ml.transcriber.decode.decode_notes — the SAME decoder the model's
output goes through.  That decoder drops notes shorter than MIN_NOTE_S (80 ms)
and, when a key is re-struck while sounding, cuts the earlier note to end
RELEASE_S before the next, which can push it under 80 ms and drop it too.  So
grace notes, trills and fast repeats were removed from the GROUND TRUTH as well
as the output, and a model missing all of them still scored 0.851.

Here the reference is built straight from the labels with no filtering: one
note per labelled onset.  Scoring is onset-and-pitch (50 ms), the same rule as
before, so the only thing that changes is what counts as the truth.

  T1 HIDDEN    how many true notes the decoder's filters remove from a reference
  T2 LENGTH    model output with the shipped filters vs with them relaxed
  T3 DYNAMICS  recall by how hard the note was played, at several thresholds —
               does the 0.93 threshold mostly cost the soft notes?
  T4 REPEATS   recall for notes re-struck within 150 ms of the same key

    python eval/spectral_detail_tests.py --ckpt <best.pt> --data <dir> --clips 600
"""
from __future__ import annotations

import argparse
import collections
import multiprocessing as mp
import pathlib
import random
import sys
import warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

FPS = 62.5
THRS = [0.5, 0.7, 0.85, 0.93]
BANDS = [("soft  (vel < 50)", 0, 50), ("medium (50-79)", 50, 80), ("loud  (vel >= 80)", 80, 128)]


def true_notes(on, vel):
    """One reference note per labelled onset: (pitch, onset_s, velocity, gap_s).

    gap_s is the time since the previous onset on the same key (inf if none).
    """
    out = []
    for k in range(on.shape[0]):
        last = -1e9
        for t in np.flatnonzero(on[k]):
            s = t / FPS
            out.append((k + 21, s, int(vel[k, t]), s - last))
            last = s
    return out


def match(ref, est, tol=0.05):
    """Greedy onset+pitch matching; returns the set of matched ref indices and n matched."""
    by_p = collections.defaultdict(list)
    for j, n in enumerate(est):
        by_p[int(n["pitch"])].append(float(n["onset"]))
    for p in by_p:
        by_p[p].sort()
    used = collections.defaultdict(set)
    hit = set()
    for i, (p, s, _, _) in sorted(enumerate(ref), key=lambda x: x[1][1]):
        cands = by_p.get(p, [])
        best, bd = None, tol + 1e-9
        for j, e in enumerate(cands):
            d = abs(e - s)
            if d <= bd and j not in used[p]:
                best, bd = j, d
            if e > s + tol:
                break
        if best is not None:
            used[p].add(best)
            hit.add(i)
    return hit


def _work(job):
    from app.ml.transcriber import decode as D
    i, po, pf, pv, on, fr, vel = job
    po, pf, pv = (a.astype(np.float32) for a in (po, pf, pv))
    ref = true_notes(on, vel)
    filt_ref = D.decode_notes(on.astype(np.float32), fr.astype(np.float32),
                              np.zeros(on.shape, np.float32), 0.5, 0.5)
    res = {"n_true": len(ref), "n_filtered_ref": len(filt_ref)}
    variants = {}
    for thr in THRS:
        variants[("shipped", thr)] = D.decode_notes(po, pf, pv, thr, 0.5)
    m0, f0 = D.MIN_NOTE_S, D.decode_notes.__defaults__
    try:
        D.MIN_NOTE_S = 0.0
        D.decode_notes.__defaults__ = (None, 0.5, 0.5, 1)    # min_frames 1
        for thr in THRS:
            variants[("relaxed", thr)] = D.decode_notes(po, pf, pv, thr, 0.5)
    finally:
        D.MIN_NOTE_S, D.decode_notes.__defaults__ = m0, f0
    for key, est in variants.items():
        hit = match(ref, est)
        res[key] = {"n_est": len(est), "tp": len(hit),
                    "band": [sum(1 for j in hit if lo <= ref[j][2] < hi) for _, lo, hi in BANDS],
                    "rep": sum(1 for j in hit if ref[j][3] < 0.15)}
    res["band_total"] = [sum(1 for r in ref if lo <= r[2] < hi) for _, lo, hi in BANDS]
    res["rep_total"] = sum(1 for r in ref if r[3] < 0.15)
    return i, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--clips", type=int, default=600)
    ap.add_argument("--procs", type=int, default=11)
    args = ap.parse_args()

    import torch
    from app.ml.spectral import frontend as S
    from app.ml.spectral.data import RenderedClips, clips
    from app.ml.spectral.model import from_checkpoint

    dev = torch.device("cuda")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    net = from_checkpoint(ck).to(dev).eval()
    fe = S.SpectralFrontend().to(dev).eval()
    fe.mu.copy_(ck["fe_mu"]); fe.sd.copy_(ck["fe_sd"])
    amp = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16

    te = clips(args.data, "test")
    random.Random(0).shuffle(te)
    te = sorted(te[: args.clips])
    ds = RenderedClips(te, crop_s=None, random_crop=False)
    jobs = []
    with torch.no_grad():
        for i in range(len(ds)):
            y, on, fr, vel, hv = ds[i]
            if not float(hv):
                continue
            with torch.autocast("cuda", dtype=amp):
                a, b, c = net(fe(y[None].to(dev)))
            jobs.append((i, *(torch.sigmoid(t.float())[0].T.cpu().numpy().astype(np.float16)
                              for t in (a, b, c)),
                         on.numpy().astype(np.uint8), fr.numpy().astype(np.uint8),
                         (vel.numpy() * 127).round().astype(np.uint8)))
    print(f"{len(jobs)} test clips with velocity labels\n", flush=True)

    tot = collections.Counter()
    agg = collections.defaultdict(lambda: collections.Counter())
    bands = collections.defaultdict(lambda: np.zeros(len(BANDS)))
    reps = collections.Counter()
    band_total, rep_total = np.zeros(len(BANDS)), 0
    with mp.Pool(args.procs) as pool:
        for _, r in pool.imap_unordered(_work, jobs, chunksize=4):
            tot["true"] += r["n_true"]; tot["filtered"] += r["n_filtered_ref"]
            band_total += r["band_total"]; rep_total += r["rep_total"]
            for k, v in r.items():
                if isinstance(k, tuple):
                    agg[k]["est"] += v["n_est"]; agg[k]["tp"] += v["tp"]
                    bands[k] += v["band"]; reps[k] += v["rep"]

    T = tot["true"]
    print("T1 HIDDEN — what the decoder's filters remove from a REFERENCE")
    print(f"   true notes (one per labelled onset)      {T}")
    print(f"   after decode_notes' filters               {tot['filtered']}  "
          f"({(T - tot['filtered']) / T * 100:.1f}% of true notes removed from the ground truth)")
    print(f"   notes re-struck within 150 ms             {rep_total}  ({rep_total / T * 100:.1f}%)\n")

    def prf(k):
        tp, est = agg[k]["tp"], agg[k]["est"]
        p, r = tp / max(1, est), tp / max(1, T)
        return p, r, 2 * p * r / max(1e-9, p + r)

    print("T2 LENGTH FILTER — model output vs UNFILTERED truth (onset + pitch, 50 ms)")
    print(f"   {'decoder':<10}{'thr':>6}{'F1':>8}{'P':>8}{'R':>8}{'notes out':>11}")
    for kind in ("shipped", "relaxed"):
        for thr in THRS:
            p, r, f = prf((kind, thr))
            print(f"   {kind:<10}{thr:>6.2f}{f:>8.3f}{p:>8.3f}{r:>8.3f}{agg[(kind, thr)]['est']:>11}")
    print("   (shipped = 80 ms minimum length, 2-frame minimum; relaxed = neither)\n")

    print("T3 DYNAMICS — recall by how hard the note was played (relaxed decoder)")
    print(f"   {'band':<20}{'notes':>8}" + "".join(f"{'thr ' + str(t):>11}" for t in THRS))
    for bi, (name, _, _) in enumerate(BANDS):
        row = "".join(f"{bands[('relaxed', t)][bi] / max(1, band_total[bi]):>11.3f}" for t in THRS)
        print(f"   {name:<20}{int(band_total[bi]):>8}{row}")
    print()
    print("T4 REPEATS — recall for notes re-struck within 150 ms, vs all notes")
    print(f"   {'decoder':<10}{'thr':>6}{'repeats':>10}{'all notes':>11}")
    for kind in ("shipped", "relaxed"):
        for thr in (0.5, 0.93):
            print(f"   {kind:<10}{thr:>6.2f}{reps[(kind, thr)] / max(1, rep_total):>10.3f}"
                  f"{agg[(kind, thr)]['tp'] / T:>11.3f}")


if __name__ == "__main__":
    main()
