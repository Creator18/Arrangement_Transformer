"""Evaluate the trained spectral model properly: tuned, held out, per piano, vs baseline.

  1 THRESHOLD SWEEP on VALIDATION clips.  Training selected the checkpoint at
    an untuned onset threshold of 0.5, where precision (0.79) trailed recall
    (0.91) — the model fires extra notes.  The threshold is the
    precision/recall dial, and for the old transcriber tuning it was worth a
    large F1 gain.  Tuned on validation ONLY, so the test score stays honest.

  2 TEST SCORE on all 3,686 test clips — pieces never seen in training or in
    checkpoint selection — at the tuned threshold and at 0.5.

  3 PER-PIANO breakdown: works on all five, or only some?

  4 BASELINE: the piano path's existing transcriber (ByteDance high-resolution
    piano transcription, checkpoint note_F1=0.9677) on the SAME clips.  Its
    published 0.9677 is on real MAESTRO recordings, a different population;
    comparing against that number would repeat the wrong-population mistake.
    It is slow, so both models are compared on a fixed 600-clip subset
    balanced across pianos.

References are decoded from the label rolls exactly as during training, and
scored with mir_eval (50 ms onset tolerance), so every number here is on one
footing.

    python eval/spectral_evaluate.py --ckpt <best.pt> --data <render dir>
"""
from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import pathlib
import random
import sys
import time
import warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

ONSET_THRS = [0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99]
FRAME_THRS = [0.3, 0.5]


def _ref(on, fr):
    from app.ml.transcriber.decode import decode_notes
    return decode_notes(on.astype(np.float32), fr.astype(np.float32),
                        np.zeros(on.shape, np.float32), 0.5, 0.5)


def _score_clip(job):
    """(clip index, {(ot, ft): scores}) — run in a worker process."""
    from app.ml.transcriber.decode import decode_notes, note_scores
    i, po, pf, pv, on, fr, thrs = job
    ref = _ref(on, fr)
    out = {}
    for ot, ft in thrs:
        est = decode_notes(po.astype(np.float32), pf.astype(np.float32),
                           pv.astype(np.float32), ot, ft)
        out[(ot, ft)] = note_scores(ref, est)
    return i, out


def _score_notes(job):
    from app.ml.transcriber.decode import note_scores
    i, est, on, fr = job
    return i, note_scores(_ref(on, fr), est)


def _piano(path):
    z = np.load(path)
    return str(z["piano"]) if "piano" in z.files else "salamander"


def mean(rows, k):
    return float(np.mean([r[k] for r in rows])) if rows else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--val-clips", type=int, default=800)
    ap.add_argument("--baseline-clips", type=int, default=600)
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--out", default="")
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--limit-test", type=int, default=0, help="smoke runs only")
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
    print(f"checkpoint step {ck['step']} (validation F1 {ck['best']:.3f} at thr 0.5), "
          f"{torch.cuda.get_device_name(0)}", flush=True)

    def infer(paths):
        ds = RenderedClips(paths, crop_s=None, random_crop=False)
        res = []
        with torch.no_grad():
            for i in range(len(ds)):
                y, on, fr, vel, _ = ds[i]
                with torch.autocast("cuda", dtype=amp):
                    a, b, c = net(fe(y[None].to(dev)))
                res.append((torch.sigmoid(a.float())[0].T.cpu().numpy().astype(np.float16),
                            torch.sigmoid(b.float())[0].T.cpu().numpy().astype(np.float16),
                            torch.sigmoid(c.float())[0].T.cpu().numpy().astype(np.float16),
                            on.numpy().astype(np.uint8), fr.numpy().astype(np.uint8),
                            _piano(paths[i])))
                if (i + 1) % 500 == 0:
                    print(f"   inference {i + 1}/{len(ds)}", flush=True)
        return res

    pool = mp.Pool(args.procs)
    report = {"ckpt": args.ckpt, "step": int(ck["step"])}

    # ── 1. threshold sweep on validation ───────────────────────────────────
    va = clips(args.data, "validation")
    random.Random(0).shuffle(va)
    va = sorted(va[: args.val_clips])
    t0 = time.time()
    print(f"\n1 THRESHOLD SWEEP on {len(va)} validation clips", flush=True)
    vres = infer(va)
    grid = [(o, f) for o in ONSET_THRS for f in FRAME_THRS]
    acc = collections.defaultdict(list)
    for _, out in pool.imap_unordered(
            _score_clip, ((i, r[0], r[1], r[2], r[3], r[4], grid) for i, r in enumerate(vres)),
            chunksize=8):
        for k, s in out.items():
            acc[k].append(s)
    print(f"   {'onset thr':>9}{'frame thr':>10}{'note F1':>9}{'P':>7}{'R':>7}")
    best = max(grid, key=lambda k: mean(acc[k], "conp_f1"))
    for k in grid:
        mark = "  <- best" if k == best else ""
        print(f"   {k[0]:>9.2f}{k[1]:>10.2f}{mean(acc[k], 'conp_f1'):>9.3f}"
              f"{mean(acc[k], 'conp_p'):>7.3f}{mean(acc[k], 'conp_r'):>7.3f}{mark}")
    edge = best[0] in (ONSET_THRS[0], ONSET_THRS[-1])
    if edge:
        print("   WARNING: best onset threshold is at the edge of the sweep; it does not "
              "bracket a peak")
    report["sweep_best"] = {"onset_thr": best[0], "frame_thr": best[1],
                            "val_note_f1": mean(acc[best], "conp_f1")}
    del vres
    print(f"   ({time.time() - t0:.0f} s)", flush=True)

    # ── 2 + 3. test, all clips, per piano ──────────────────────────────────
    te = clips(args.data, "test")
    if args.limit_test:
        te = te[:: max(1, len(te) // args.limit_test)][: args.limit_test]
    print(f"\n2 TEST on all {len(te)} test clips (never seen in training or selection)",
          flush=True)
    tres = infer(te)
    both = [best, (0.5, 0.5)]
    per = {k: [None] * len(tres) for k in both}
    for i, out in pool.imap_unordered(
            _score_clip, ((i, r[0], r[1], r[2], r[3], r[4], both) for i, r in enumerate(tres)),
            chunksize=8):
        for k in both:
            per[k][i] = out[k]
    for k, label in ((best, "tuned"), ((0.5, 0.5), "untuned 0.5")):
        rows = per[k]
        print(f"   {label:<12} thr {k[0]:.2f}/{k[1]:.2f}:  note F1 {mean(rows, 'conp_f1'):.3f}  "
              f"(P {mean(rows, 'conp_p'):.3f}  R {mean(rows, 'conp_r'):.3f})   "
              f"with offsets {mean(rows, 'conpoff_f1'):.3f}")
    report["test"] = {"tuned": {k: mean(per[best], k) for k in ("conp_f1", "conp_p", "conp_r", "conpoff_f1")},
                      "untuned": {k: mean(per[(0.5, 0.5)], k) for k in ("conp_f1", "conp_p", "conp_r", "conpoff_f1")},
                      "n": len(tres)}
    print("\n3 PER PIANO (test, tuned threshold)")
    pianos = sorted({r[5] for r in tres})
    report["per_piano"] = {}
    for p in pianos:
        rows = [per[best][i] for i, r in enumerate(tres) if r[5] == p]
        report["per_piano"][p] = mean(rows, "conp_f1")
        print(f"   {p:<11} {len(rows):>5} clips   note F1 {mean(rows, 'conp_f1'):.3f}  "
              f"(P {mean(rows, 'conp_p'):.3f}  R {mean(rows, 'conp_r'):.3f})")

    # ── 4. baseline on a fixed, piano-balanced subset ──────────────────────
    if not args.no_baseline:
        # Loaded directly, exactly as app/services/transcription.py does it,
        # rather than importing the whole service and its dependency chain.
        import librosa
        from piano_transcription_inference import PianoTranscription
        from piano_transcription_inference import sample_rate as OF_SR
        ckpt_of = pathlib.Path.home() / "piano_transcription_inference_data" / \
            "note_F1=0.9677_pedal_F1=0.9186.pth"
        tr_of = PianoTranscription(device="cuda", checkpoint_path=str(ckpt_of))
        by_p = collections.defaultdict(list)
        for i, r in enumerate(tres):
            by_p[r[5]].append(i)
        rng = random.Random(0)
        take = args.baseline_clips // len(by_p)
        sub = sorted(i for p in by_p for i in rng.sample(by_p[p], min(take, len(by_p[p]))))
        print(f"\n4 BASELINE (ByteDance piano transcriber) on {len(sub)} test clips, "
              f"{take} per piano", flush=True)
        jobs = []
        for n, i in enumerate(sub):
            y = np.load(te[i])["y"].astype(np.float32) / 32767.0
            y16 = librosa.resample(y, orig_sr=S.SR, target_sr=OF_SR)
            ev = tr_of.transcribe(y16, midi_path=None).get("est_note_events", [])
            est = [{"pitch": int(e["midi_note"]), "onset": float(e["onset_time"]),
                    "offset": float(e["offset_time"]), "velocity": 80}
                   for e in ev if 21 <= int(e["midi_note"]) <= 108
                   and float(e["offset_time"]) > float(e["onset_time"])]
            jobs.append((i, est, tres[i][3], tres[i][4]))
            if (n + 1) % 100 == 0:
                print(f"   baseline {n + 1}/{len(sub)}", flush=True)
        base = dict(pool.imap_unordered(_score_notes, jobs, chunksize=4))
        ours = [per[best][i] for i in sub]
        theirs = [base[i] for i in sub]
        print(f"   {'model':<26}{'note F1':>9}{'P':>7}{'R':>7}{'+offsets':>10}")
        for name, rows in (("spectral (ours, tuned)", ours), ("ByteDance baseline", theirs)):
            print(f"   {name:<26}{mean(rows, 'conp_f1'):>9.3f}{mean(rows, 'conp_p'):>7.3f}"
                  f"{mean(rows, 'conp_r'):>7.3f}{mean(rows, 'conpoff_f1'):>10.3f}")
        print("   per piano (ours / baseline):")
        report["baseline"] = {"ours": mean(ours, "conp_f1"), "bytedance": mean(theirs, "conp_f1"),
                              "n": len(sub), "per_piano": {}}
        for p in pianos:
            o = [per[best][i] for i in sub if tres[i][5] == p]
            t = [base[i] for i in sub if tres[i][5] == p]
            report["baseline"]["per_piano"][p] = [mean(o, "conp_f1"), mean(t, "conp_f1")]
            print(f"     {p:<11} {mean(o, 'conp_f1'):.3f} / {mean(t, 'conp_f1'):.3f}")

    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(report, indent=1))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
