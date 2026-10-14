"""Stage-1 hum checks: are strikes MISSING or MISPLACED?

  --stage rolls (GPU)  run the hum-LoRA checkpoint and the frozen base on held-out hums,
                       save onset/frame probabilities (float16)
  --stage score (CPU)  note F1 over a sweep of onset thresholds (check 2) at strict 50 ms
                       and tolerant 150 ms onset windows (check 3), both models

    python eval/hum_s1_checks.py --stage rolls --data <piano_hums> --ckpt <hum_s1/best.pt> --base <spectral_real/best.pt> --out <dir>
    python eval/hum_s1_checks.py --stage score --out <dir>
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import pathlib
import random
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
THRS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def rolls(a):
    import torch
    from app.ml.spectral import frontend as S
    from app.ml.spectral.data import RenderedClips, clips
    from app.ml.spectral.imagine import add_lora
    from app.ml.spectral.model import SpectralTranscriber, from_checkpoint
    dev = "cuda"
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    net = SpectralTranscriber(**ck["arch"])
    if ck.get("lora"):                                  # LoRA stage 1; from-scratch checkpoints have none
        add_lora(net, ck["lora"])
    net.load_state_dict(ck["model"])
    base = from_checkpoint(torch.load(a.base, map_location=dev, weights_only=False))
    models = {"hum_lora": net.to(dev).eval(), "frozen_base": base.to(dev).eval()}
    fe = S.SpectralFrontend().to(dev).eval(); fe.mu.copy_(ck["fe_mu"]); fe.sd.copy_(ck["fe_sd"])
    va = clips(pathlib.Path(a.data), "validation")
    va = random.Random(0).sample(va, min(a.n, len(va)))     # same 300 as training's val (when n covers it)
    ds = RenderedClips(va, 20.0, random_crop=False)
    out = {k: ([], []) for k in models}; G = ([], [])
    with torch.no_grad():
        for i in range(len(ds)):
            wav, on, fr = ds[i][:3]
            x = fe(wav[None].to(dev))
            for k, m in models.items():
                o = m(x)
                out[k][0].append(torch.sigmoid(o[0].float())[0].cpu().numpy().astype(np.float16))
                out[k][1].append(torch.sigmoid(o[1].float())[0].cpu().numpy().astype(np.float16))
            G[0].append(on.T.numpy().astype(np.uint8)); G[1].append(fr.T.numpy().astype(np.uint8))
    o = pathlib.Path(a.out); o.mkdir(parents=True, exist_ok=True)
    np.savez(o / "rolls.npz", **{f"{k}_on": np.stack(v[0]) for k, v in out.items()},
             **{f"{k}_fr": np.stack(v[1]) for k, v in out.items()}, gt_on=np.stack(G[0]), gt_fr=np.stack(G[1]),
             step=ck["step"], names=np.array([p.name for p in va]))
    print(f"saved rolls for {len(va)} clips (checkpoint step {ck['step']})")


def _one(args):
    from app.ml.transcriber.decode import decode_notes
    from mir_eval.transcription import precision_recall_f1_overlap as prf
    po, pf, go, gf, thr = args
    full = dict(min_frames=1, min_note_s=0.0, keep_restruck=True)
    est = decode_notes(po.T.astype(np.float32), pf.T.astype(np.float32), np.zeros_like(po.T, np.float32), thr, 0.5, **full)
    ref = decode_notes(go.T.astype(np.float32), gf.T.astype(np.float32), np.zeros_like(go.T, np.float32), .5, .5, **full)
    arr = lambda ns: (np.array([[n["onset"], max(n["offset"], n["onset"] + 1e-3)] for n in ns]).reshape(-1, 2),
                      np.array([440 * 2 ** ((n["pitch"] - 69) / 12) for n in ns]))
    if not ref:
        return None
    ri, rp = arr(ref)
    if not est:
        return (0, 0, 0, 0, 0, 0, len(ref), 0)
    ei, ep = arr(est)
    s = prf(ri, rp, ei, ep, onset_tolerance=0.05, offset_ratio=None)
    t = prf(ri, rp, ei, ep, onset_tolerance=0.15, offset_ratio=None)
    return (*s[:3], *t[:3], len(ref), len(est))


def score(a):
    z = np.load(pathlib.Path(a.out) / "rolls.npz")
    names = list(z["names"]) if "names" in z.files else []
    groups = {"all": np.arange(len(z["gt_on"]))}
    for tag, label in (("mae", "classical"), ("m2m", "pop arrangements"), ("pop", "POP909")):
        g = np.array([i for i, n in enumerate(names) if str(n).startswith(tag + "_")])
        if len(g):
            groups[label] = g
    for gname, gi in groups.items():
        print(f"===== {gname}: {len(gi)} clips =====")
        _score(z, gi, a)


def _offsets(args):
    """Signed onset errors (s) of matched notes within 150 ms, estimate minus truth."""
    from app.ml.transcriber.decode import decode_notes
    from mir_eval.transcription import match_notes
    po, pf, go, gf, thr = args
    full = dict(min_frames=1, min_note_s=0.0, keep_restruck=True)
    est = decode_notes(po.T.astype(np.float32), pf.T.astype(np.float32), np.zeros_like(po.T, np.float32), thr, 0.5, **full)
    ref = decode_notes(go.T.astype(np.float32), gf.T.astype(np.float32), np.zeros_like(go.T, np.float32), .5, .5, **full)
    if not est or not ref:
        return []
    arr = lambda ns: (np.array([[n["onset"], max(n["offset"], n["onset"] + 1e-3)] for n in ns]),
                      np.array([440 * 2 ** ((n["pitch"] - 69) / 12) for n in ns]))
    (ri, rp), (ei, ep) = arr(ref), arr(est)
    return [ei[b, 0] - ri[x, 0] for x, b in match_notes(ri, rp, ei, ep, onset_tolerance=0.15, offset_ratio=None)]


def _score(z, gi, a):
    go, gf = z["gt_on"][gi], z["gt_fr"][gi]
    print(f"checkpoint step {int(z['step'])}, {len(go)} held-out hums; note F1 by onset threshold\n")
    print(f"{'model':<13}{'thr':>5}  {'strict 50 ms  F1 (P / R)':<28}{'tolerant 150 ms  F1 (P / R)':<30}{'notes/ref':>9}")
    with mp.Pool(max(1, mp.cpu_count() - 1)) as pool:
        for k in ("frozen_base", "hum_lora"):
            po, pf = z[f"{k}_on"][gi], z[f"{k}_fr"][gi]
            best = (-1, None)
            for thr in THRS:
                r = [x for x in pool.map(_one, [(po[i], pf[i], go[i], gf[i], thr) for i in range(len(go))]) if x]
                m = np.mean(np.array(r)[:, :6], 0); ratio = sum(x[7] for x in r) / max(1, sum(x[6] for x in r))
                best = max(best, (m[2], thr))
                print(f"{k:<13}{thr:>5.1f}  {m[2]:.3f} ({m[0]:.3f} / {m[1]:.3f}){'':<8}"
                      f"{m[5]:.3f} ({m[3]:.3f} / {m[4]:.3f}){'':<10}{ratio:>9.2f}")
            d = np.array([x for r in pool.map(_offsets, [(po[i], pf[i], go[i], gf[i], best[1]) for i in range(len(go))])
                          for x in r]) * 1000
            if len(d):
                print(f"  timing at its best threshold {best[1]}: median {np.median(d):+.0f} ms; early >16 ms "
                      f"{np.mean(d < -16):.0%}, within 16 ms {np.mean(np.abs(d) <= 16):.0%}, late >16 ms {np.mean(d > 16):.0%}")
            print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["rolls", "score"], required=True)
    ap.add_argument("--data"); ap.add_argument("--ckpt"); ap.add_argument("--base")
    ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    rolls(a) if a.stage == "rolls" else score(a)


if __name__ == "__main__":
    main()
