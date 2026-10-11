"""Can the patch transformer memorise 10 clips?  A wiring test, not a result.

The full model, trained on only 10 rendered clips with fixed crops, should
drive training note F1 close to 1.  If it cannot memorise ten clips, something
is broken — labels misaligned with audio, a loss that does not reach the
heads, a layer that is not learning — and that is far cheaper to find in five
minutes here than after hours on the cluster.

This exact check once caught an underpowered setup in this project: an
overfit run with too few gradient steps was read as "the model cannot learn",
and the tell was poor scores on TRAINING data.  So this runs enough steps to
memorise, and reports training F1 directly.

Velocity is not trained: rendered clips store onset and frame labels only.

    venv/Scripts/python.exe eval/spectral_overfit.py [--steps 600]
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from app.ml.spectral import frontend as S                  # noqa: E402
from app.ml.spectral import render as R                    # noqa: E402
from app.ml.spectral.model import SpectralTranscriber      # noqa: E402
from app.ml.transcriber.decode import decode_notes, note_scores   # noqa: E402

DEV = torch.device("cuda")
CROP_S = 8.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=10)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    # Salamander-only Phase 2 cache: on disk locally, labels already aligned.
    paths = R.build("train", args.clips)
    n = int(CROP_S * S.SR)
    T = S.n_frames(n)
    ys, ons, frs = [], [], []
    for p in paths:
        y, on, fr = R.load(p)
        ys.append(y[:n]); ons.append(on[:, :T]); frs.append(fr[:, :T])
    wav = torch.from_numpy(np.stack(ys)).to(DEV)
    on_t = torch.from_numpy(np.stack(ons)).float().to(DEV).transpose(1, 2)   # (B, T, 88)
    fr_t = torch.from_numpy(np.stack(frs)).float().to(DEV).transpose(1, 2)
    print(f"{len(paths)} clips x {CROP_S:.0f} s, {int(on_t.sum())} onsets, "
          f"{int(fr_t.sum())} active key-frames")

    fe = S.SpectralFrontend().to(DEV).eval()
    fe.calibrate([wav])
    with torch.no_grad():
        x = fe(wav)

    net = SpectralTranscriber().to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.0)
    sc = torch.amp.GradScaler("cuda")
    pw_on, pw_fr = torch.tensor(30.0, device=DEV), torch.tensor(2.0, device=DEV)
    print(f"model {net.n_params() / 1e6:.2f}M params, {args.steps} steps\n")
    print(f"{'step':>6}{'loss':>9}{'note F1':>9}{'P':>7}{'R':>7}{'frame F1':>10}{'time':>7}")

    def evaluate():
        net.eval()
        with torch.no_grad():
            a, b, _ = net(x)
        net.train()
        po, pf = torch.sigmoid(a).cpu().numpy(), torch.sigmoid(b).cpu().numpy()
        go, gf = on_t.cpu().numpy(), fr_t.cpu().numpy()
        s = {"conp_f1": 0, "conp_p": 0, "conp_r": 0}
        for i in range(len(po)):
            est = decode_notes(po[i].T, pf[i].T, np.zeros_like(po[i].T), 0.5, 0.5)
            ref = decode_notes(go[i].T, gf[i].T, np.zeros_like(go[i].T), 0.5, 0.5)
            r = note_scores(ref, est)
            for k in s:
                s[k] += r[k] / len(po)
        pf_b, gf_b = pf >= 0.5, gf >= 0.5
        tp = (pf_b & gf_b).sum(); fp = (pf_b & ~gf_b).sum(); fn = (~pf_b & gf_b).sum()
        ff = 2 * tp / max(1, 2 * tp + fp + fn)
        return s, ff

    t0 = time.time()
    for step in range(1, args.steps + 1):
        with torch.amp.autocast("cuda"):
            a, b, _ = net(x)
            loss = (F.binary_cross_entropy_with_logits(a, on_t, pos_weight=pw_on)
                    + F.binary_cross_entropy_with_logits(b, fr_t, pos_weight=pw_fr))
        opt.zero_grad(set_to_none=True)
        sc.scale(loss).backward()
        sc.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        sc.step(opt); sc.update()
        if step % 50 == 0 or step == 1:
            s, ff = evaluate()
            print(f"{step:>6}{float(loss):>9.4f}{s['conp_f1']:>9.3f}{s['conp_p']:>7.3f}"
                  f"{s['conp_r']:>7.3f}{ff:>10.3f}{time.time() - t0:>6.0f}s", flush=True)

    s, ff = evaluate()
    print(f"\nfinal training note F1 {s['conp_f1']:.3f}, frame F1 {ff:.3f}")
    print("-> PASS: the architecture can memorise" if s["conp_f1"] > 0.9 else
          "-> FAIL: cannot memorise 10 clips; look for a wiring bug before scaling up")


if __name__ == "__main__":
    main()
