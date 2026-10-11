"""Check the patch transformer before training it.

  1 SHAPES     outputs are (B, T, 88) with T = the front end's frame count,
               for even AND odd clip lengths (patches are 2 frames wide, so an
               odd T is where an off-by-one would hide)
  2 GRADIENTS  every parameter receives a gradient — a layer wired up but not
               used would train silently as dead weight
  3 SIZE       params, peak GPU memory and seconds per training step for a few
               configurations, measured the way the transcriber was sized:
               real forward + backward under AMP at the real batch size

    venv/Scripts/python.exe eval/spectral_model_check.py
"""
from __future__ import annotations

import pathlib
import sys
import time

import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.ml.spectral import frontend as S                 # noqa: E402
from app.ml.spectral.model import SpectralTranscriber     # noqa: E402

DEV = torch.device("cuda")


def main():
    fe = S.SpectralFrontend().to(DEV).eval()

    print("1 SHAPES")
    net = SpectralTranscriber().to(DEV)
    for secs in (8.0, 8.016, 3.3):
        n = int(secs * S.SR)
        with torch.no_grad():
            x = fe(torch.randn(2, n, device=DEV) * 0.1)
            on, fr, vl = net(x)
        T = S.n_frames(n)
        ok = on.shape == fr.shape == vl.shape == (2, T, S.N_KEYS)
        print(f"   {secs:>6.3f} s  features {tuple(x.shape)}  outputs {tuple(on.shape)}  "
              f"expected (2, {T}, 88)  {'ok' if ok else 'MISMATCH'}")

    print("\n2 GRADIENTS")
    net.zero_grad()
    x = fe(torch.randn(2, 8 * S.SR, device=DEV) * 0.1)
    on, fr, vl = net(x)
    (on.mean() + fr.mean() + vl.mean()).backward()
    dead = [n for n, p in net.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    print(f"   {sum(1 for _ in net.parameters())} tensors, "
          f"{'all receive gradient' if not dead else 'NO GRADIENT: ' + ', '.join(dead[:6])}")

    print("\n3 SIZE  (batch 8, AMP, forward + backward + optimiser step)")
    print(f"   {'config':<26}{'params':>9}{'8 s: mem':>11}{'s/step':>8}{'20 s: mem':>11}{'s/step':>8}")
    for d, depth, heads in ((256, 6, 4), (256, 8, 4), (384, 8, 6), (512, 8, 8)):
        row = []
        for secs in (8, 20):
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            m = SpectralTranscriber(d=d, depth=depth, heads=heads).to(DEV)
            opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
            sc = torch.amp.GradScaler("cuda")
            wav = torch.randn(8, secs * S.SR, device=DEV) * 0.1
            tgt = (torch.rand(8, S.n_frames(secs * S.SR), 88, device=DEV) > 0.98).float()
            try:
                for i in range(4):
                    if i == 1:
                        torch.cuda.synchronize(); t0 = time.time()
                    with torch.no_grad():
                        x = fe(wav)
                    with torch.amp.autocast("cuda"):
                        a, b, c = m(x)
                        loss = (F.binary_cross_entropy_with_logits(a, tgt)
                                + F.binary_cross_entropy_with_logits(b, tgt) + c.pow(2).mean())
                    opt.zero_grad(); sc.scale(loss).backward(); sc.step(opt); sc.update()
                torch.cuda.synchronize()
                row.append((torch.cuda.max_memory_allocated() / 1e9, (time.time() - t0) / 3))
            except torch.cuda.OutOfMemoryError:
                row.append(None)
            n_p = m.n_params() / 1e6
            del m, opt
        cell = lambda r: f"{r[0]:>10.2f}G{r[1]:>8.2f}" if r else f"{'OOM':>11}{'':>8}"
        print(f"   d={d} depth={depth} heads={heads:<6}{n_p:>8.2f}M{cell(row[0])}{cell(row[1])}",
              flush=True)


if __name__ == "__main__":
    main()
