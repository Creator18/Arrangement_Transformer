"""Stage 1 of the hum plan: the frozen solo-piano transcriber + LoRA learns to hear piano notes in a HUM.

Input: sine-tone hums of rendered piano clips (eval/make_piano_hums.py), in
the same audio format the transcriber was trained on, with the clips' exact
note labels.  The base (spectral_real, note F1 0.955 on real piano) is
frozen; LoRA adapters (rank --lora) on every block's time/pitch attention and
feed-forward, plus the output head, are trained.  Turning the adapters off
gives back the solo-piano transcriber unchanged.

Validation: note F1 of the adapted model on held-out hums, next to the frozen
base on the same hums (the starting point: it mistakes swells for strikes).

    python -m app.ml.spectral.hum_train --data /scratch/.../piano_hums --base <spectral_real/best.pt> \\
        --ckpt-dir <dir>
"""
from __future__ import annotations

import argparse
import math
import os
import pathlib
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import frontend as S
from .data import RenderedClips, clips
from .imagine import add_lora
from .model import from_checkpoint
from .train import log_to, note_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--base", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--lora", type=int, default=16)
    ap.add_argument("--crop", type=float, default=8.0); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--total-steps", type=int, default=15000); ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--val-clips", type=int, default=300); ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--max-minutes", type=float, default=460)
    a = ap.parse_args()
    ckdir = pathlib.Path(os.path.expanduser(a.ckpt_dir)); ckdir.mkdir(parents=True, exist_ok=True)
    say = log_to(ckdir / "train.log"); t0 = time.time(); dev = "cuda"
    bf16 = torch.cuda.is_bf16_supported(); amp = torch.bfloat16 if bf16 else torch.float16

    ck = torch.load(os.path.expanduser(a.base), map_location=dev, weights_only=False)
    fe = S.SpectralFrontend().to(dev).eval(); fe.mu.copy_(ck["fe_mu"]); fe.sd.copy_(ck["fe_sd"])
    net = from_checkpoint(ck); add_lora(net, a.lora)
    for p in net.head.parameters():
        p.requires_grad_(True)
    net = net.to(dev)
    base = from_checkpoint(ck).to(dev).eval()
    for p in base.parameters():
        p.requires_grad_(False)
    params = [p for p in net.parameters() if p.requires_grad]
    say(f"hum stage 1: frozen {pathlib.Path(a.base).parent.name} + LoRA r={a.lora}; "
        f"trainable {sum(p.numel() for p in params) / 1e3:.0f}k of {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M")

    tr = clips(pathlib.Path(a.data), "train")
    va = random.Random(0).sample(clips(pathlib.Path(a.data), "validation"), a.val_clips)
    say(f"hums: {len(tr)} train / {len(va)} val")
    tr_dl = DataLoader(RenderedClips(tr, a.crop, random_crop=True), batch_size=a.batch, shuffle=True,
                       num_workers=a.workers, pin_memory=True, drop_last=True, persistent_workers=a.workers > 0)
    va_dl = DataLoader(RenderedClips(va, 20.0, random_crop=False), batch_size=2, num_workers=a.workers)

    def validate(models):
        for m in models.values():
            m.eval()
        P = {k: ([], []) for k in models}; G = ([], [])
        with torch.no_grad():
            for wav, on, fr, *_ in va_dl:
                x = fe(wav.to(dev))
                for k, m in models.items():
                    with torch.autocast("cuda", dtype=amp):
                        o = m(x)
                    P[k][0].append(torch.sigmoid(o[0].float()).cpu().numpy())
                    P[k][1].append(torch.sigmoid(o[1].float()).cpu().numpy())
                G[0].append(on.transpose(1, 2).numpy()); G[1].append(fr.transpose(1, 2).numpy())
        net.train()
        go, gf = np.concatenate(G[0]), np.concatenate(G[1])
        return {k: note_f1(np.concatenate(po), np.concatenate(pf), go, gf) for k, (po, pf) in P.items()}

    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / a.warmup, 0.5 * (
        1 + math.cos(math.pi * min(1.0, max(0, s - a.warmup) / max(1, a.total_steps - a.warmup))))))
    scaler = torch.amp.GradScaler("cuda", enabled=not bf16)
    step, best = 0, -1.0
    last = ckdir / "last.pt"
    if last.exists():
        c = torch.load(last, map_location=dev, weights_only=False)
        net.load_state_dict(c["model"]); opt.load_state_dict(c["opt"]); sched.load_state_dict(c["sched"])
        step, best = c["step"], c["best"]; say(f"resumed at step {step}")
    else:
        r = validate({"frozen base": base})["frozen base"]
        say(f"VAL step 0: frozen base on hums  note F1 {r[0]['conp_f1']:.3f} "
            f"(P {r[0]['conp_p']:.3f} R {r[0]['conp_r']:.3f})  frame F1 {r[1]:.3f}")

    def save(path):
        tmp = path.with_suffix(".tmp")
        torch.save({"model": net.state_dict(), "arch": ck["arch"], "lora": a.lora, "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "step": step, "best": best,
                    "fe_mu": fe.mu.cpu(), "fe_sd": fe.sd.cpu(), "args": vars(a)}, tmp)
        tmp.replace(path)

    pw_on, pw_fr = torch.tensor(30.0, device=dev), torch.tensor(2.0, device=dev)
    run, n, t_log = 0.0, 0, time.time()
    net.train()
    while step < a.total_steps:
        for wav, on, fr, vel, hv in tr_dl:
            with torch.no_grad():
                x = fe(wav.to(dev, non_blocking=True))
            onT, frT = on.to(dev).transpose(1, 2), fr.to(dev).transpose(1, 2)
            vlT, hv = vel.to(dev).transpose(1, 2), hv.to(dev)
            with torch.autocast("cuda", dtype=amp):
                o = net(x)
            a_, b_, c_ = (t.float() for t in o[:3])
            m = (onT > 0.5) * hv.view(-1, 1, 1)
            l_vel = (((torch.sigmoid(c_) - vlT) ** 2) * m).sum() / m.sum().clamp(min=1)
            loss = (F.binary_cross_entropy_with_logits(a_, onT, pos_weight=pw_on)
                    + F.binary_cross_entropy_with_logits(b_, frT, pos_weight=pw_fr) + 0.5 * l_vel)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update(); sched.step(); step += 1
            run += float(loss); n += 1
            if step % 100 == 0:
                say(f"step {step:>6}  loss {run / n:.4f}  {100 / (time.time() - t_log):.2f} steps/s")
                run, n, t_log = 0.0, 0, time.time()
            if step % a.val_every == 0 or step >= a.total_steps:
                r = validate({"hum LoRA": net})["hum LoRA"]
                if r[0]["conp_f1"] > best:
                    best = r[0]["conp_f1"]; save(ckdir / "best.pt")
                save(last)
                say(f"VAL step {step}: note F1 {r[0]['conp_f1']:.3f} (P {r[0]['conp_p']:.3f} "
                    f"R {r[0]['conp_r']:.3f})  frame F1 {r[1]:.3f}")
            if (time.time() - t0) / 60 > a.max_minutes:
                save(last); say("time limit: saved last.pt, resubmit to resume"); return
            if step >= a.total_steps:
                break
    say(f"done, best note F1 {best:.3f}")


if __name__ == "__main__":
    main()
