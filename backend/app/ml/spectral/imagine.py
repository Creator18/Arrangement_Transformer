"""Hearing and imagination: song -> the piano a pianist would play, in pitch space, then the piano ears.

The user's idea.  A = the song as heard, B = the human arrangement played on a
piano (eval/make_m2m_pairs.py), both through the same HCQT front end.  The
imaginer is v1 (song-to-piano transcriber, difficulty-conditioned) with one
new head that EDITS the song's features toward the piano's:

    x_hat = x_song + delta(x_song, difficulty)        delta zero-initialised

so at step 0 it passes the song through unchanged, and everything it learns
is what to keep, drop or add to what it hears.  x_hat goes into the FROZEN
piano transcriber (spectral_real, F1 0.955 on real piano) to become notes.

Losses
  feat   L1(x_hat, x_piano)                    imagine the piano's sound
  notes  BCE(piano_ears(x_hat), arrangement)   ... well enough to be heard right
  kda    per key and frame, from A = what the piano ears hear in the SONG and
         B = the arrangement:  kept A&B,  dropped A-B,  added B-A
Validation reports note F1 of the imagined piano against the human
arrangement next to v1 alone (same clips) and the oracle (the real rendered
arrangement through the piano ears: the ceiling of this design).

    python -m app.ml.spectral.imagine --data /scratch/.../m2m_pairs --ckpt-dir <dir> \\
        --init-v1 <s2p_v1/best.pt> --ears <spectral_real/best.pt>
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from . import frontend as S
from .model import from_checkpoint, PATCH_W, F_PAD
from .train import log_to, note_f1


class Pairs(Dataset):
    def __init__(self, paths, crop_s=8.0, random_crop=True):
        self.paths, self.random_crop = paths, random_crop
        self.n = int(crop_s * S.SR) if crop_s else None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        z = np.load(self.paths[i])
        ys = z["y"].astype(np.float32) / 32767; yp = z["yp"].astype(np.float32) / 32767
        on, fr = z["on"].astype(np.float32), z["fr"].astype(np.float32)
        if self.n and len(ys) > self.n:
            T = on.shape[1]; nT = S.n_frames(self.n)
            f0 = random.randrange(0, T - nT) if self.random_crop else 0
            a = int(round(f0 / S.FPS * S.SR))
            ys, yp = ys[a:a + self.n], yp[a:a + self.n]
            ys = np.pad(ys, (0, self.n - len(ys))); yp = np.pad(yp, (0, self.n - len(yp)))
            on, fr = on[:, f0:f0 + nT], fr[:, f0:f0 + nT]
        return (torch.from_numpy(ys), torch.from_numpy(yp), torch.from_numpy(on),
                torch.from_numpy(fr), torch.tensor(int(z["difficulty"])))


class LoRALinear(nn.Module):
    """W x + (alpha/r) B A x, with W frozen and B zero-initialised (starts exactly as W)."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.t().to(x.dtype)) @ self.B.t().to(x.dtype) * self.scale


def add_lora(v1, r):
    """Freeze v1; adapters on every block's time/pitch attention and feed-forward."""
    for p in v1.parameters():
        p.requires_grad_(False)
    for blk in v1.blocks:
        for att in (blk.time, blk.pitch):
            att.qkv, att.out = LoRALinear(att.qkv, r), LoRALinear(att.out, r)
        blk.ff[0], blk.ff[2] = LoRALinear(blk.ff[0], r), LoRALinear(blk.ff[2], r)


class Imaginer(nn.Module):
    """v1's trunk + a zero-initialised feature-edit head and a kept/dropped/added head."""

    def __init__(self, v1, lora_r: int = 0):
        super().__init__()
        if lora_r:
            add_lora(v1, lora_r)
        self.v1 = v1
        c = v1.arch.get("stem", 32); C = len(S.HARMONICS) + 1
        self.edit = nn.Conv2d(c, C, 3, padding=1)
        nn.init.zeros_(self.edit.weight); nn.init.zeros_(self.edit.bias)
        self.kda = nn.Conv2d(c, 3, (3, 1), stride=(3, 1))

    def trunk(self, x, cond):
        m = self.v1
        B, _, Fb, T = x.shape
        s = m.stem(x)
        Tp = math.ceil(T / PATCH_W) * PATCH_W
        z = m.patch(F.pad(s, (0, Tp - T, 0, F_PAD - Fb))).permute(0, 2, 3, 1) + m.pitch_emb
        if m.cond_emb is not None and cond is not None:
            z = z + m.cond_emb(cond.long())[:, None, None, :]
        for blk in m.blocks:
            z = blk(z)
        z = m.norm(z).permute(0, 3, 1, 2)
        u = m.unpatch(z)[:, :, :Fb, :T]
        return m.fuse(torch.cat([u, s], dim=1))

    def forward(self, x, cond):
        h = self.trunk(x, cond)
        return x + self.edit(h), self.kda(h).permute(1, 0, 3, 2)       # (B,7,264,T), (3,B,T,88)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--init-v1", required=True); ap.add_argument("--ears", required=True)
    ap.add_argument("--crop", type=float, default=8.0); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--total-steps", type=int, default=20000)
    ap.add_argument("--val-every", type=int, default=1000); ap.add_argument("--val-clips", type=int, default=240)
    ap.add_argument("--lora", type=int, default=8, help="LoRA rank on frozen v1; 0 = full fine-tune")
    ap.add_argument("--w-feat", type=float, default=0.1); ap.add_argument("--w-notes", type=float, default=1.0)
    ap.add_argument("--w-kda", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--max-minutes", type=float, default=460)
    ap.add_argument("--limit-train", type=int, default=0)
    a = ap.parse_args()
    ckdir = pathlib.Path(os.path.expanduser(a.ckpt_dir)); ckdir.mkdir(parents=True, exist_ok=True)
    say = log_to(ckdir / "train.log"); t_start = time.time()
    dev = "cuda"
    bf16 = torch.cuda.is_bf16_supported(); amp = torch.bfloat16 if bf16 else torch.float16

    ck1 = torch.load(os.path.expanduser(a.init_v1), map_location=dev, weights_only=False)
    cke = torch.load(os.path.expanduser(a.ears), map_location=dev, weights_only=False)
    fe = S.SpectralFrontend().to(dev).eval()
    fe.mu.copy_(ck1["fe_mu"]); fe.sd.copy_(ck1["fe_sd"])
    if not (torch.allclose(ck1["fe_mu"], cke["fe_mu"]) and torch.allclose(ck1["fe_sd"], cke["fe_sd"])):
        say("WARNING: v1 and the piano ears use different front-end normalisation")
    net = Imaginer(from_checkpoint(ck1), a.lora).to(dev)
    nt = sum(p.numel() for p in net.parameters() if p.requires_grad)
    say(f"trainable {nt / 1e3:.0f}k of {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M params (LoRA r={a.lora})")
    v1 = from_checkpoint(ck1).to(dev).eval()                    # baseline, frozen copy
    ears = from_checkpoint(cke).to(dev).eval()
    for p in list(ears.parameters()) + list(v1.parameters()):
        p.requires_grad_(False)

    root = pathlib.Path(a.data)
    tr = sorted((root / "train").glob("*.npz")); va = sorted((root / "validation").glob("*.npz"))
    if a.limit_train:
        tr = tr[: a.limit_train]
    va = random.Random(0).sample(va, min(a.val_clips, len(va)))
    say(f"pairs: {len(tr)} train / {len(va)} val")
    tr_dl = DataLoader(Pairs(tr, a.crop), batch_size=a.batch, shuffle=True, num_workers=a.workers,
                       pin_memory=True, drop_last=True, persistent_workers=a.workers > 0)
    va_dl = DataLoader(Pairs(va, 20.0, random_crop=False), batch_size=2, num_workers=a.workers)

    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / a.warmup, 0.5 * (
        1 + math.cos(math.pi * min(1.0, max(0, s - a.warmup) / max(1, a.total_steps - a.warmup))))))
    scaler = torch.amp.GradScaler("cuda", enabled=not bf16)
    step, best = 0, -1.0
    last = ckdir / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=dev, weights_only=False)
        net.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        step, best = ck["step"], ck["best"]; say(f"resumed at step {step}")
    pw_on, pw_fr = torch.tensor(30.0, device=dev), torch.tensor(2.0, device=dev)

    def save(path):
        tmp = path.with_suffix(".tmp")
        torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "step": step, "best": best, "fe_mu": fe.mu.cpu(), "fe_sd": fe.sd.cpu(),
                    "v1_arch": ck1["arch"], "args": vars(a)}, tmp)
        tmp.replace(path)

    def validate():
        net.eval()
        R = {k: ([], []) for k in ("imagined", "v1", "oracle", "song_ears")}
        G = ([], [])
        with torch.no_grad():
            for ys, yp, on, fr, d in va_dl:
                xs, xp = fe(ys.to(dev)), fe(yp.to(dev)); d = d.to(dev)
                with torch.autocast("cuda", dtype=amp):
                    xh, _ = net(xs, d)
                    outs = {"imagined": ears(xh.float()), "v1": v1(xs, d), "oracle": ears(xp),
                            "song_ears": ears(xs)}
                for k, o in outs.items():
                    R[k][0].append(torch.sigmoid(o[0].float()).cpu().numpy())
                    R[k][1].append(torch.sigmoid(o[1].float()).cpu().numpy())
                G[0].append(on.transpose(1, 2).numpy()); G[1].append(fr.transpose(1, 2).numpy())
        net.train()
        go, gf = np.concatenate(G[0]), np.concatenate(G[1])
        return {k: note_f1(np.concatenate(po), np.concatenate(pf), go, gf)[0] for k, (po, pf) in R.items()}

    net.train(); t_log, acc = time.time(), np.zeros(4); n = 0
    while step < a.total_steps:
        for ys, yp, on, fr, d in tr_dl:
            with torch.no_grad():
                xs, xp = fe(ys.to(dev, non_blocking=True)), fe(yp.to(dev, non_blocking=True))
                onT = on.to(dev).transpose(1, 2); frT = fr.to(dev).transpose(1, 2)
                with torch.autocast("cuda", dtype=amp):
                    A = torch.sigmoid(ears(xs)[1].float()) > 0.5          # what the ears hear in the song
                B = frT > 0.5
                kda_t = torch.stack([A & B, A & ~B, B & ~A]).float()
            d = d.to(dev)
            with torch.autocast("cuda", dtype=amp):
                xh, kz = net(xs, d)
                o = ears(xh)
            l_feat = F.l1_loss(xh.float(), xp)
            l_notes = (F.binary_cross_entropy_with_logits(o[0].float(), onT, pos_weight=pw_on)
                       + F.binary_cross_entropy_with_logits(o[1].float(), frT, pos_weight=pw_fr))
            l_kda = F.binary_cross_entropy_with_logits(kz.float(), kda_t, pos_weight=pw_fr)
            loss = a.w_feat * l_feat + a.w_notes * l_notes + a.w_kda * l_kda
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); step += 1
            acc += [float(loss), float(l_feat), float(l_notes), float(l_kda)]; n += 1
            if step % 100 == 0:
                m = acc / n
                say(f"step {step:>6}  loss {m[0]:.4f}  feat {m[1]:.4f}  notes {m[2]:.4f}  kda {m[3]:.4f}  "
                    f"{100 / (time.time() - t_log):.2f} steps/s")
                t_log, acc, n = time.time(), np.zeros(4), 0
            if step % a.val_every == 0 or step >= a.total_steps:
                r = validate()
                if r["imagined"]["conp_f1"] > best:
                    best = r["imagined"]["conp_f1"]; save(ckdir / "best.pt")
                save(last)
                say(f"VAL step {step}: " + "  ".join(
                    f"{k} F1 {v['conp_f1']:.3f} (P {v['conp_p']:.3f} R {v['conp_r']:.3f})" for k, v in r.items()))
            if (time.time() - t_start) / 60 > a.max_minutes:
                save(last); say("time limit: saved last.pt, resubmit to resume"); return
            if step >= a.total_steps:
                break
    say(f"done, best imagined note F1 {best:.3f}")


if __name__ == "__main__":
    main()
