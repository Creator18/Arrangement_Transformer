"""Fine-tune the render-trained arranger on new renders, with SIDE OUTPUTS for how each note was sounded.

The user's idea: the renders know, for every written note, how it was played — its attack, its
release, whether it was reached by a slide, whether its line was legato, whether it was sung
(eval/render_arrangements.py, note_meta).  Those are not given to the model (real songs never
come with them); they are extra things it must PREDICT at each note's onset, so its picture of
the sound has to tell a slow swell from a new strike, a long tail from a held note, a glide from
two notes.  Weighted lightly (--aux-weight) so the notes stay the main job.

Modes (all start from the render-trained model):
  lora        frozen model + LoRA adapters (+ output head)
  lora_stem   the same, plus the first layers (stem, patch) trainable — they read attacks and tails
  full        every layer trainable

The saved model loads in the app like any spectral checkpoint (aux heads are kept apart).

    python -m app.ml.spectral.ft_aux --data <dir with train/ validation/> --base <arr_scratch/best.pt> \\
        --ckpt-dir <dir> --mode lora_stem --aux-weight 0.2
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
from .data import clips
from .imagine import add_lora
from .model import SpectralTranscriber, from_checkpoint, PATCH_W, F_PAD
from .train import log_to, note_f1

N_AUX = 10        # attack class (4) | release class (3) | slid | legato | sung


def aux_targets(meta, T, f0):
    """note_meta rows (onset s, pitch, attack_s, release_tc, decay_tc, sustain_cb, slid, legato, sung)
    -> targets (N_AUX, 88, T) and a mask (88, T) at each note's onset frame."""
    tgt = np.zeros((N_AUX, 88, T), np.float32); mask = np.zeros((88, T), np.float32)
    for on, p, att, rel, _dec, _sus, slid, leg, sung in meta:
        k = int(p) - 21; f = int(round(on * S.FPS)) - f0
        if not (0 <= k < 88 and 0 <= f < T):
            continue
        ac = 0 if att <= 0.005 else 1 if att <= 0.05 else 2 if att <= 0.15 else 3
        rc = 0 if rel < 0 else 1 if rel < 1800 else 2
        tgt[ac, k, f] = 1; tgt[4 + rc, k, f] = 1; tgt[7, k, f] = slid; tgt[8, k, f] = leg; tgt[9, k, f] = sung
        mask[k, f] = 1
    return tgt, mask


class Clips(Dataset):
    def __init__(self, paths, crop_s=8.0, train=True):
        self.paths, self.train = paths, train
        self.n = int(crop_s * S.SR) if crop_s else None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]; z = np.load(p)
        y = z["y"].astype(np.float32) / 32767; on, fr = z["on"].astype(np.float32), z["fr"].astype(np.float32)
        vp = p.with_name(p.stem + ".vel.npy")
        vel = np.load(vp).astype(np.float32) / 127 if vp.exists() else np.zeros_like(on)
        hv = 1.0 if vp.exists() else 0.0
        meta = z["note_meta"] if "note_meta" in z.files else np.zeros((0, 9), np.float32)
        T = on.shape[1]; f0 = 0
        if self.n and len(y) > self.n:
            nT = S.n_frames(self.n)
            f0 = random.randrange(0, T - nT) if self.train else 0
            a = int(round(f0 / S.FPS * S.SR)); y = np.pad(y[a:a + self.n], (0, max(0, self.n - len(y[a:a + self.n]))))
            on, fr, vel = on[:, f0:f0 + nT], fr[:, f0:f0 + nT], vel[:, f0:f0 + nT]
        tgt, mask = aux_targets(meta, on.shape[1], f0)
        return (torch.from_numpy(y), torch.from_numpy(on), torch.from_numpy(fr), torch.from_numpy(vel),
                torch.tensor(hv), torch.from_numpy(tgt), torch.from_numpy(mask))


def trunk(net, x):
    """The transcriber's forward up to its fused features h (B, stem, 264, T)."""
    B, _, Fb, T = x.shape
    s = net.stem(x)
    Tp = math.ceil(T / PATCH_W) * PATCH_W
    z = net.patch(F.pad(s, (0, Tp - T, 0, F_PAD - Fb))).permute(0, 2, 3, 1) + net.pitch_emb
    for blk in net.blocks:
        z = blk(z)
    z = net.norm(z).permute(0, 3, 1, 2)
    u = net.unpatch(z)[:, :, :Fb, :T]
    return net.fuse(torch.cat([u, s], dim=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--base", required=True); ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--mode", choices=["lora", "lora_stem", "full"], default="lora_stem")
    ap.add_argument("--lora", type=int, default=16); ap.add_argument("--aux-weight", type=float, default=0.2)
    ap.add_argument("--crop", type=float, default=8.0); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--total-steps", type=int, default=20000); ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--val-clips", type=int, default=300); ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--max-minutes", type=float, default=460)
    a = ap.parse_args()
    ckdir = pathlib.Path(os.path.expanduser(a.ckpt_dir)); ckdir.mkdir(parents=True, exist_ok=True)
    say = log_to(ckdir / "train.log"); t0 = time.time(); dev = "cuda"
    bf16 = torch.cuda.get_device_capability(0)[0] >= 8; amp = torch.bfloat16 if bf16 else torch.float16
    ck = torch.load(os.path.expanduser(a.base), map_location=dev, weights_only=False)
    fe = S.SpectralFrontend().to(dev).eval(); fe.mu.copy_(ck["fe_mu"]); fe.sd.copy_(ck["fe_sd"])
    net = from_checkpoint(ck)
    if a.mode != "full":
        add_lora(net, a.lora)
        for p in net.head.parameters():
            p.requires_grad_(True)
        if a.mode == "lora_stem":
            for mod in (net.stem, net.patch):
                for p in mod.parameters():
                    p.requires_grad_(True)
    net = net.to(dev)
    aux = nn.Conv2d(net.arch.get("stem", 32), N_AUX, (3, 1), stride=(3, 1)).to(dev)
    params = [p for p in net.parameters() if p.requires_grad] + list(aux.parameters())
    say(f"ft_aux: mode {a.mode}, aux weight {a.aux_weight}; trainable {sum(p.numel() for p in params) / 1e3:.0f}k "
        f"of {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M; GPU {torch.cuda.get_device_name(0)}")
    tr = clips(pathlib.Path(a.data), "train")
    va = random.Random(0).sample(clips(pathlib.Path(a.data), "validation"), min(a.val_clips, len(clips(pathlib.Path(a.data), "validation"))))
    say(f"clips: {len(tr)} train / {len(va)} val")
    dl = DataLoader(Clips(tr, a.crop), batch_size=a.batch, shuffle=True, num_workers=a.workers, pin_memory=True,
                    drop_last=True, persistent_workers=a.workers > 0)
    vdl = DataLoader(Clips(va, 20.0, train=False), batch_size=2, num_workers=a.workers)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / a.warmup, 0.5 * (
        1 + math.cos(math.pi * min(1.0, max(0, s - a.warmup) / max(1, a.total_steps - a.warmup))))))
    scaler = torch.amp.GradScaler("cuda", enabled=not bf16)
    step, best = 0, -1.0; last = ckdir / "last.pt"
    if last.exists():
        c = torch.load(last, map_location=dev, weights_only=False)
        net.load_state_dict(c["model"]); aux.load_state_dict(c["aux"]); opt.load_state_dict(c["opt"]); sched.load_state_dict(c["sched"])
        step, best = c["step"], c["best"]; say(f"resumed at step {step}")

    def save(path):
        tmp = path.with_suffix(".tmp")
        torch.save({"model": net.state_dict(), "aux": aux.state_dict(), "arch": ck["arch"],
                    "lora": a.lora if a.mode != "full" else 0, "mode": a.mode, "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "step": step, "best": best, "fe_mu": fe.mu.cpu(), "fe_sd": fe.sd.cpu(),
                    "args": vars(a)}, tmp)
        tmp.replace(path)

    def validate():
        net.eval(); aux.eval(); P = ([], []); G = ([], []); acc = np.zeros(4); cnt = 0
        with torch.no_grad():
            for wav, on, fr, _v, _h, tgt, mask in vdl:
                x = fe(wav.to(dev))
                with torch.autocast("cuda", dtype=amp):
                    h = trunk(net, x); o = net.head(h).permute(1, 0, 3, 2); za = aux(h)
                P[0].append(torch.sigmoid(o[0].float()).cpu().numpy()); P[1].append(torch.sigmoid(o[1].float()).cpu().numpy())
                G[0].append(on.transpose(1, 2).numpy()); G[1].append(fr.transpose(1, 2).numpy())
                m = mask.to(dev) > 0
                if m.any():
                    za = za.float(); t = tgt.to(dev)
                    acc[0] += (za[:, :4].argmax(1)[m] == t[:, :4].argmax(1)[m]).float().sum().item()
                    acc[1] += (za[:, 4:7].argmax(1)[m] == t[:, 4:7].argmax(1)[m]).float().sum().item()
                    acc[2] += ((za[:, 7] > 0)[m] == (t[:, 7] > .5)[m]).float().sum().item()
                    acc[3] += ((za[:, 9] > 0)[m] == (t[:, 9] > .5)[m]).float().sum().item()
                    cnt += int(m.sum())
        net.train(); aux.train()
        r = note_f1(np.concatenate(P[0]), np.concatenate(P[1]), np.concatenate(G[0]), np.concatenate(G[1]))
        return r, (acc / max(1, cnt)).round(3).tolist()

    pw_on, pw_fr = torch.tensor(30.0, device=dev), torch.tensor(2.0, device=dev)
    run = np.zeros(2); n = 0; t_log = time.time(); net.train()
    while step < a.total_steps:
        for wav, on, fr, vel, hv, tgt, mask in dl:
            with torch.no_grad():
                x = fe(wav.to(dev, non_blocking=True))
            onT, frT, vlT = on.to(dev).transpose(1, 2), fr.to(dev).transpose(1, 2), vel.to(dev).transpose(1, 2)
            hv = hv.to(dev); tgt, mask = tgt.to(dev), mask.to(dev)
            with torch.autocast("cuda", dtype=amp):
                h = trunk(net, x); o = net.head(h).permute(1, 0, 3, 2); za = aux(h)
            a_, b_, c_ = (t.float() for t in (o[0], o[1], o[2]))
            mv = (onT > 0.5) * hv.view(-1, 1, 1)
            l_vel = (((torch.sigmoid(c_) - vlT) ** 2) * mv).sum() / mv.sum().clamp(min=1)
            loss = (F.binary_cross_entropy_with_logits(a_, onT, pos_weight=pw_on)
                    + F.binary_cross_entropy_with_logits(b_, frT, pos_weight=pw_fr) + 0.5 * l_vel)
            l_aux = torch.zeros((), device=dev)
            if a.aux_weight and mask.sum() > 0:
                za = za.float(); m = mask > 0
                ce = lambda lo, hi: F.cross_entropy(za[:, lo:hi].permute(0, 2, 3, 1)[m], tgt[:, lo:hi].permute(0, 2, 3, 1)[m].argmax(-1))
                bce = lambda c: F.binary_cross_entropy_with_logits(za[:, c][m], tgt[:, c][m])
                l_aux = ce(0, 4) + ce(4, 7) + bce(7) + bce(8) + bce(9)
                loss = loss + a.aux_weight * l_aux
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0); scaler.step(opt); scaler.update(); sched.step(); step += 1
            run += [float(loss), float(l_aux)]; n += 1
            if step % 100 == 0:
                say(f"step {step:>6}  loss {run[0] / n:.4f}  aux {run[1] / n:.4f}  {100 / (time.time() - t_log):.2f} steps/s")
                run, n, t_log = np.zeros(2), 0, time.time()
            if step % a.val_every == 0 or step >= a.total_steps:
                r, aa = validate()
                if r[0]["conp_f1"] > best:
                    best = r[0]["conp_f1"]; save(ckdir / "best.pt")
                save(last)
                say(f"VAL step {step}: note F1 {r[0]['conp_f1']:.3f} (P {r[0]['conp_p']:.3f} R {r[0]['conp_r']:.3f})  "
                    f"frame F1 {r[1]:.3f}  |  side outputs: attack {aa[0]:.2f}  release {aa[1]:.2f}  slide {aa[2]:.2f}  sung {aa[3]:.2f}")
            if step % 50 == 0 and (ckdir / "HANDOVER").exists():
                save(last); say(f"handing over at step {step}: saved last.pt for the faster GPU"); return
            if (time.time() - t0) / 60 > a.max_minutes:
                save(last); say("time limit: saved last.pt, resubmit to resume"); return
            if step >= a.total_steps:
                break
    save(last); say(f"stopping at step {step} (finished); best note F1 {best:.3f}")


if __name__ == "__main__":
    main()
