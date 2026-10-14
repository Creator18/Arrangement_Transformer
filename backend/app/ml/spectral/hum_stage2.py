"""Stage 2 of the hum plan: SONG hum -> imagined PIANO hum, read by the frozen stage-1 model.

Both sides are hums (eval/make_pair_hums.py), so the difference between them
is arranging, not timbre.

Model
  trunk    a copy of stage 1 (hum_s1_scratch: trained from scratch on piano
           hums), every layer fine-tuned (--lora 0, default; --lora r instead
           freezes it and adds adapters), K style codes (K=1: the overall
           version, no styles) added before the last --style-blocks blocks
  edit     x_hat = x_song_hum + edit(h), edit zero-initialised: at step 0 the
           song hum passes through unchanged — everything learned is what to
           keep, drop or add
  kda      per key and frame: kept (A&B), dropped (A-B), added (B-A), where
           A = what the stage-1 reader hears in the SONG hum and B = the human
           arrangement.  Targets only, never inputs
  reader   a second frozen copy of stage 1 turns x_hat into notes (onset
           threshold --thr, 0.8 = stage 1's best)

Loss per style code k and human arrangement r of a unit (one song at one level):
  w_feat * L1(x_hat_k, x_piano_hum_r) + BCE(reader(x_hat_k), arrangement r) + w_kda * BCE(kda_k, remainders_r)
summed over the unit's crops, then winner-takes-all over codes per arrangement
(eps share to the losers), as in style_train.py.

Validation per unit (all its windows), note F1 at --thr against each arrangement:
  song hum as-is   the reader on the unedited song hum (the starting point)
  best-of-K / mean-of-K
  piano hum        the reader on the REAL piano hum (the ceiling of this design)

    python -m app.ml.spectral.hum_stage2 --hums /scratch/.../pair_hums --units m2m_units.json \\
        --stage1 <hum_s1/best.pt> --ckpt-dir <dir>
"""
from __future__ import annotations

import argparse
import collections
import json
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
from .imagine import add_lora
from .model import SpectralTranscriber, PATCH_W, F_PAD
from .train import log_to

RMAX = 4


def load_stage1(ck):
    """Stage-1 model from its checkpoint: from scratch (hum_s1_scratch) or the older LoRA version."""
    net = SpectralTranscriber(**ck["arch"])
    if ck.get("lora"):
        add_lora(net, ck["lora"])
    net.load_state_dict(ck["model"])
    return net


def index(root: pathlib.Path):
    """arrangement id -> {window: path} for one side (song/ or piano/), across split folders."""
    idx = collections.defaultdict(dict)
    for p in root.glob("*/*.npz"):
        if not p.name.endswith(".tmp.npz"):
            sid, w = p.stem.rsplit("_", 1); idx[sid][int(w)] = p
    return idx


class Units(Dataset):
    def __init__(self, units, song, piano, diff, crops=1, crop_s=8.0, train=True, max_windows=6):
        self.units = [u for u in units if all(i in song and i in piano for i in u)]
        self.song, self.piano, self.diff = song, piano, diff
        self.M, self.train, self.maxw = crops, train, max_windows
        self.n = int(crop_s * S.SR); self.nT = S.n_frames(self.n)

    def __len__(self):
        return len(self.units)

    def __getitem__(self, j):
        u = self.units[j]
        refs = random.sample(u, min(RMAX, len(u))) if self.train else u[:RMAX]
        ws = sorted(set.intersection(*(set(self.song[i]) & set(self.piano[i]) for i in refs)))
        pick = [random.choice(ws) for _ in range(self.M)] if self.train else ws[:: max(1, len(ws) // self.maxw)][: self.maxw]
        YS, YP, ON, FR = [], [], [], []
        for w in pick:
            ys = np.load(self.song[refs[0]][w])["y"].astype(np.float32) / 32767
            n = len(ys); T = S.n_frames(n)
            if self.train and n > self.n:
                f0 = random.randrange(0, S.n_frames(n) - self.nT); a = int(round(f0 / S.FPS * S.SR)); L = self.nT
                ys = ys[a:a + self.n]
            else:
                f0, a, L = 0, 0, T
            nn_ = len(ys)
            yp = np.zeros((RMAX, nn_), np.float32); on = np.zeros((RMAX, 88, L), np.float32); fr = on.copy()
            for r, i in enumerate(refs):
                z = np.load(self.piano[i][w])
                yp[r] = (z["y"].astype(np.float32) / 32767)[a:a + nn_]
                o, f = z["on"][:, f0:f0 + L], z["fr"][:, f0:f0 + L]
                on[r, :, :o.shape[1]] = o; fr[r, :, :f.shape[1]] = f
            YS.append(ys); YP.append(yp); ON.append(on); FR.append(fr)
        mask = np.zeros(RMAX, np.float32); mask[:len(refs)] = 1
        return (torch.from_numpy(np.stack(YS)), torch.from_numpy(np.stack(YP)), torch.from_numpy(np.stack(ON)),
                torch.from_numpy(np.stack(FR)), torch.from_numpy(mask), torch.tensor(self.diff[refs[0]]))


class Stage2(nn.Module):
    def __init__(self, trunk, K=4, style_blocks=2, lora_r=8, style_std=0.5):
        super().__init__()
        if lora_r:
            add_lora(trunk, lora_r)                               # frozen stage 1 + new adapters
        else:
            for p in trunk.parameters():                          # full fine-tune: every layer learns
                p.requires_grad_(True)
        self.t, self.K, self.sb = trunk, K, style_blocks
        d, c = trunk.arch["d"], trunk.arch.get("stem", 32)
        self.style = nn.Parameter(torch.randn(K, d) * style_std)
        self.edit = nn.Conv2d(c, len(S.HARMONICS) + 1, 3, padding=1)
        nn.init.zeros_(self.edit.weight); nn.init.zeros_(self.edit.bias)
        self.kda = nn.Conv2d(c, 3, (3, 1), stride=(3, 1))

    def forward(self, x, codes=None):
        """x (B,7,264,T) song-hum features -> x_hat (Kc,B,7,264,T), kda logits (Kc,3,B,T,88)."""
        m = self.t
        B, _, Fb, T = x.shape
        s = m.stem(x)
        Tp = math.ceil(T / PATCH_W) * PATCH_W
        z = m.patch(F.pad(s, (0, Tp - T, 0, F_PAD - Fb))).permute(0, 2, 3, 1) + m.pitch_emb
        for blk in m.blocks[:-self.sb]:
            z = blk(z)
        xs, ks = [], []
        for k in (range(self.K) if codes is None else codes):
            h = z + self.style[k]
            for blk in m.blocks[-self.sb:]:
                h = blk(h)
            h = m.norm(h).permute(0, 3, 1, 2)
            u = m.unpatch(h)[:, :, :Fb, :T]
            g = m.fuse(torch.cat([u, s], dim=1))
            xs.append(x + self.edit(g)); ks.append(self.kda(g).permute(1, 0, 3, 2))
        return torch.stack(xs), torch.stack(ks)


def f1_at(po, pf, go, gf, thr):
    """Mean note F1 (50 ms, pitch) over (N, T, 88) rolls, prediction decoded at onset threshold thr."""
    from app.ml.transcriber.decode import decode_notes, note_scores
    full = dict(min_frames=1, min_note_s=0.0, keep_restruck=True)
    s = []
    for i in range(len(po)):
        est = decode_notes(po[i].T, pf[i].T, np.zeros_like(po[i].T), thr, 0.5, **full)
        ref = decode_notes(go[i].T, gf[i].T, np.zeros_like(go[i].T), 0.5, 0.5, **full)
        if ref:
            s.append(note_scores(ref, est)["conp_f1"])
    return float(np.mean(s)) if s else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hums", required=True); ap.add_argument("--units", required=True)
    ap.add_argument("--stage1", required=True); ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--K", type=int, default=4); ap.add_argument("--style-blocks", type=int, default=2)
    ap.add_argument("--lora", type=int, default=0, help="0 = full fine-tune of the stage-1 trunk"); ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--thr", type=float, default=0.8)
    ap.add_argument("--w-feat", type=float, default=0.1); ap.add_argument("--w-kda", type=float, default=0.5)
    ap.add_argument("--units-per-batch", type=int, default=4); ap.add_argument("--crops", type=int, default=1)
    ap.add_argument("--crop", type=float, default=8.0)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--style-lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=300); ap.add_argument("--total-steps", type=int, default=8000)
    ap.add_argument("--val-every", type=int, default=500); ap.add_argument("--val-units", type=int, default=119)
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--max-minutes", type=float, default=460)
    ap.add_argument("--limit-units", type=int, default=0)
    a = ap.parse_args()
    ckdir = pathlib.Path(os.path.expanduser(a.ckpt_dir)); ckdir.mkdir(parents=True, exist_ok=True)
    say = log_to(ckdir / "train.log"); t0 = time.time(); dev = "cuda"
    bf16 = torch.cuda.is_bf16_supported(); amp = torch.bfloat16 if bf16 else torch.float16

    ck1 = torch.load(os.path.expanduser(a.stage1), map_location=dev, weights_only=False)
    fe = S.SpectralFrontend().to(dev).eval(); fe.mu.copy_(ck1["fe_mu"]); fe.sd.copy_(ck1["fe_sd"])
    net = Stage2(load_stage1(ck1), a.K, a.style_blocks, a.lora).to(dev)
    reader = load_stage1(ck1).to(dev).eval()
    for p in reader.parameters():
        p.requires_grad_(False)
    params = [p for p in net.parameters() if p.requires_grad]
    say(f"stage 2: stage-1 trunk {'+ LoRA r=' + str(a.lora) if a.lora else 'fully fine-tuned'}, K={a.K} codes on last {a.style_blocks} blocks, "
        f"reader thr {a.thr}; trainable {sum(p.numel() for p in params) / 1e3:.0f}k")

    U = json.loads(pathlib.Path(a.units).read_text())
    diff = {i: {"beginner": 0, "intermediate": 1, "advanced": 2}[v] for i, v in U["difficulty"].items()}
    song, piano = index(pathlib.Path(a.hums) / "song"), index(pathlib.Path(a.hums) / "piano")
    tr = Units(U["units"]["train"][: a.limit_units or None], song, piano, diff, a.crops, a.crop, True)
    ev = Units(U["units"]["validation"][: a.limit_units or a.val_units], song, piano, diff, train=False)
    say(f"units with hums: {len(tr)} train ({sum(len(u) > 1 for u in tr.units)} multi), {len(ev)} validation")

    def evaluate():
        net.eval(); R = collections.defaultdict(list); wins = np.zeros(a.K)
        with torch.no_grad():
            for (YS, YP, ON, FR, mask, d), in DataLoader(ev, batch_size=1, collate_fn=lambda b: b, num_workers=a.workers):
                nr = int(mask.sum()); xs = fe(YS.to(dev))
                go, gf = ON.transpose(2, 3).numpy(), FR.transpose(2, 3).numpy()          # (W, R, T, 88)
                with torch.autocast("cuda", dtype=amp):
                    xh, _ = net(xs)
                    outs = [reader(xh[k].float()) for k in range(a.K)] + [reader(xs)]
                    ps = [reader(fe(YP[:, r].to(dev))) for r in range(nr)]
                pr = lambda o: (torch.sigmoid(o[0].float()).cpu().numpy(), torch.sigmoid(o[1].float()).cpu().numpy())
                F1 = np.array([[f1_at(*pr(outs[k]), go[:, r], gf[:, r], a.thr) for r in range(nr)] for k in range(a.K)])
                R["song hum as-is"] += [f1_at(*pr(outs[-1]), go[:, r], gf[:, r], a.thr) for r in range(nr)]
                R["piano hum (ceiling)"] += [f1_at(*pr(ps[r]), go[:, r], gf[:, r], a.thr) for r in range(nr)]
                R["best-of-K"] += list(F1.max(0)); R["mean-of-K"] += list(F1.mean(0))
                for r in range(nr):
                    wins[F1[:, r].argmax()] += 1
        net.train()
        out = {k: float(np.nanmean(v)) for k, v in R.items()}
        out["wins"] = (wins / max(1, wins.sum())).round(2).tolist()
        return out

    opt = torch.optim.AdamW([{"params": [p for p in params if p is not net.style], "lr": a.lr},
                             {"params": [net.style], "lr": a.style_lr, "weight_decay": 0.0}], lr=a.lr, weight_decay=0.01)
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
        say("VAL step 0: " + "  ".join(f"{k} {v}" if k == "wins" else f"{k} {v:.3f}" for k, v in evaluate().items()))

    def save(path):
        tmp = path.with_suffix(".tmp")
        torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step,
                    "best": best, "stage1_arch": ck1["arch"], "stage1_lora": ck1.get("lora", 0), "K": a.K,
                    "style_blocks": a.style_blocks, "lora": a.lora, "thr": a.thr,
                    "fe_mu": fe.mu.cpu(), "fe_sd": fe.sd.cpu(), "args": vars(a)}, tmp)
        tmp.replace(path)

    pw_on, pw_fr = torch.tensor(30.0, device=dev), torch.tensor(2.0, device=dev)
    dl = DataLoader(tr, batch_size=a.units_per_batch, shuffle=True, drop_last=True, collate_fn=lambda b: b,
                    num_workers=a.workers, persistent_workers=a.workers > 0)
    acc, n, t_log, wins_run = np.zeros(4), 0, time.time(), np.zeros(a.K)
    net.train()
    while step < a.total_steps:
        for batch in dl:
            Uu, M = len(batch), a.crops
            YS = torch.cat([b[0] for b in batch]).to(dev)                           # (U*M, n)
            YP = torch.cat([b[1] for b in batch]).to(dev)                           # (U*M, R, n)
            ON = torch.cat([b[2] for b in batch]).to(dev).transpose(2, 3)           # (U*M, R, T, 88)
            FR = torch.cat([b[3] for b in batch]).to(dev).transpose(2, 3)
            mask = torch.stack([b[4] for b in batch]).to(dev)                        # (U, R)
            with torch.no_grad():
                xs = fe(YS)
                xp = torch.stack([fe(YP[:, r]) for r in range(RMAX)], 1)            # (U*M, R, 7, 264, T)
                with torch.autocast("cuda", dtype=amp):
                    A = torch.sigmoid(reader(xs)[1].float()) > 0.5                   # what the SONG hum holds
                Bf = FR > 0.5
                kda_t = torch.stack([A[:, None] & Bf, A[:, None] & ~Bf, Bf & ~A[:, None]], 2).float()  # (U*M,R,3,T,88)
            with torch.autocast("cuda", dtype=amp):
                xh, kz = net(xs)                                                     # (K,U*M,7,264,T), (K,3,U*M,T,88)
                ro = [reader(xh[k]) for k in range(a.K)]
            L = []
            for k in range(a.K):
                o0, o1 = ro[k][0].float()[:, None], ro[k][1].float()[:, None]       # (U*M,1,T,88)
                l_notes = (F.binary_cross_entropy_with_logits(o0.expand_as(ON), ON, pos_weight=pw_on, reduction="none")
                           + F.binary_cross_entropy_with_logits(o1.expand_as(FR), FR, pos_weight=pw_fr, reduction="none")
                           ).mean((-1, -2))                                          # (U*M, R)
                l_feat = (xh[k].float()[:, None] - xp).abs().mean((-1, -2, -3))      # (U*M, R)
                kk = kz[k].float().permute(1, 0, 2, 3)[:, None]                      # (U*M,1,3,T,88)
                l_kda = F.binary_cross_entropy_with_logits(kk.expand_as(kda_t), kda_t, pos_weight=pw_fr,
                                                           reduction="none").mean((-1, -2, -3))
                L.append(l_notes + a.w_feat * l_feat + a.w_kda * l_kda)
            L = torch.stack(L).view(a.K, Uu, M, RMAX).sum(2)                          # (K, U, R)
            Lmin, kmin = L.min(0)
            loss = (((1 - a.eps) * Lmin + a.eps * L.mean(0)) * mask).sum() / mask.sum() / M
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update(); sched.step(); step += 1
            wins_run += np.bincount(kmin[mask > 0].cpu().numpy(), minlength=a.K)
            acc += [float(loss), float((Lmin * mask).sum() / mask.sum() / M), 0, 0]; n += 1
            if step % 100 == 0:
                say(f"step {step:>6}  loss {acc[0] / n:.4f}  winner {acc[1] / n:.4f}  "
                    f"wins {(wins_run / wins_run.sum()).round(2).tolist()}  {100 / (time.time() - t_log):.2f} steps/s")
                acc, n, t_log, wins_run = np.zeros(4), 0, time.time(), np.zeros(a.K)
            if step % a.val_every == 0 or step >= a.total_steps:
                r = evaluate()
                if r["best-of-K"] > best:
                    best = r["best-of-K"]; save(ckdir / "best.pt")
                save(last)
                say(f"VAL step {step}: " + "  ".join(f"{k} {v}" if k == "wins" else f"{k} {v:.3f}" for k, v in r.items()))
            if (time.time() - t0) / 60 > a.max_minutes:
                save(last); say("time limit: saved last.pt, resubmit to resume"); return
            if step >= a.total_steps:
                break
    say(f"done, best best-of-K {best:.3f}")


if __name__ == "__main__":
    main()
