"""v1 with style codes, trained winner-takes-all: commit to ONE flavour, keep it, vary it across codes.

The user's scoring.  Per-frame loss against "every arrangement" teaches the
average of all arrangers' styles — the consensus survives, every style's own
choices fall under the threshold.  Instead the model is asked for the same
song K times, once per style code, and only the version CLOSEST to a human
arrangement is rewarded (multiple-choice learning):

    L(unit) = sum over the unit's human arrangements r of
              (1 - eps) * min_k L(k, r)  +  eps * mean_k L(k, r)

  commit   a blurred average matches no one arrangement well, so it loses to
           a version that commits to one style
  vary     two codes giving the same thing cannot both win; codes spread out
  stick    the winner is chosen over the WHOLE unit (several windows of the
           song at once), so a code cannot switch flavour mid-song
  eps      a small share for the losers so no code goes dead

A unit = one song audio at one difficulty (eval/m2m_bysong.py), with 1-7
human arrangements.  Split by SONG (no audio on both sides).

Model: frozen v1 + LoRA (r=8) on every block; the style code is added to the
tokens before the last --style-blocks blocks, so the first blocks run once
and only the tail runs K times.  Output stays v1's: onset/frame per key —
a selection of what it hears, not a writer.

Validation, per unit over its windows, against each human arrangement:
  v1          frozen v1 alone
  best-of-K   each arrangement's closest code (what the loss rewards)
  mean-of-K   any code, on average
  diversity   note F1 between codes' outputs (lower = more distinct)
  wins        how often each code is the closest (all codes should win some)

    python -m app.ml.spectral.style_train --clips /scratch/.../m2m/clips --units m2m_units.json \\
        --init-v1 <s2p_v1/best.pt> --ckpt-dir <dir>
"""
from __future__ import annotations

import argparse
import collections
import itertools
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
from .model import from_checkpoint, PATCH_W, F_PAD
from .train import log_to, note_f1

RMAX = 4                     # arrangements per unit used at once (units have 1-7)


def index_clips(root: pathlib.Path):
    """arrangement id -> {window: path}, across the old split folders."""
    idx = collections.defaultdict(dict)
    for p in root.glob("*/*.npz"):
        if p.name.endswith(".tmp.npz"):
            continue
        sid, w = p.stem.rsplit("_", 1)
        idx[sid][int(w)] = p
    return idx


class Units(Dataset):
    """One item = one unit: M aligned crops of the song + every arrangement's labels for them."""

    def __init__(self, units, idx, diff, crops=2, crop_s=8.0, train=True, max_windows=6):
        self.units = [u for u in units if all(i in idx for i in u)]
        self.idx, self.diff, self.M, self.train = idx, diff, crops, train
        self.n = int(crop_s * S.SR); self.nT = S.n_frames(self.n); self.maxw = max_windows

    def __len__(self):
        return len(self.units)

    def windows(self, u):
        return sorted(set.intersection(*(set(self.idx[i]) for i in u)))

    def __getitem__(self, j):
        u = self.units[j]
        refs = random.sample(u, min(RMAX, len(u))) if self.train else u[:RMAX]
        ws = self.windows(u)
        if self.train:
            pick = [random.choice(ws) for _ in range(self.M)]
        else:
            pick = ws[:: max(1, len(ws) // self.maxw)][: self.maxw]
        Y, ON, FR = [], [], []
        for w in pick:
            z0 = np.load(self.idx[refs[0]][w])
            y = z0["y"].astype(np.float32) / 32767
            T = z0["on"].shape[1]
            if self.train and T > self.nT:
                f0 = random.randrange(0, T - self.nT); a = int(round(f0 / S.FPS * S.SR))
                y = np.pad(y[a:a + self.n], (0, max(0, self.n - len(y[a:a + self.n]))))
            else:
                f0, a = 0, 0
            on = np.zeros((RMAX, 88, self.nT if self.train else T), np.float32); fr = on.copy()
            for r, i in enumerate(refs):
                z = z0 if r == 0 else np.load(self.idx[i][w])
                L = on.shape[2]
                on[r] = z["on"][:, f0:f0 + L]; fr[r] = z["fr"][:, f0:f0 + L]
            Y.append(y); ON.append(on); FR.append(fr)
        mask = np.zeros(RMAX, np.float32); mask[:len(refs)] = 1
        return (torch.from_numpy(np.stack(Y)), torch.from_numpy(np.stack(ON)), torch.from_numpy(np.stack(FR)),
                torch.from_numpy(mask), torch.tensor(self.diff[refs[0]]))


def collate(batch):
    return batch                                   # units differ in shape at validation; train stacks below


class StyleV1(nn.Module):
    def __init__(self, v1, K, style_blocks=2, lora_r=8, init_std=0.5):
        super().__init__()
        add_lora(v1, lora_r)
        for p in v1.head.parameters():
            p.requires_grad_(True)
        self.v1, self.K, self.sb = v1, K, style_blocks
        d = v1.arch["d"]
        self.style = nn.Parameter(torch.randn(K, d) * init_std)  # distinct from the start: breaks the tie

    def forward(self, x, cond, codes=None):
        """x (B,7,264,T) -> onset, frame logits (Kc, B, T, 88) for codes (default: all K)."""
        m = self.v1
        B, _, Fb, T = x.shape
        s = m.stem(x)
        Tp = math.ceil(T / PATCH_W) * PATCH_W
        z = m.patch(F.pad(s, (0, Tp - T, 0, F_PAD - Fb))).permute(0, 2, 3, 1) + m.pitch_emb
        if m.cond_emb is not None and cond is not None:
            z = z + m.cond_emb(cond.long())[:, None, None, :]
        for blk in m.blocks[:-self.sb]:
            z = blk(z)
        codes = range(self.K) if codes is None else codes
        ons, frs = [], []
        for k in codes:
            h = z + self.style[k]
            for blk in m.blocks[-self.sb:]:
                h = blk(h)
            h = m.norm(h).permute(0, 3, 1, 2)
            u = m.unpatch(h)[:, :, :Fb, :T]
            y = m.head(m.fuse(torch.cat([u, s], dim=1))).permute(1, 0, 3, 2)
            ons.append(y[0]); frs.append(y[1])
        return torch.stack(ons), torch.stack(frs)

    def one(self, x, cond, k: int):
        """App inference: onset, frame, velocity logits (B, T, 88) for one style code."""
        m = self.v1
        B, _, Fb, T = x.shape
        s = m.stem(x)
        Tp = math.ceil(T / PATCH_W) * PATCH_W
        z = m.patch(F.pad(s, (0, Tp - T, 0, F_PAD - Fb))).permute(0, 2, 3, 1) + m.pitch_emb
        if m.cond_emb is not None and cond is not None:
            z = z + m.cond_emb(cond.long())[:, None, None, :]
        for blk in m.blocks[:-self.sb]:
            z = blk(z)
        h = z + self.style[k]
        for blk in m.blocks[-self.sb:]:
            h = blk(h)
        h = m.norm(h).permute(0, 3, 1, 2)
        u = m.unpatch(h)[:, :, :Fb, :T]
        y = m.head(m.fuse(torch.cat([u, s], dim=1))).permute(1, 0, 3, 2)
        return y[0], y[1], y[2]


def load_for_app(ck: dict, code: int):
    """A style checkpoint as a v1-shaped module: forward(x, cond) -> (onset, frame, velocity)."""
    from .model import SpectralTranscriber
    net = StyleV1(SpectralTranscriber(**ck["v1_arch"]), ck["K"], ck["style_blocks"], ck["lora"])
    net.load_state_dict(ck["model"])

    class _One(nn.Module):
        """`code` is set per SONG by the caller (one flavour for the whole song)."""
        def __init__(self):
            super().__init__(); self.net = net; self.code = code; self.K = ck["K"]

        def forward(self, x, cond=None):
            return self.net.one(x, cond, self.code)
    return _One()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True); ap.add_argument("--units", required=True)
    ap.add_argument("--init-v1", required=True); ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--K", type=int, default=4); ap.add_argument("--style-blocks", type=int, default=2)
    ap.add_argument("--lora", type=int, default=8); ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--units-per-batch", type=int, default=4); ap.add_argument("--crops", type=int, default=2)
    ap.add_argument("--crop", type=float, default=8.0)
    ap.add_argument("--style-std", type=float, default=0.5)
    ap.add_argument("--style-lr", type=float, default=1e-3, help="style codes learn faster than the adapters")
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--total-steps", type=int, default=8000); ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--val-units", type=int, default=119); ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-minutes", type=float, default=460)
    ap.add_argument("--eval-only", default="", help="score this checkpoint on --eval-split and exit")
    ap.add_argument("--eval-split", default="validation")
    ap.add_argument("--limit-units", type=int, default=0, help="smoke runs only")
    a = ap.parse_args()
    ckdir = pathlib.Path(os.path.expanduser(a.ckpt_dir)); ckdir.mkdir(parents=True, exist_ok=True)
    say = log_to(ckdir / ("eval.log" if a.eval_only else "train.log")); t0 = time.time()
    dev = "cuda"
    bf16 = torch.cuda.is_bf16_supported(); amp = torch.bfloat16 if bf16 else torch.float16

    ck1 = torch.load(os.path.expanduser(a.init_v1), map_location=dev, weights_only=False)
    fe = S.SpectralFrontend().to(dev).eval(); fe.mu.copy_(ck1["fe_mu"]); fe.sd.copy_(ck1["fe_sd"])
    net = StyleV1(from_checkpoint(ck1), a.K, a.style_blocks, a.lora, a.style_std).to(dev)
    v1 = from_checkpoint(ck1).to(dev).eval()
    for p in v1.parameters():
        p.requires_grad_(False)
    nt = sum(p.numel() for p in net.parameters() if p.requires_grad)
    say(f"StyleV1: K={a.K} codes on the last {a.style_blocks} blocks, LoRA r={a.lora}; trainable {nt / 1e3:.0f}k params")

    U = json.loads(pathlib.Path(a.units).read_text())
    diff = {i: {"beginner": 0, "intermediate": 1, "advanced": 2}[v] for i, v in U["difficulty"].items()}
    idx = index_clips(pathlib.Path(a.clips))
    tr_units = U["units"]["train"][: a.limit_units or None]
    ev_units = U["units"][a.eval_split][: (a.limit_units or a.val_units)]
    tr = Units(tr_units, idx, diff, a.crops, a.crop, train=True)
    ev = Units(ev_units, idx, diff, train=False)
    say(f"units: {len(tr)} train ({sum(len(u) > 1 for u in tr.units)} multi-arrangement), "
        f"{len(ev)} {a.eval_split} ({sum(len(u) > 1 for u in ev.units)} multi)")

    def evaluate():
        net.eval()
        res = collections.defaultdict(list); wins = np.zeros(a.K); cover = []
        with torch.no_grad():
            for (Y, ON, FR, mask, d), in DataLoader(ev, batch_size=1, collate_fn=collate, num_workers=a.workers):
                nr = int(mask.sum()); dd = d.view(1).repeat(len(Y)).to(dev)
                x = fe(Y.to(dev))
                with torch.autocast("cuda", dtype=amp):
                    po, pf = net(x, dd)
                    vo, vf = v1(x, dd)[:2]
                po, pf = torch.sigmoid(po.float()).cpu().numpy(), torch.sigmoid(pf.float()).cpu().numpy()
                vo, vf = torch.sigmoid(vo.float()).cpu().numpy(), torch.sigmoid(vf.float()).cpu().numpy()
                go, gf = ON.transpose(2, 3).numpy(), FR.transpose(2, 3).numpy()        # (W, R, T, 88)
                F1 = np.array([[note_f1(po[k], pf[k], go[:, r], gf[:, r])[0]["conp_f1"] for r in range(nr)]
                               for k in range(a.K)])                                   # (K, R)
                res["v1"] += [note_f1(vo, vf, go[:, r], gf[:, r])[0]["conp_f1"] for r in range(nr)]
                res["best-of-K"] += list(F1.max(0)); res["mean-of-K"] += list(F1.mean(0))
                for r in range(nr):
                    wins[F1[:, r].argmax()] += 1
                if nr > 1:
                    cover.append(len(set(F1.argmax(0))) / nr)
                res["diversity"] += [note_f1(po[i], pf[i], po[j] > .5, pf[j] > .5)[0]["conp_f1"]
                                     for i, j in itertools.combinations(range(a.K), 2)]
        net.train()
        out = {k: float(np.mean(v)) for k, v in res.items()}
        out["wins"] = (wins / max(1, wins.sum())).round(2).tolist()
        out["coverage"] = float(np.mean(cover)) if cover else float("nan")
        return out

    def show(step, r):
        return (f"{a.eval_split.upper()} step {step}: v1 {r['v1']:.3f}  best-of-K {r['best-of-K']:.3f}  "
                f"mean-of-K {r['mean-of-K']:.3f}  code-vs-code F1 {r['diversity']:.3f}  wins {r['wins']}  "
                f"multi-arrangement coverage {r['coverage']:.2f}")

    if a.eval_only:
        net.load_state_dict(torch.load(os.path.expanduser(a.eval_only), map_location=dev, weights_only=False)["model"])
        say(show("final", evaluate())); return

    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": [p for p in params if p is not net.style], "lr": a.lr},
                             {"params": [net.style], "lr": a.style_lr, "weight_decay": 0.0}],
                            lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / a.warmup, 0.5 * (
        1 + math.cos(math.pi * min(1.0, max(0, s - a.warmup) / max(1, a.total_steps - a.warmup))))))
    scaler = torch.amp.GradScaler("cuda", enabled=not bf16)
    step, best = 0, -1.0
    last = ckdir / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=dev, weights_only=False)
        net.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        step, best = ck["step"], ck["best"]; say(f"resumed at step {step}")

    def save(path):
        tmp = path.with_suffix(".tmp")
        torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "step": step, "best": best, "fe_mu": fe.mu.cpu(), "fe_sd": fe.sd.cpu(),
                    "v1_arch": ck1["arch"], "K": a.K, "style_blocks": a.style_blocks, "lora": a.lora,
                    "args": vars(a)}, tmp)
        tmp.replace(path)

    pw_on, pw_fr = torch.tensor(30.0, device=dev), torch.tensor(2.0, device=dev)
    dl = DataLoader(tr, batch_size=a.units_per_batch, shuffle=True, drop_last=True, collate_fn=collate,
                    num_workers=a.workers, persistent_workers=a.workers > 0)
    acc, wins_run, n, t_log = np.zeros(2), np.zeros(a.K), 0, time.time()
    net.train()
    while step < a.total_steps:
        for batch in dl:
            Y = torch.cat([b[0] for b in batch]).to(dev, non_blocking=True)          # (U*M, n)
            ON = torch.cat([b[1] for b in batch]).to(dev).transpose(2, 3)            # (U*M, R, T, 88)
            FR = torch.cat([b[2] for b in batch]).to(dev).transpose(2, 3)
            mask = torch.stack([b[3] for b in batch]).to(dev)                         # (U, R)
            d = torch.cat([b[4].view(1).repeat(len(b[0])) for b in batch]).to(dev)
            Uu, M = len(batch), a.crops
            with torch.no_grad():
                x = fe(Y)
            with torch.autocast("cuda", dtype=amp):
                po, pf = net(x, d)                                                    # (K, U*M, T, 88)
            po, pf = po.float()[:, :, None], pf.float()[:, :, None]                   # (K, U*M, 1, T, 88)
            bce = lambda z, y, pw: F.binary_cross_entropy_with_logits(
                z.expand(-1, -1, RMAX, -1, -1), y[None].expand(z.shape[0], -1, -1, -1, -1),
                pos_weight=pw, reduction="none").mean((-1, -2))                         # (K, U*M, R)
            L = bce(po, ON, pw_on) + bce(pf, FR, pw_fr)
            L = L.view(a.K, Uu, M, RMAX).sum(2)                                       # whole unit: (K, U, R)
            Lmin, kmin = L.min(0)
            per = (1 - a.eps) * Lmin + a.eps * L.mean(0)                              # (U, R)
            loss = (per * mask).sum() / mask.sum() / M
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update(); sched.step(); step += 1
            km = kmin[mask > 0].cpu().numpy(); wins_run += np.bincount(km, minlength=a.K)
            acc += [float(loss), float((Lmin * mask).sum() / mask.sum() / M)]; n += 1
            if step % 100 == 0:
                m = acc / n
                say(f"step {step:>6}  loss {m[0]:.4f}  winner loss {m[1]:.4f}  "
                    f"wins {(wins_run / wins_run.sum()).round(2).tolist()}  {100 / (time.time() - t_log):.2f} steps/s")
                acc, wins_run, n, t_log = np.zeros(2), np.zeros(a.K), 0, time.time()
            if step % a.val_every == 0 or step >= a.total_steps:
                r = evaluate()
                if r["best-of-K"] > best:
                    best = r["best-of-K"]; save(ckdir / "best.pt")
                save(last); say(show(step, r))
            if (time.time() - t0) / 60 > a.max_minutes:
                save(last); say("time limit: saved last.pt, resubmit to resume"); return
            if step >= a.total_steps:
                break
    say(f"done, best best-of-K F1 {best:.3f}")


if __name__ == "__main__":
    main()
