"""One piano grammar learnt from NOTES and SOUND together: both inputs, both losses, minimised jointly.

The user's next step (run regardless of how alike the separate grammars are).
Every token is one octave window x 64 ms, as in piano_grammar.py, and carries
BOTH that cell's notes and its sound (two embeddings, summed).  Each side can be
blanked per cell with its own [MASK]; the model fills in both.

Masking, per example:
  60%  the SAME cells blanked in notes and sound      -> must infer from the music around (grammar)
  20%  cells blanked on one side only                  -> learns how notes and sound relate
  10%  the whole notes side dropped (sound only in)    -> works from sound alone
  10%  the whole sound side dropped (notes only in)    -> works from notes alone
Loss on blanked cells only:  notes (weighted BCE: sounding keys 5x, onsets 20x) and
sound (L1), balanced by learned uncertainty weights  sum_i exp(-s_i) L_i + s_i,
so neither scale drowns the other.

    python -m app.ml.grammar.joint_grammar --clips /scratch/.../pop909_render --fe <spectral_real/best.pt> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.ml.spectral import frontend as S
from .piano_grammar import (Clips, tokens_notes, tokens_sound, notes_loss, sound_loss, span_mask,
                            BANDS, DN, DS, TOK, CROP, _Block)


class JointEncoder(nn.Module):
    def __init__(self, d=256, depth=6, heads=4):
        super().__init__()
        self.en = nn.Sequential(nn.Linear(DN, d), nn.GELU(), nn.Linear(d, d))
        self.es = nn.Sequential(nn.Linear(DS, d), nn.GELU(), nn.Linear(d, d))
        self.mn, self.ms = nn.Parameter(torch.zeros(d)), nn.Parameter(torch.zeros(d))
        self.band = nn.Parameter(torch.randn(1, BANDS, 1, d) * 0.02)
        self.blocks = nn.ModuleList(_Block(d, heads) for _ in range(depth)); self.norm = nn.LayerNorm(d)
        self.hn, self.hs = nn.Linear(d, DN), nn.Linear(d, DS)

    def forward(self, xn, xs, mn, ms):        # tokens (B,R,N,*), masks (B,R,N) bool -> notes, sound predictions
        hn = torch.where(mn[..., None], self.mn.to(xn.dtype), self.en(xn))
        hs = torch.where(ms[..., None], self.ms.to(xs.dtype), self.es(xs))
        h = hn + hs + self.band
        for b in self.blocks:
            h = b(h)
        h = self.norm(h)
        return self.hn(h), self.hs(h)


def joint_masks(B, R, N, dev):
    """Per example: same cells / one side / drop notes / drop sound (60/20/10/10)."""
    mn = span_mask(B, R, N, dev=dev); ms = mn.clone()
    for b in range(B):
        u = random.random()
        if u < 0.6:
            continue
        if u < 0.8:                                                    # one side only
            if random.random() < 0.5:
                ms[b] = False
            else:
                mn[b] = False
        elif u < 0.9:
            mn[b] = True                                               # notes dropped: sound only in
            ms[b] = False
        else:
            ms[b] = True                                               # sound dropped: notes only in
            mn[b] = False
    return mn, ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True); ap.add_argument("--fe", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=32); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=500); ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    from app.ml.spectral.train import log_to
    say = log_to(out / "train.log"); dev = "cuda"; torch.manual_seed(0); random.seed(0); np.random.seed(0)
    ck = torch.load(a.fe, map_location=dev, weights_only=False)
    fe = S.SpectralFrontend().to(dev).eval(); fe.mu.copy_(ck["fe_mu"]); fe.sd.copy_(ck["fe_sd"])
    clip = lambda s: sorted(p for p in (pathlib.Path(a.clips) / s).glob("pop*_[0-9][0-9][0-9].npz"))
    tr, va = clip("train")[: a.limit or None], clip("validation")[: 300 if not a.limit else 8]
    dl = DataLoader(Clips(tr), batch_size=a.batch, shuffle=True, drop_last=True, num_workers=a.workers,
                    persistent_workers=a.workers > 0)
    vdl = DataLoader(Clips(va, train=False), batch_size=a.batch, num_workers=a.workers)
    net = JointEncoder().to(dev); logvar = nn.Parameter(torch.zeros(2, device=dev))   # uncertainty weights
    opt = torch.optim.AdamW(list(net.parameters()) + [logvar], lr=a.lr, weight_decay=0.01)
    sc = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 300) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
    say(f"joint grammar: {len(tr)} train / {len(va)} val POP909 clips; {sum(p.numel() for p in net.parameters()) / 1e6:.1f}M params")
    amp = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp == torch.float16)

    def batches():
        while True:
            for y, roll in dl:
                yield y.to(dev, non_blocking=True), roll.to(dev, non_blocking=True)
    gen = batches(); curve = []
    for step in range(1, a.steps + 1):
        y, roll = next(gen)
        with torch.no_grad():
            tn, ts = tokens_notes(roll), tokens_sound(fe(y)[..., :CROP])
        mn, ms = joint_masks(*tn.shape[:3], dev)
        with torch.autocast("cuda", dtype=amp):
            pn, ps = net(tn, ts, mn, ms)
        ln, ls = notes_loss(pn.float(), tn, mn), sound_loss(ps.float(), ts, ms)
        loss = torch.exp(-logvar[0]) * ln + logvar[0] + torch.exp(-logvar[1]) * ls + logvar[1]
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(net.parameters(), 1.0); scaler.step(opt); scaler.update(); sc.step()
        if step % a.eval_every == 0:
            net.eval(); vn, vs = [], []
            with torch.no_grad():
                for yv, rv in vdl:
                    tn, ts = tokens_notes(rv.to(dev)), tokens_sound(fe(yv.to(dev))[..., :CROP])
                    np.random.seed(1); m = span_mask(*tn.shape[:3], dev=dev)            # same cells, both sides
                    pn, ps = net(tn, ts, m, m)
                    vn.append(float(notes_loss(pn.float(), tn, m))); vs.append(float(sound_loss(ps.float(), ts, m)))
            net.train(); curve.append((step, float(np.mean(vn)), float(np.mean(vs))))
            say(f"  step {step}: val fill-in (same cells blanked)  notes {np.mean(vn):.4f}  sound {np.mean(vs):.4f}  "
                f"weights notes {float(torch.exp(-logvar[0])):.2f} sound {float(torch.exp(-logvar[1])):.2f}")
    torch.save({"net": net.state_dict(), "logvar": logvar.detach().cpu()}, out / "joint_grammar.pt")
    (out / "joint_results.json").write_text(json.dumps({"curve": curve}, indent=1))
    say("done")


if __name__ == "__main__":
    main()
