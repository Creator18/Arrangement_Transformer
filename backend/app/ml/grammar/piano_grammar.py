"""Piano grammar three ways: from the NOTES, from the SOUND, and the two joined by cross-attention.

The user's experiment.  On rendered POP909 piano (eval/render_pop909.py, labels
exact; human chord/melody/key labels from eval/pop909_labels.py):

  notes grammar   a transformer reads the piano roll (onset + frame, 88 keys) in
                  64 ms tokens, some tokens masked out, and fills them back in
  sound grammar   the same transformer and the same masking on the SOUND of those
                  notes (the solo-piano front end's HCQT features), filling the
                  masked sound back in
  cross           each grammar's encoder + cross-attention layers translating one
                  into the other: notes from sound (all notes masked, attend to the
                  sound) and sound from notes.  Three variants on the same budget:
                    pretrained  both grammars' encoders, frozen
                    random      encoders of the same size, random and frozen (control)
                    scratch     everything trained from scratch
                  If the grammars overlap, "pretrained" translates fast and well.

Then probes (eval/piano_grammar_probe.py) read chords, melody, density and
low-register muddiness out of each frozen encoder, and CKA measures how alike
the two encoders' internal pictures are.

    python -m app.ml.grammar.piano_grammar --clips /scratch/.../pop909_render --fe <spectral_real/best.pt> --out <dir>
"""
from __future__ import annotations

import argparse
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

from app.ml.spectral import frontend as S

TOK = 4                 # frames per token: 64 ms
CROP = 500              # frames per training example: 8 s -> 125 tokens


class Clips(Dataset):
    def __init__(self, paths, train=True):
        self.paths, self.train = paths, train

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        z = np.load(self.paths[i])
        T = z["on"].shape[1]
        f0 = random.randrange(0, T - CROP) if self.train else 0
        f0 -= f0 % TOK
        a = int(round(f0 / S.FPS * S.SR)); n = int(round(CROP / S.FPS * S.SR))
        y = z["y"].astype(np.float32)[a:a + n] / 32767
        y = np.pad(y, (0, n - len(y)))
        roll = np.stack([z["on"][:, f0:f0 + CROP], z["fr"][:, f0:f0 + CROP]]).astype(np.float32)
        return torch.from_numpy(y), torch.from_numpy(roll)


def tokens_notes(roll):                      # (B, 2, 88, T) -> (B, N, 2*88*TOK)
    B, C, K, T = roll.shape
    return roll.reshape(B, C, K, T // TOK, TOK).permute(0, 3, 1, 2, 4).reshape(B, T // TOK, -1)


def tokens_sound(feat):                      # (B, 7, 264, T) -> (B, N, 7*264*TOK)
    B, C, K, T = feat.shape
    T = T // TOK * TOK
    return feat[..., :T].reshape(B, C, K, T // TOK, TOK).permute(0, 3, 1, 2, 4).reshape(B, T // TOK, -1)


class Encoder(nn.Module):
    """Token embedding + transformer; masked tokens are replaced by a learned [MASK]."""

    def __init__(self, din, d=256, depth=6, heads=4, n=CROP // TOK):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(din, d), nn.GELU(), nn.Linear(d, d))
        self.mask = nn.Parameter(torch.zeros(d)); self.pos = nn.Parameter(torch.randn(1, n, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, depth); self.norm = nn.LayerNorm(d)

    def forward(self, x, m=None):            # x (B, N, din), m (B, N) bool = masked
        h = self.inp(x)
        if m is not None:
            h = torch.where(m[..., None], self.mask.to(h.dtype), h)
        return self.norm(self.tf(h + self.pos[:, : h.shape[1]]))


class Cross(nn.Module):
    """Decoder layers whose queries are one modality (fully masked) and keys/values the other's encoding."""

    def __init__(self, d=256, depth=3, heads=4, dout=1, n=CROP // TOK):
        super().__init__()
        self.q = nn.Parameter(torch.zeros(1, n, d)); nn.init.normal_(self.q, std=0.02)
        layer = nn.TransformerDecoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(layer, depth); self.head = nn.Linear(d, dout)

    def forward(self, memory):
        return self.head(self.dec(self.q.expand(memory.shape[0], -1, -1), memory))


def span_mask(B, N, frac=0.3, span=4, dev="cuda"):
    m = torch.zeros(B, N, dtype=torch.bool, device=dev)
    k = max(1, int(N * frac / span))
    for b in range(B):
        for s in random.sample(range(0, N - span), k):
            m[b, s:s + span] = True
    return m


def notes_loss(pred, roll_tok, m=None):      # BCE on the roll tokens (onsets weighted, they are rare)
    w = torch.ones_like(roll_tok); w[..., : 88 * TOK] = 10.0
    l = F.binary_cross_entropy_with_logits(pred, roll_tok, weight=w, reduction="none").mean(-1)
    return (l * m).sum() / m.sum().clamp(min=1) if m is not None else l.mean()


def sound_loss(pred, feat_tok, m=None):
    l = (pred - feat_tok).abs().mean(-1)
    return (l * m).sum() / m.sum().clamp(min=1) if m is not None else l.mean()


def frame_f1(pred_logit, roll_tok):
    p = torch.sigmoid(pred_logit[..., 88 * TOK:]) > 0.5; g = roll_tok[..., 88 * TOK:] > 0.5
    tp = (p & g).sum().item(); return 2 * tp / max(1, p.sum().item() + g.sum().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True); ap.add_argument("--fe", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--grammar-steps", type=int, default=8000); ap.add_argument("--cross-steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=32); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=500); ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    from app.ml.spectral.train import log_to
    say = log_to(out / "train.log"); dev = "cuda"; torch.manual_seed(0); random.seed(0)
    ck = torch.load(a.fe, map_location=dev, weights_only=False)
    fe = S.SpectralFrontend().to(dev).eval(); fe.mu.copy_(ck["fe_mu"]); fe.sd.copy_(ck["fe_sd"])
    clip = lambda s: sorted(p for p in (pathlib.Path(a.clips) / s).glob("pop*_[0-9][0-9][0-9].npz"))
    tr, va = clip("train")[: a.limit or None], clip("validation")[: 300 if not a.limit else 8]
    say(f"piano grammar: {len(tr)} train / {len(va)} validation POP909 clips (8 s crops, 64 ms tokens)")
    dl = DataLoader(Clips(tr), batch_size=a.batch, shuffle=True, drop_last=True, num_workers=a.workers,
                    persistent_workers=a.workers > 0)
    vdl = DataLoader(Clips(va, train=False), batch_size=a.batch, num_workers=a.workers)
    DN, DS = 2 * 88 * TOK, (len(S.HARMONICS) + 1) * S.N_BINS * TOK

    def batches():
        while True:
            for y, roll in dl:
                yield y.to(dev, non_blocking=True), roll.to(dev, non_blocking=True)

    gen = batches()

    @torch.no_grad()
    def prep(y, roll):
        return tokens_notes(roll), tokens_sound(fe(y)[..., :CROP])

    def sched(opt, steps):
        return torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 300) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))

    results = json.loads((out / "results.json").read_text()) if (out / "results.json").exists() else {}

    def save_results():
        (out / "results.json").write_text(json.dumps(results, indent=1))

    # ---- 1 & 2: the two grammars (masked filling-in) ----
    for kind in ("notes", "sound"):
        path = out / f"{kind}_grammar.pt"
        if path.exists():
            say(f"{kind} grammar: already trained"); continue
        enc = Encoder(DN if kind == "notes" else DS).to(dev)
        head = nn.Linear(256, DN if kind == "notes" else DS).to(dev)
        opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=a.lr, weight_decay=0.01)
        sc = sched(opt, a.grammar_steps); curve = []
        for step in range(1, a.grammar_steps + 1):
            tn, ts = prep(*next(gen)); x = tn if kind == "notes" else ts
            m = span_mask(*x.shape[:2], dev=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                pred = head(enc(x, m))
            loss = notes_loss(pred.float(), x, m) if kind == "notes" else sound_loss(pred.float(), x, m)
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
            opt.step(); sc.step()
            if step % a.eval_every == 0:
                enc.eval(); vl, vf = [], []
                with torch.no_grad():
                    for y, roll in vdl:
                        tn, ts = prep(y.to(dev), roll.to(dev)); x = tn if kind == "notes" else ts
                        g = torch.Generator(device="cpu").manual_seed(1); random.seed(1)
                        m = span_mask(*x.shape[:2], dev=dev); p = head(enc(x, m)).float()
                        vl.append(float(notes_loss(p, x, m) if kind == "notes" else sound_loss(p, x, m)))
                enc.train(); curve.append((step, float(np.mean(vl))))
                say(f"  {kind} grammar step {step}: masked fill-in loss (val) {np.mean(vl):.4f}")
        torch.save({"enc": enc.state_dict(), "head": head.state_dict(), "din": DN if kind == "notes" else DS}, path)
        results[f"{kind}_grammar_curve"] = curve; save_results()

    # ---- 3: cross-attention translation, three variants ----
    for variant in ("pretrained", "random", "scratch"):
        if f"cross_{variant}" in results:
            say(f"cross {variant}: already done"); continue
        en, es = Encoder(DN).to(dev), Encoder(DS).to(dev)
        if variant == "pretrained":
            en.load_state_dict(torch.load(out / "notes_grammar.pt", weights_only=False)["enc"])
            es.load_state_dict(torch.load(out / "sound_grammar.pt", weights_only=False)["enc"])
        frozen = variant in ("pretrained", "random")
        for p in list(en.parameters()) + list(es.parameters()):
            p.requires_grad_(not frozen)
        s2n, n2s = Cross(dout=DN).to(dev), Cross(dout=DS).to(dev)          # sound -> notes, notes -> sound
        params = [p for mdl in (en, es, s2n, n2s) for p in mdl.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01); sc = sched(opt, a.cross_steps); curve = []
        say(f"cross {variant}: encoders {'frozen' if frozen else 'trained'}; {sum(p.numel() for p in params) / 1e6:.1f}M trainable")
        for step in range(1, a.cross_steps + 1):
            tn, ts = prep(*next(gen))
            with torch.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                pn = s2n(es(ts)); ps = n2s(en(tn))
            loss = notes_loss(pn.float(), tn) + sound_loss(ps.float(), ts)
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sc.step()
            if step % a.eval_every == 0:
                for mdl in (en, es, s2n, n2s):
                    mdl.eval()
                f1s, sl = [], []
                with torch.no_grad():
                    for y, roll in vdl:
                        tn, ts = prep(y.to(dev), roll.to(dev))
                        f1s.append(frame_f1(s2n(es(ts)).float(), tn)); sl.append(float(sound_loss(n2s(en(tn)).float(), ts)))
                for mdl in (en, es, s2n, n2s):
                    mdl.train(not frozen if mdl in (en, es) else True)
                curve.append((step, float(np.mean(f1s)), float(np.mean(sl))))
                say(f"  cross {variant} step {step}: sound->notes frame F1 {np.mean(f1s):.3f}  notes->sound L1 {np.mean(sl):.4f}")
        results[f"cross_{variant}"] = curve; save_results()
    say("done: " + json.dumps({k: (v[-1] if isinstance(v, list) and v else v) for k, v in results.items()}))


if __name__ == "__main__":
    main()
