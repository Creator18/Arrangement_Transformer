"""Piano grammar three ways: from the NOTES, from the SOUND, and the two joined by cross-attention.

The user's experiment, in OVERLAPPING OCTAVE WINDOWS: every token is one octave
(12 keys / 36 pitch bins) x 64 ms, windows sliding by half an octave — 14 across
the keyboard, exactly the solo-piano transcriber's pitch patches — so notes,
sound and labels all sit on the same small bands and stay consistent.
On rendered POP909 piano (eval/render_pop909.py, labels
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


BANDS = 14              # octave windows: keys 6r .. 6r+11 (last one 76..87), bins 18r .. 18r+35
KEY0 = [min(6 * r, 76) for r in range(BANDS)]
DN = 2 * 12 * TOK       # notes token: onset + frame, 12 keys, 4 frames
DS = (len(S.HARMONICS) + 1) * 36 * TOK      # sound token: 7 channels, 36 bins, 4 frames


def tokens_notes(roll):                      # (B, 2, 88, T) -> (B, BANDS, N, DN)
    B, C, K, T = roll.shape
    t = torch.stack([roll[:, :, k:k + 12] for k in KEY0], 1)                    # (B, R, 2, 12, T)
    return t.reshape(B, BANDS, C, 12, T // TOK, TOK).permute(0, 1, 4, 2, 3, 5).reshape(B, BANDS, T // TOK, -1)


def tokens_sound(feat):                      # (B, 7, 264, T) -> (B, BANDS, N, DS)
    B, C, K, T = feat.shape
    T = T // TOK * TOK
    f = F.pad(feat[..., :T], (0, 0, 0, 3 * KEY0[-1] + 36 - K))
    t = torch.stack([f[:, :, 3 * k:3 * k + 36] for k in KEY0], 1)               # (B, R, 7, 36, T)
    return t.reshape(B, BANDS, C, 36, T // TOK, TOK).permute(0, 1, 4, 2, 3, 5).reshape(B, BANDS, T // TOK, -1)


from app.ml.spectral.model import Block as _Block, ROWS as _ROWS  # noqa: E402
assert _ROWS == BANDS


class Encoder(nn.Module):
    """Octave-window tokens -> time attention within each window, then attention across windows
    (the transcriber's Block).  Masked cells are replaced by a learned [MASK]."""

    def __init__(self, din, d=256, depth=6, heads=4):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(din, d), nn.GELU(), nn.Linear(d, d))
        self.mask = nn.Parameter(torch.zeros(d)); self.band = nn.Parameter(torch.randn(1, BANDS, 1, d) * 0.02)
        self.blocks = nn.ModuleList(_Block(d, heads) for _ in range(depth)); self.norm = nn.LayerNorm(d)

    def forward(self, x, m=None):            # x (B, R, N, din), m (B, R, N) bool = masked
        h = self.inp(x)
        if m is not None:
            h = torch.where(m[..., None], self.mask.to(h.dtype), h)
        h = h + self.band
        for b in self.blocks:
            h = b(h)
        return self.norm(h)


class Cross(nn.Module):
    """Translate one modality into the other: query cells (no input) attend, per octave window,
    to the other modality's encoding of that window and its two neighbours, then self-attend."""

    def __init__(self, d=256, depth=3, heads=4, dout=1):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, BANDS, 1, d) * 0.02); self.t = nn.Parameter(torch.randn(1, 1, 512, d) * 0.02)
        self.att = nn.ModuleList(nn.MultiheadAttention(d, heads, batch_first=True) for _ in range(depth))
        self.n1 = nn.ModuleList(nn.LayerNorm(d) for _ in range(depth))
        self.blocks = nn.ModuleList(_Block(d, heads) for _ in range(depth)); self.head = nn.Linear(d, dout)

    def forward(self, mem):                  # mem (B, R, N, d) -> (B, R, N, dout)
        B, R, N, d = mem.shape
        h = (self.q + self.t[:, :, :N]).expand(B, R, N, d)
        pad = F.pad(mem, (0, 0, 0, 0, 1, 1))                                     # neighbours of each window
        kv = torch.cat([pad[:, :-2], pad[:, 1:-1], pad[:, 2:]], 2).reshape(B * R, 3 * N, d)
        for att, n1, blk in zip(self.att, self.n1, self.blocks):
            q = n1(h).reshape(B * R, N, d)
            h = h + att(q, kv, kv, need_weights=False)[0].view(B, R, N, d)
            h = blk(h)
        return self.head(h)


def span_mask(B, R, N, frac=0.3, span=4, dev="cuda"):
    """Mask ~frac of the octave-window x time cells, in runs of `span` tokens within a window."""
    m = torch.zeros(B, R, N, dtype=torch.bool, device=dev)
    k = max(1, int(R * N * frac / span))
    for b in range(B):
        rs = np.random.randint(0, R, k); ts = np.random.randint(0, N - span, k)
        for r, t in zip(rs, ts):
            m[b, r, t:t + span] = True
    return m


def notes_loss(pred, roll_tok, m=None):
    """BCE on the roll tokens.  Piano rolls are mostly empty, so plain BCE is minimised by
    predicting nothing (the first cross run did: sound->notes frame F1 ~0.10 for every
    variant).  Sounding keys weigh 5x, onsets 20x."""
    pw = torch.ones_like(roll_tok[0, 0, 0]); pw[: 12 * TOK] = 20.0; pw[12 * TOK:] = 5.0
    l = F.binary_cross_entropy_with_logits(pred, roll_tok, pos_weight=pw, reduction="none").mean(-1)
    return (l * m).sum() / m.sum().clamp(min=1) if m is not None else l.mean()


def sound_loss(pred, feat_tok, m=None):
    l = (pred - feat_tok).abs().mean(-1)
    return (l * m).sum() / m.sum().clamp(min=1) if m is not None else l.mean()


def frame_counts(pred_logit, roll_tok, thrs=(0.3, 0.5, 0.7, 0.8, 0.9)):
    """(tp, n_pred, n_true) of sounding keys at each threshold — F1 is reported at the best one."""
    p = torch.sigmoid(pred_logit[..., 12 * TOK:]); g = roll_tok[..., 12 * TOK:] > 0.5
    return np.array([[((p > t) & g).sum().item(), (p > t).sum().item(), g.sum().item()] for t in thrs])


def best_f1(c, thrs=(0.3, 0.5, 0.7, 0.8, 0.9)):
    f = 2 * c[:, 0] / np.maximum(1, c[:, 1] + c[:, 2])
    return float(f.max()), thrs[int(f.argmax())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True); ap.add_argument("--fe", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--grammar-steps", type=int, default=8000); ap.add_argument("--cross-steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=500); ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--grammars-only", action="store_true", help="train the two grammars, skip cross-attention")
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    from app.ml.spectral.train import log_to
    say = log_to(out / "train.log"); dev = "cuda"; torch.manual_seed(0); random.seed(0)
    ck = torch.load(a.fe, map_location=dev, weights_only=False)
    fe = S.SpectralFrontend().to(dev).eval(); fe.mu.copy_(ck["fe_mu"]); fe.sd.copy_(ck["fe_sd"])
    clip = lambda s: sorted(p for p in (pathlib.Path(a.clips) / s).glob("pop*_[0-9][0-9][0-9].npz"))
    tr, va = clip("train")[: a.limit or None], clip("validation")[: 300 if not a.limit else 8]
    say(f"piano grammar: {len(tr)} train / {len(va)} validation POP909 clips (8 s crops; tokens = octave window x 64 ms, {BANDS} windows)")
    dl = DataLoader(Clips(tr), batch_size=a.batch, shuffle=True, drop_last=True, num_workers=a.workers,
                    persistent_workers=a.workers > 0)
    vdl = DataLoader(Clips(va, train=False), batch_size=a.batch, num_workers=a.workers)

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
            m = span_mask(*x.shape[:3], dev=dev)
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
                        np.random.seed(1)
                        m = span_mask(*x.shape[:3], dev=dev); p = head(enc(x, m)).float()
                        vl.append(float(notes_loss(p, x, m) if kind == "notes" else sound_loss(p, x, m)))
                enc.train(); curve.append((step, float(np.mean(vl))))
                say(f"  {kind} grammar step {step}: masked fill-in loss (val) {np.mean(vl):.4f}")
        torch.save({"enc": enc.state_dict(), "head": head.state_dict(), "din": DN if kind == "notes" else DS}, path)
        results[f"{kind}_grammar_curve"] = curve; save_results()

    if a.grammars_only:
        say("grammars only: done"); return
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
                cnt, sl = 0, []
                with torch.no_grad():
                    for y, roll in vdl:
                        tn, ts = prep(y.to(dev), roll.to(dev))
                        cnt = cnt + frame_counts(s2n(es(ts)).float(), tn); sl.append(float(sound_loss(n2s(en(tn)).float(), ts)))
                f1, thr = best_f1(cnt); f1s = [f1]
                for mdl in (en, es, s2n, n2s):
                    mdl.train(not frozen if mdl in (en, es) else True)
                curve.append((step, float(np.mean(f1s)), float(np.mean(sl))))
                say(f"  cross {variant} step {step}: sound->notes frame F1 {f1:.3f} (thr {thr})  notes->sound L1 {np.mean(sl):.4f}")
        results[f"cross_{variant}"] = curve; save_results()
    say("done: " + json.dumps({k: (v[-1] if isinstance(v, list) and v else v) for k, v in results.items()}))


if __name__ == "__main__":
    main()
