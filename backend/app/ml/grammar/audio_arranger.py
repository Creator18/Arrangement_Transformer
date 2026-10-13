"""Audio arranger: the structure-aware grammar decoder, listening to the song through v1.

    song audio -> v1 (frozen, cached) -+-> v1's notes ---------> grammar encoder ---+
                                       +-> v1's trunk features -> audio adapter ----+-> decoder -> structured arrangement

The decoder writes the arrangement in the structure-aware language (tokens2):
bars, meter, key, section, rhythm group, chords on their beats, and every note
with its role — so it must work out WHAT KIND OF PART this is from what it
hears before choosing notes, and learns how human arrangers handle each kind
of part (the user's design).  Its memory is the concatenation of
  - the grammar encoder reading v1's own notes (plain tokens, as in pretraining)
  - a small audio adapter over v1's features: projection 1792 -> 384, a learned
    position per 32 ms patch, a type embedding, 2 transformer layers
Difficulty is a LEVEL token given at the start of the target (never predicted).

Training pairs (Music2MIDI train songs): a window of 4-12 bars of the human
arrangement, re-aligned to the audio (eval/realign_m2m.py) and structure-
labelled (grammar2 corpus), with the same bars' v1 features and v1 notes.
Warm-started from the structure-aware grammar (train2); only the adapter and
the three level tokens start fresh.

    python -m app.ml.grammar.audio_arranger --corpus <grammar2 dir> --cache <v1 cache> \\
        --m2m <dir> --init <grammar2 best.pt> --ckpt-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from . import tokens as T, tokens2 as T2
from .model import GrammarModel
from .train2 import warm_start

LEVEL0 = T2.VOCAB                   # beginner / intermediate / advanced
VOCAB = T2.VOCAB + 3
PATCH_S = 0.032
FEAT = 1792
MAX_SRC, MAX_AUDIO, MAX_TGT = 1000, 1000, 950
LEVELS = ["beginner", "intermediate", "advanced"]


class AudioArranger(nn.Module):
    def __init__(self, d=384):
        super().__init__()
        self.grammar = GrammarModel(max_len=1024, vocab=VOCAB)
        self.adapter = nn.Sequential(nn.LayerNorm(FEAT), nn.Linear(FEAT, d), nn.GELU(), nn.Linear(d, d))
        self.apos = nn.Embedding(MAX_AUDIO, d)
        self.mtype = nn.Parameter(torch.zeros(2, d))
        layer = nn.TransformerEncoderLayer(d, 6, 4 * d, 0.1, batch_first=True, norm_first=True)
        self.audio_enc = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)

    def memory(self, src, audio, audio_len):
        mn = self.grammar.encode(src) + self.mtype[0]
        A = audio.shape[1]
        am = torch.arange(A, device=audio.device)[None] >= audio_len[:, None]          # True = pad
        a = self.adapter(audio) + self.apos(torch.arange(A, device=audio.device))[None] + self.mtype[1]
        ma = self.audio_enc(a, src_key_padding_mask=am)
        return torch.cat([mn, ma], 1), torch.cat([src == T.PAD, am], 1)

    def forward(self, src, audio, audio_len, tgt_in):
        mem, pad = self.memory(src, audio, audio_len)
        return self.grammar.decode(tgt_in, mem, pad)

    @torch.no_grad()
    def generate(self, src, audio, audio_len, prefix, max_len=950):
        mem, pad = self.memory(src, audio, audio_len)
        y = prefix.clone()
        done = torch.zeros(y.shape[0], dtype=torch.bool, device=y.device)
        for _ in range(max_len):
            nxt = self.grammar.decode(y, mem, pad)[:, -1].argmax(-1)
            nxt[done] = T.PAD
            y = torch.cat([y, nxt[:, None]], 1)
            done |= nxt == T.EOS
            if done.all():
                break
        return y[:, prefix.shape[1]:]


# ── data ─────────────────────────────────────────────────────────────────────
def load(corpus, cache, m2m, split):
    sp = json.loads((pathlib.Path(m2m) / "m2m_splits.json").read_text())
    out = []
    for f in sorted(pathlib.Path(corpus).glob(f"m2m*_{split}.npz")):
        z = np.load(f, allow_pickle=True)
        for i, sid in enumerate(z["ids"]):
            c = pathlib.Path(cache) / f"{sid}.npz"
            if not c.exists():
                continue
            out.append(dict(id=str(sid), tok=z["tokens"][z["starts"][i]:z["starts"][i + 1]].astype(np.int64),
                            beats=np.asarray(z["beats"][i], float), grid=np.asarray(z["bars"][i], int),
                            cache=c, level=LEVELS.index(sp["difficulty"][str(sid)])))
    return out


def window(song, rng, bars=(4, 12)):
    tok, beats, grid = song["tok"], song["beats"], song["grid"]
    bar_ix = np.flatnonzero(tok == T2.BAR)
    if len(bar_ix) < 2:
        return None
    nb = int(rng.integers(*bars))
    b0 = int(rng.integers(0, max(1, len(bar_ix) - nb)))
    b1 = min(b0 + nb, len(bar_ix))
    a, e = bar_ix[b0], (bar_ix[b1] if b1 < len(bar_ix) else len(tok))
    head = [t for t in tok[:a] if T2.METER0 <= t < T2.KEY0][-1:] + \
           [t for t in tok[:a] if T2.KEY0 <= t < T2.SEC0][-1:] + \
           [t for t in tok[:a] if T2.SEC0 <= t < T2.GROUP0][-1:]
    clean = [T2.BAR] + head + list(tok[a + 1:e])
    while len(clean) > MAX_TGT - 1:
        last = max(i for i, t in enumerate(clean) if t == T2.BAR)
        if last == 0:
            clean = clean[:MAX_TGT - 1]; break
        clean = clean[:last]; b1 = b0 + sum(1 for t in clean if t == T2.BAR)
    if b0 >= len(grid) - 1:
        return None
    t_a = beats[min(grid[b0], len(beats) - 1)]
    t_b = beats[min(grid[min(b1, len(grid) - 1)], len(beats) - 1)]
    if t_b - t_a > MAX_AUDIO * PATCH_S:
        t_b = t_a + MAX_AUDIO * PATCH_S
    z = np.load(song["cache"])
    f0, f1 = int(t_a / PATCH_S), int(t_b / PATCH_S) + 1
    feat = z["feat"][f0:f1][:MAX_AUDIO].astype(np.float32)
    n = z["notes"]
    n = n[(n[:, 0] >= t_a) & (n[:, 0] < t_b)].astype(np.float64).copy()
    n[:, :2] -= t_a
    src = T.encode(n)[:MAX_SRC] if len(n) else []
    return src, feat, [LEVEL0 + song["level"]] + clean


class Windows(IterableDataset):
    def __init__(self, songs, seed=0):
        self.songs, self.seed = songs, seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + (info.id if info else 0) * 104729 + random.randrange(1 << 30))
        while True:
            w = window(self.songs[rng.integers(len(self.songs))], rng)
            if w is not None and len(w[1]):
                yield w


def collate(batch):
    L1 = max(max(len(s) for s, _, _ in batch), 1)
    A = max(len(f) for _, f, _ in batch)
    L2 = max(len(t) for _, _, t in batch) + 1
    src = torch.full((len(batch), L1), T.PAD, dtype=torch.long)
    aud = torch.zeros((len(batch), A, FEAT))
    alen = torch.tensor([len(f) for _, f, _ in batch])
    tin = torch.full((len(batch), L2), T.PAD, dtype=torch.long)
    tout = torch.full((len(batch), L2), T.PAD, dtype=torch.long)
    for b, (s, f, t) in enumerate(batch):
        if len(s):
            src[b, :len(s)] = torch.tensor(s)
        aud[b, :len(f)] = torch.from_numpy(f)
        tin[b, :len(t) + 1] = torch.tensor([T.BOS] + t)
        tout[b, :len(t) + 1] = torch.tensor(t + [T.EOS])
        tout[b, 0] = T.PAD                                    # the level token is given, not predicted
    return src, aud, alen, tin, tout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--m2m", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--init", default="")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--total-steps", type=int, default=30000)
    ap.add_argument("--val-every", type=int, default=2500)
    ap.add_argument("--max-minutes", type=float, default=170)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    dev = torch.device("cuda")
    ckdir = pathlib.Path(args.ckpt_dir); ckdir.mkdir(parents=True, exist_ok=True)
    log = open(ckdir / "train.log", "a")

    def say(m):
        line = f"{time.strftime('%m-%d %H:%M:%S')}  {m}"; print(line, flush=True); log.write(line + "\n"); log.flush()

    tr = load(args.corpus, args.cache, args.m2m, "train")
    va = load(args.corpus, args.cache, args.m2m, "validation")
    say(f"{len(tr)} train songs, {len(va)} validation songs with v1 features")
    net = AudioArranger()
    if args.init:
        say(f"decoder warm start from {args.init} (step {warm_start(net.grammar, args.init)})")
    net = net.to(dev)
    fresh = [p for n, p in net.named_parameters() if not n.startswith("grammar.")]
    opt = torch.optim.AdamW([{"params": net.grammar.parameters(), "lr": args.lr},
                             {"params": fresh, "lr": 3 * args.lr}], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / args.warmup) * 0.5 *
                                              (1 + math.cos(math.pi * min(1, s / args.total_steps))))
    bf16 = torch.cuda.get_device_capability(0)[0] >= 8
    amp = torch.bfloat16 if bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=not bf16)
    step, best = 0, float("inf")
    last = ckdir / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=dev, weights_only=False)
        net.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        step, best = ck["step"], ck["best"]; say(f"resumed at step {step}")
    rng = np.random.default_rng(13)
    va_fixed = [w for w in (window(va[rng.integers(len(va))], rng) for _ in range(500)) if w is not None and len(w[1])][:256]
    dl = DataLoader(Windows(tr), batch_size=args.batch, collate_fn=collate, num_workers=args.workers)
    kinds = {"note": (T.PITCH0, T.DUR0), "chord": (T2.CHORD0, T2.CHORD_NONE + 1), "role": (T2.ROLE0, T2.VOCAB),
             "group": (T2.GROUP0, T2.POS0), "pos": (T2.POS0, T2.CHORD0), "key": (T2.KEY0, T2.SEC0)}

    def validate():
        net.eval(); tl, tn = 0.0, 0
        hit = {k: [0, 0] for k in kinds}
        with torch.no_grad():
            for i in range(0, len(va_fixed), 8):
                src, aud, alen, tin, tout = (x.to(dev) for x in collate(va_fixed[i:i + 8]))
                with torch.autocast("cuda", dtype=amp):
                    lg = net(src, aud, alen, tin).float()
                m = tout != T.PAD
                tl += float(F.cross_entropy(lg[m], tout[m], reduction="sum")); tn += int(m.sum())
                pr = lg.argmax(-1)
                for k, (lo, hi) in kinds.items():
                    sel = m & (tout >= lo) & (tout < hi)
                    hit[k][0] += int((pr[sel] == tout[sel]).sum()); hit[k][1] += int(sel.sum())
        net.train()
        return tl / tn, {k: h[0] / max(1, h[1]) for k, h in hit.items()}

    t0, run, k = time.time(), 0.0, 0
    net.train()
    for src, aud, alen, tin, tout in dl:
        src, aud, alen, tin, tout = (x.to(dev, non_blocking=True) for x in (src, aud, alen, tin, tout))
        with torch.autocast("cuda", dtype=amp):
            lg = net(src, aud, alen, tin)
        loss = F.cross_entropy(lg.float().reshape(-1, VOCAB), tout.reshape(-1), ignore_index=T.PAD)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        step += 1; run += float(loss); k += 1
        if step % 200 == 0:
            say(f"step {step:>6}  loss {run / k:.4f}"); run, k = 0.0, 0
        stop = step >= args.total_steps or (time.time() - t0) / 60 > args.max_minutes
        if step % args.val_every == 0 or stop:
            vl, acc = validate()
            ck = {"model": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                  "step": step, "best": min(best, vl)}
            mark = ""
            if vl < best:
                best = vl; torch.save(ck, ckdir / "best.pt"); mark = "  * best"
            torch.save(ck, last)
            say(f"VAL step {step}: loss {vl:.4f}  next-token accuracy — " +
                ", ".join(f"{k} {v:.3f}" for k, v in acc.items()) + mark)
        if stop:
            say(f"stopping at step {step}"); break


if __name__ == "__main__":
    main()
