"""Structure-aware grammar: read notes that may be wrong, write the STRUCTURED arrangement.

Encoder input  — notes in the plain format (tokens.py: seconds, no structure),
                 corrupted with transcriber-like errors: exactly what the
                 ears hand over later (v1's notes, joined by v1's audio features
                 in the audio stage).
Decoder output — the same passage in the structure-aware language (tokens2.py):
                 bars, meter, key, section, rhythm group, chords on their beats,
                 and every note with its role, bar position and pitch.

So the decoder must INFER the structure (which chord, which group, which notes
are melody) and FIX the notes in one go — the user's correction idea, now in
musical terms.  Windows are whole bars (up to ~950 tokens).

Warm start from the grammar v1 checkpoint: the note tokens keep their ids, so
their embeddings and every transformer layer carry over; only the structure
tokens' embedding/output rows are new.

    python -m app.ml.grammar.train2 --corpus <grammar2 dir> --init <grammar_v1 best.pt> --ckpt-dir <dir>
"""
from __future__ import annotations

import argparse
import math
import pathlib
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from . import tokens as T, tokens2 as T2
from .model import GrammarModel

MAX_TGT, MAX_SRC = 950, 1000


def load(corpus, split):
    out = []
    for f in sorted(pathlib.Path(corpus).glob(f"*_{split}.npz")):
        z = np.load(f, allow_pickle=True)
        for i in range(len(z["ids"])):
            out.append((z["tokens"][z["starts"][i]:z["starts"][i + 1]].astype(np.int64),
                        np.asarray(z["beats"][i], float), np.asarray(z["bars"][i], int)))
    return out


class Windows(IterableDataset):
    def __init__(self, songs, seed=0, bars=(4, 12)):
        self.songs, self.seed, self.bars = songs, seed, bars

    def sample(self, rng):
        tok, beats, grid = self.songs[rng.integers(len(self.songs))]
        bar_ix = np.flatnonzero(tok == T2.BAR)
        if len(bar_ix) < 2:
            return None
        nb = int(rng.integers(*self.bars))
        b0 = int(rng.integers(0, max(1, len(bar_ix) - nb)))
        a = bar_ix[b0]
        e = bar_ix[b0 + nb] if b0 + nb < len(bar_ix) else len(tok)
        # the window must carry its context: re-state meter/key/section at its first bar
        head = [t for t in tok[:a] if T2.METER0 <= t < T2.KEY0][-1:] + \
               [t for t in tok[:a] if T2.KEY0 <= t < T2.SEC0][-1:] + \
               [t for t in tok[:a] if T2.SEC0 <= t < T2.GROUP0][-1:]
        clean = [T2.BAR] + head + list(tok[a + 1:e])
        while len(clean) > MAX_TGT:                      # drop bars from the end
            last = max(i for i, t in enumerate(clean) if t == T2.BAR)
            if last == 0:
                clean = clean[:MAX_TGT]; break
            clean = clean[:last]
        g0 = b0 if b0 < len(grid) else len(grid) - 1
        notes, _, _ = T2.decode(clean, beats, list(grid[g0:]))
        if len(notes) == 0:
            return None
        t0 = notes[:, 0].min()
        noisy = notes.copy(); noisy[:, :2] -= t0
        noisy = T.corrupt(noisy, rng)
        return T.encode(noisy)[:MAX_SRC], clean

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + (info.id if info else 0) * 7919 + random.randrange(1 << 30))
        while True:
            s = self.sample(rng)
            if s is not None:
                yield s


def collate(batch):
    L1 = max(len(s) for s, _ in batch) or 1
    L2 = max(len(t) for _, t in batch) + 1
    src = torch.full((len(batch), L1), T.PAD, dtype=torch.long)
    tin = torch.full((len(batch), L2), T.PAD, dtype=torch.long)
    tout = torch.full((len(batch), L2), T.PAD, dtype=torch.long)
    for b, (s, t) in enumerate(batch):
        src[b, :len(s)] = torch.tensor(s)
        tin[b, :len(t) + 1] = torch.tensor([T.BOS] + t)
        tout[b, :len(t) + 1] = torch.tensor(t + [T.EOS])
    return src, tin, tout


def warm_start(net, path):
    """Copy grammar v1 weights; the new structure tokens get fresh rows."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd, own = ck["model"], net.state_dict()
    for k, v in sd.items():
        if k not in own:
            continue
        if own[k].shape == v.shape:
            own[k] = v
        elif k in ("emb.weight", "out.weight") or k == "out.bias":
            own[k][: v.shape[0]] = v                     # old vocab rows; structure rows stay random
    net.load_state_dict(own)
    return ck.get("step", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--init", default="")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--total-steps", type=int, default=40000)
    ap.add_argument("--val-every", type=int, default=2500)
    ap.add_argument("--max-minutes", type=float, default=170)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    dev = torch.device("cuda")
    ckdir = pathlib.Path(args.ckpt_dir); ckdir.mkdir(parents=True, exist_ok=True)
    log = open(ckdir / "train.log", "a")

    def say(m):
        line = f"{time.strftime('%m-%d %H:%M:%S')}  {m}"; print(line, flush=True); log.write(line + "\n"); log.flush()

    tr, va = load(args.corpus, "train"), load(args.corpus, "validation")
    say(f"{len(tr)} train songs, {len(va)} validation songs, vocab {T2.VOCAB}")
    net = GrammarModel(max_len=1024, vocab=T2.VOCAB)
    if args.init:
        say(f"warm start from {args.init} (grammar step {warm_start(net, args.init)})")
    net = net.to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
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
    rng = np.random.default_rng(11)
    vds = Windows(va, seed=5)
    va_fixed = [s for s in (vds.sample(rng) for _ in range(700)) if s is not None][:384]
    dl = DataLoader(Windows(tr), batch_size=args.batch, collate_fn=collate, num_workers=args.workers)
    kinds = {"note": (T.PITCH0, T.DUR0), "chord": (T2.CHORD0, T2.CHORD_NONE + 1), "role": (T2.ROLE0, T2.VOCAB),
             "group": (T2.GROUP0, T2.POS0), "pos": (T2.POS0, T2.CHORD0), "key": (T2.KEY0, T2.SEC0)}

    def validate():
        net.eval(); tl, tn = 0.0, 0
        hit = {k: [0, 0] for k in kinds}
        with torch.no_grad():
            for i in range(0, len(va_fixed), 32):
                src, tin, tout = (x.to(dev) for x in collate(va_fixed[i:i + 32]))
                with torch.autocast("cuda", dtype=amp):
                    lg = net(src, tin).float()
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
    for src, tin, tout in dl:
        src, tin, tout = src.to(dev), tin.to(dev), tout.to(dev)
        with torch.autocast("cuda", dtype=amp):
            lg = net(src, tin)
        loss = F.cross_entropy(lg.float().reshape(-1, T2.VOCAB), tout.reshape(-1), ignore_index=T.PAD)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        step += 1; run += float(loss); k += 1
        if step % 200 == 0:
            say(f"step {step:>6}  loss {run / k:.4f}  lr {sched.get_last_lr()[0]:.2e}"); run, k = 0.0, 0
        stop = step >= args.total_steps or (time.time() - t0) / 60 > args.max_minutes
        if step % args.val_every == 0 or stop:
            vl, acc = validate()
            ck = {"model": net.state_dict(), "arch": net.arch, "opt": opt.state_dict(),
                  "sched": sched.state_dict(), "step": step, "best": min(best, vl)}
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
