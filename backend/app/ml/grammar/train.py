"""Pretrain the grammar model: read wrong notes, write the right ones.

Each example is a window of a real piece (~up to 480 tokens of clean notes).
The encoder sees it CORRUPTED with the errors a transcriber makes (semitone
and octave slips, extra and missing notes, timing jitter — tokens.corrupt);
the decoder must write the clean version.  To do that it has to know what
belongs: which note fits the harmony, where the bass goes, how a pattern
continues — music's grammar, learned from the corpora themselves.

Sources are sampled by WEIGHT, not by size, so the largest corpus cannot drown
the others (--weights m2m=0.3,lakh=0.3,...).

Validation: loss and token accuracy on held-out windows, and the number that
matters — note F1 of the model's correction against the clean notes, beside
the F1 of the corrupted input it was given.  Correction F1 > input F1 means it
fixes more than it breaks.

    python -m app.ml.grammar.train --corpus <dir> --ckpt-dir <dir>
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

from . import tokens as T
from .model import GrammarModel

MAX_TGT = 480


class Windows(IterableDataset):
    def __init__(self, archives, weights, seed=0, max_tgt=MAX_TGT, strength=1.0):
        self.arch = archives; self.w = np.array(weights) / sum(weights)
        self.seed, self.max_tgt, self.strength = seed, max_tgt, strength

    def sample(self, rng):
        a = self.arch[rng.choice(len(self.arch), p=self.w)]
        i = rng.integers(0, len(a["starts"]) - 1)
        n = a["notes"][a["starts"][i]:a["starts"][i + 1]]
        j = rng.integers(0, max(1, len(n) - 20))
        w = n[j:j + 200].astype(np.float64).copy()
        w[:, :2] -= w[0, 0]
        clean = T.encode(w)
        if len(clean) > self.max_tgt:                        # cut on a note boundary
            k = self.max_tgt
            while k > 0 and not (T.PITCH0 <= clean[k] < T.DUR0):
                k -= 1
            clean = clean[:k]
            w = T.decode(clean)
        noisy = T.encode(T.corrupt(w, rng, self.strength))[: self.max_tgt + 120]
        return noisy, clean

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + (info.id if info else 0) * 9973 + random.randrange(1 << 30))
        while True:
            yield self.sample(rng)


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


def note_f1(ref, est):
    from mir_eval.transcription import precision_recall_f1_overlap as prf
    if len(ref) == 0 or len(est) == 0:
        return 0.0
    iv = lambda n: np.c_[n[:, 0], np.maximum(n[:, 1], n[:, 0] + 1e-3)]
    hz = lambda n: 440 * 2 ** ((n[:, 2] - 69) / 12)
    return prf(iv(ref), hz(ref), iv(est), hz(est), onset_tolerance=0.05, offset_ratio=None)[2]


def load(corpus, split, names):
    out = []
    for n in names:
        fs = sorted(pathlib.Path(corpus).glob(f"{n}_{split}.npz")) + \
            sorted(pathlib.Path(corpus).glob(f"{n}_[0-9][0-9]_{split}.npz"))
        if not fs:
            continue
        notes, starts = [], [0]
        for f in fs:
            z = np.load(f)
            notes.append(z["notes"]); starts += list(z["starts"][1:] + starts[-1])
        out.append((n, {"notes": np.concatenate(notes), "starts": np.array(starts)}))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--weights", default="m2m=0.3,lakh=0.3,pop909=0.15,maestro=0.15,asap=0.1")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--total-steps", type=int, default=60000)
    ap.add_argument("--val-every", type=int, default=2500)
    ap.add_argument("--max-minutes", type=float, default=175)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    dev = torch.device("cuda")
    ckdir = pathlib.Path(args.ckpt_dir); ckdir.mkdir(parents=True, exist_ok=True)
    log = open(ckdir / "train.log", "a")
    def say(m):
        line = f"{time.strftime('%m-%d %H:%M:%S')}  {m}"; print(line, flush=True); log.write(line + "\n"); log.flush()

    wts = dict(kv.split("=") for kv in args.weights.split(","))
    tr = load(args.corpus, "train", wts); va = load(args.corpus, "validation", wts)
    say("sources: " + ", ".join(f"{n} {len(a['starts']) - 1} pieces (w {wts[n]})" for n, a in tr))
    tr_ds = Windows([a for _, a in tr], [float(wts[n]) for n, _ in tr])
    va_ds = Windows([a for _, a in va], [float(wts[n]) for n, _ in va], seed=12345)
    rng = np.random.default_rng(7)
    va_fixed = [va_ds.sample(rng) for _ in range(512)]          # identical every validation
    dl = DataLoader(tr_ds, batch_size=args.batch, collate_fn=collate, num_workers=args.workers)

    net = GrammarModel().to(dev)
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
    say(f"model {sum(p.numel() for p in net.parameters()) / 1e6:.1f}M params, batch {args.batch}, "
        f"{torch.cuda.get_device_name(0)}")

    def validate(n_gen=64):
        net.eval(); tl, ta, tn = 0.0, 0, 0
        with torch.no_grad():
            for i in range(0, len(va_fixed), 64):
                src, tin, tout = (x.to(dev) for x in collate(va_fixed[i:i + 64]))
                with torch.autocast("cuda", dtype=amp):
                    lg = net(src, tin).float()
                m = tout != T.PAD
                tl += float(F.cross_entropy(lg[m], tout[m], reduction="sum")); tn += int(m.sum())
                ta += int((lg.argmax(-1)[m] == tout[m]).sum())
            src, _, _ = collate(va_fixed[:n_gen])
            with torch.autocast("cuda", dtype=amp):
                gen = net.generate(src.to(dev)).cpu().numpy()
        f_in, f_out = [], []
        for (noisy, clean), g in zip(va_fixed[:n_gen], gen):
            ref = T.decode(clean)
            f_in.append(note_f1(ref, T.decode(noisy)))
            f_out.append(note_f1(ref, T.decode([x for x in g if x != T.PAD])))
        net.train()
        return tl / tn, ta / tn, float(np.mean(f_in)), float(np.mean(f_out))

    t0, run, k = time.time(), 0.0, 0
    net.train()
    for src, tin, tout in dl:
        src, tin, tout = src.to(dev), tin.to(dev), tout.to(dev)
        with torch.autocast("cuda", dtype=amp):
            lg = net(src, tin)
        loss = F.cross_entropy(lg.float().reshape(-1, T.VOCAB), tout.reshape(-1), ignore_index=T.PAD)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        step += 1; run += float(loss); k += 1
        if step % 200 == 0:
            say(f"step {step:>6}  loss {run / k:.4f}  lr {sched.get_last_lr()[0]:.2e}"); run, k = 0.0, 0
        stop = step >= args.total_steps or (time.time() - t0) / 60 > args.max_minutes
        if step % args.val_every == 0 or stop:
            vl, acc, fi, fo = validate()
            mark = ""
            ck = {"model": net.state_dict(), "arch": net.arch, "opt": opt.state_dict(),
                  "sched": sched.state_dict(), "step": step, "best": min(best, vl)}
            if vl < best:
                best = vl; torch.save(ck, ckdir / "best.pt"); mark = "  * best"
            torch.save(ck, last)
            say(f"VAL step {step}: loss {vl:.4f}  token acc {acc:.3f}  correction note F1 "
                f"{fi:.3f} (corrupted input) -> {fo:.3f} (model){mark}")
        if stop:
            say(f"stopping at step {step}"); break


if __name__ == "__main__":
    main()
