"""Clean-up v3 (2026-10-27): v2 + a MERGE output (the user's plan).  Per note: keep / drop as before, plus "merge into
the previous strike of the same key" -- the merged note keeps that strike's start and ends at the later end (only the
gap between them is filled: never a new start, never past the base's own sound).

Labels: eval/merge_oracle.py --labels, the plan per key that brings the base's notes closest to the written
arrangement, starts AND ends (/scratch/basu.ang/merge_labels: 0 drop, 1 keep, 2 merge).  Measured ceiling on the
held-out clips: note F1 with ends 0.619 with merges vs 0.567 for perfect keep / drop (v2 at keep 0.3: 0.519).

Fine-tuned from v2 (checkpoints/cleanup2/best.pt): same inputs and encoder; the merge output starts at zero, so step 0
is v2.  Runs in pieces (--budget-min, e.g. gpu-short's 2 h wall): saves last.pt and exits, the next piece resumes.

    python -m app.ml.spectral.cleanup3_train --data /scratch/basu.ang/cleanup_data,/scratch/basu.ang/cleanup_data_tr \
        --beats /scratch/basu.ang/clip_beats --labels /scratch/basu.ang/merge_labels --init checkpoints/cleanup2/best.pt \
        --split checkpoints/cleanup1/split.json --out checkpoints/cleanup3 --budget-min 110
"""
import argparse
import json
import math
import pathlib
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from app.ml.spectral import cleanup2_train as C2
from app.ml.spectral import cleanup_train as C1

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "eval"))
MAXN = C2.MAXN


class Cleanup3(C2.Cleanup2):
    def __init__(self, d=256, depth=6, heads=4):
        super().__init__(d, depth, heads)
        self.merge = nn.Linear(d, 1)
        nn.init.zeros_(self.merge.weight)
        nn.init.constant_(self.merge.bias, -4.0)               # ~2%: merges nothing until it learns to

    def forward(self, pitch, t, x, pad):
        h = self.pitch(pitch) + self.feat(x) + self.time_emb(t)
        e = self.enc(h, src_key_padding_mask=pad)
        o = self.head(e)
        return o[..., 0], o[..., 1:], self.merge(self.head[0](e))[..., 0]


def decide(n, pk, pm, keep_bar, merge_bar):
    """the clean-up's notes: (M, 3) onset, offset, pitch (+ velocity in col 3).  n: (N, >=4) the base's notes."""
    out = []
    for p in np.unique(n[:, 2]).astype(int):
        idx = np.nonzero(n[:, 2] == p)[0]
        idx = idx[np.argsort(n[idx, 0], kind="stable")]
        alive = False                                          # the previous strike on this key is in the output
        for i in idx:
            if alive and pm[i] >= merge_bar:
                out[-1][1] = max(out[-1][1], n[i, 1])
                out[-1][3] = max(out[-1][3], n[i, 3])
            elif pk[i] >= keep_bar:
                out.append([n[i, 0], n[i, 1], p, n[i, 3]])
                alive = True
            else:
                alive = False
    return np.array(out, float).reshape(-1, 4)


def batch(clips, dev):
    P, T, X, Y, Q, pad = C2.batch(clips, dev)
    B, N = P.shape
    A = torch.full((B, N), -1, dtype=torch.long)
    for i, c in enumerate(clips):
        n = min(N, len(c["act"]))
        A[i, :n] = torch.from_numpy(c["act"][:n].astype(np.int64))
    return P, T, X, Y, Q, pad, A.to(dev)


@torch.no_grad()
def evaluate(net, clips, data_dirs, dev, bs=32):
    """note F1 (mir_eval; starts, and starts + ends) of the decisions on held-out clips, at several bars."""
    from merge_oracle import f1s
    net.eval()
    bars = [(0.3, None), (0.3, 0.3), (0.3, 0.5), (0.3, 0.7), (0.4, 0.5)]
    R = {b: dict(pred=0, gt=0, on=0, onoff=0) for b in bars}
    for i in range(0, len(clips), bs):
        cs = clips[i:i + bs]
        P, T, X, Y, Q, pad, A = batch(cs, dev)
        lk, _, lm = net(P, T, X, pad)
        pk_, pm_ = torch.sigmoid(lk.float()).cpu().numpy(), torch.sigmoid(lm.float()).cpu().numpy()
        for j, c in enumerate(cs):
            z = np.load(c["path"])
            n, gt = z["notes"].astype(float), z["gt"].astype(float).reshape(-1, 3)
            N = min(MAXN, len(n))
            pk = np.ones(len(n)); pm = np.zeros(len(n))
            pk[:N], pm[:N] = pk_[j, :N], pm_[j, :N]
            for kb, mb in bars:
                out = decide(n[:, [0, 1, 2, 3]], pk, pm if mb is not None else np.zeros(len(n)), kb, mb or 2.0)
                r = f1s(out[:, :3], gt)
                R[(kb, mb)]["pred"] += r["n_pred"]; R[(kb, mb)]["gt"] += r["n_gt"]
                R[(kb, mb)]["on"] += r["on"][0]; R[(kb, mb)]["onoff"] += r["onoff"][0]
    net.train()

    def f1(h, r):
        rc, pr = h / max(1, r["gt"]), h / max(1, r["pred"])
        return round(2 * rc * pr / max(1e-9, rc + pr), 4)
    return {f"keep {kb}" + ("" if mb is None else f" merge {mb}"): dict(F1=f1(r["on"], r), F1_ends=f1(r["onoff"], r),
                                                                      notes=r["pred"]) for (kb, mb), r in R.items()}


def pack_labels(lab, threads=64):
    """all <clip>.npy labels in one file (_all.pt): one read instead of 95k small reads on /scratch."""
    import os
    from concurrent.futures import ThreadPoolExecutor
    names = [n[:-4] for n in os.listdir(lab) if n.endswith(".npy") and ".tmp" not in n]
    with ThreadPoolExecutor(threads) as ex:
        arrs = list(ex.map(lambda n: np.load(pathlib.Path(lab) / (n + ".npy")), names))
    torch.save(dict(zip(names, arrs)), pathlib.Path(lab) / "_all.pt")
    print(f"packed {len(names)} label files", flush=True)


def load(a):
    import os
    clips = C2.load(a.data.split(","), a.beats, a.workers)
    lab = pathlib.Path(a.labels)
    packed = torch.load(lab / "_all.pt", weights_only=False) if (lab / "_all.pt").exists() else None
    where = {}
    for d in a.data.split(","):                                  # one listing per dir, not a stat per clip
        for n in os.listdir(d):
            if n.endswith(".npz") and ".tmp" not in n:
                where.setdefault(n[:-4], str(pathlib.Path(d) / n))
    out = []
    for c in clips:
        if packed is not None:
            act = packed.get(c["name"])
        else:
            f = lab / (c["name"] + ".npy")
            act = np.load(f) if f.exists() else None
        if act is None or len(act) != len(c["y"]) or c["name"] not in where:
            continue
        c["act"], c["path"] = act, where[c["name"]]
        out.append(c)
    return out


def main():
    ap = argparse.ArgumentParser()
    if "--pack-labels" in sys.argv:
        pack_labels(sys.argv[sys.argv.index("--pack-labels") + 1])
        return
    ap.add_argument("--data", required=True)
    ap.add_argument("--beats", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--init", required=True, help="v2 checkpoint to start from")
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--merge-weight", type=float, default=1.0)
    ap.add_argument("--pos-weight", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--val-clips", type=int, default=600, help="held-out clips per validation (all at the end)")
    ap.add_argument("--budget-min", type=float, default=0, help="save and exit after this many minutes (0 = none)")
    a = ap.parse_args()
    t_start = time.time()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ck = out / "last.pt"
    if ck.exists() and torch.load(ck, map_location="cpu", weights_only=False)["step"] >= a.steps:
        print("already finished", flush=True)
        return
    dev = "cuda"
    clips = load(a)
    val_s = set(json.loads(pathlib.Path(a.split).read_text()))
    train = [c for c in clips if C1.song_of(c["name"]) not in val_s]
    val = [c for c in clips if C1.song_of(c["name"]) in val_s]
    acts = np.concatenate([c["act"] for c in train])
    print(f"{len(clips)} labelled clips: train {len(train)}, val {len(val)};  train notes keep {np.mean(acts == 1):.3f} "
          f"merge {np.mean(acts == 2):.3f} drop {np.mean(acts == 0):.3f}", flush=True)
    vsub = random.Random(0).sample(val, min(a.val_clips, len(val)))
    s = torch.load(a.init, map_location="cpu", weights_only=False)
    cfg = dict(s["cfg"]); cfg.pop("version", None)
    net = Cleanup3(**cfg)
    missing = net.load_state_dict(s["model"], strict=False)
    print(f"from {a.init} (step {s['step']}); new: {missing.missing_keys}", flush=True)
    net = net.to(dev)
    opt = torch.optim.AdamW(net.parameters(), a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda st: min(1, st / 500) * 0.5 * (1 + math.cos(math.pi * min(1, st / a.steps))))
    step, best = 0, -1
    if ck.exists():
        st_ = torch.load(ck, map_location=dev, weights_only=False)
        net.load_state_dict(st_["model"]); opt.load_state_dict(st_["opt"]); sched.load_state_dict(st_["sched"])
        step, best = st_["step"], st_["best"]
        print(f"resumed at step {step}", flush=True)
    else:
        ev = evaluate(net, vsub, a.data, dev)
        print(f"VAL step 0 " + json.dumps(ev), flush=True)
    cfg = dict(cfg, version=3)
    rng = random.Random(step)
    t0 = time.time()
    lk_a = lm_a = None

    def save_last():
        torch.save(dict(model=net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), step=step, best=best,
                        cfg=cfg), ck)
    while step < a.steps:
        P, T, X, Y, Q, pad, A = batch(rng.sample(train, a.bs), dev)
        with torch.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
            lk, lp, lm = net(P, T, X, pad)
        m = A >= 0
        l_keep = nn.functional.binary_cross_entropy_with_logits(lk.float()[m], (A[m] == 1).float())
        l_merge = nn.functional.binary_cross_entropy_with_logits(lm.float()[m], (A[m] == 2).float())
        mq = Q >= 0
        l_pos = nn.functional.cross_entropy(lp.float()[mq], Q[mq]) if mq.any() else torch.zeros((), device=dev)
        loss = l_keep + a.merge_weight * l_merge + a.pos_weight * l_pos
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        step += 1
        lk_a = l_keep.item() if lk_a is None else 0.98 * lk_a + 0.02 * l_keep.item()
        lm_a = l_merge.item() if lm_a is None else 0.98 * lm_a + 0.02 * l_merge.item()
        if step % 200 == 0:
            print(f"step {step} keep {lk_a:.4f} merge {lm_a:.4f} {(time.time() - t0) / 60:.1f} min", flush=True)
        if step % 2000 == 0 or step == a.steps:
            ev = evaluate(net, val if step == a.steps else vsub, a.data, dev)
            score = max(v["F1_ends"] for v in ev.values())
            print(f"VAL step {step}{' (all held-out clips)' if step == a.steps else ''} " + json.dumps(ev), flush=True)
            if score > best:
                best = score
                torch.save(dict(model=net.state_dict(), step=step, score=score, cfg=cfg, val=ev), out / "best.pt")
            save_last()
        if a.budget_min and (time.time() - t_start) / 60 > a.budget_min:
            save_last()
            print(f"budget reached at step {step}: saved, exiting for the next piece", flush=True)
            return
        if step % 50 == 0 and (out / "HANDOVER").exists():           # a faster GPU is waiting to take over
            save_last()
            print(f"HANDOVER at step {step}: saved, exiting", flush=True)
            return
    print("done", flush=True)


if __name__ == "__main__":
    main()
