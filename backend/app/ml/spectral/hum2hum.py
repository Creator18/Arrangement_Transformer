"""Stage 2, the user's design: song hum -> piano hum, two steps, from scratch, hum space only.

Data: per-pitch loudness maps of each window's song hum (A) and piano hum (B)
(eval/make_hum_maps.py), 264 bins x frames, compared in log space.
True differences, from the two hums alone:  A-B = relu(a - b) (in the song,
not the piano: dropped),  B-A = relu(b - a) (in the piano, not the song: added).
No arrangement notes and no other model anywhere in this stage.

  step 1   A -> its own (A-B, B-A).  Learns ONLY from the true differences.
  step 2   (A, step 1's differences) -> piano hum = song hum + a learned edit
           (zero at the start, so it begins exactly at the song hum: hums are
           ~94% silence and a model built up from nothing learns to output
           silence).  Learns from its own result, always on step 1's PREDICTED
           differences — never the true ones.

A unit is one song at one level; when it has several human arrangements,
every result is scored against the CLOSEST real piano hum, and how far the
humans are from each other is reported as the bar for "good enough".

Training, in rounds (the user's plan):
  round 0   step 1 studies the true differences; then step 2 trains on step 1's
            output until its score plateaus
  round r   when step 2 has plateaued, the remaining error is the differences'.
            Random hill climb over HOW step 1 studies the true differences: each
            candidate recipe re-weights step 1's comparison by pitch band, drop
            vs add, and where step 2's piano hum is still wrong (error-weighted,
            so it looks harder exactly where it missed something).  Each
            candidate: step 1 studies, step 2 re-adapts briefly, scored on
            held-out songs; the best is kept (a worse one only occasionally, to
            avoid getting stuck).  Then step 2 trains to a plateau again.
  stop      when --patience rounds bring no improvement, or --rounds is reached

Validation (held-out songs, full windows), mean L1 to the closest real piano hum
(lower is better), with the song hum as-is as the starting point:
  song hum as-is | step 2 with step 1's differences | step 2 with NO differences
  | step 1's differences vs the true ones | human vs human (multi-arrangement units)
plus pitch-activity F1 against the real piano hum (a measurement only).

    python -m app.ml.spectral.hum2hum --maps /scratch/.../hum_maps --units m2m_units.json --ckpt-dir <dir>
"""
from __future__ import annotations

import argparse
import collections
import copy
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

from .model import Block, PATCH_H, PATCH_H_STEP, PATCH_W, ROWS, F_PAD
from .train import log_to

RMAX = 4
NB = 264
ACTIVE = 0.15          # log-space level counted as "sounding" (map amplitude ~0.35; hums peak ~20)
SCALE = 2.0            # log1p(map) / SCALE: hum maps peak ~20 -> ~1.5


def to_log(m):
    return torch.log1p(m.float().clamp(min=0)) / SCALE


class HumNet(nn.Module):
    """Pitch-aligned patch transformer (time then pitch attention), map in -> map out, from scratch."""

    def __init__(self, cin, cout, d=192, depth=4, heads=4, stem=32, residual=False):
        super().__init__()
        self.residual = residual
        self.stem = nn.Sequential(nn.Conv2d(cin, stem, 3, padding=1), nn.GELU(),
                                  nn.Conv2d(stem, stem, 3, padding=1), nn.GELU())
        self.patch = nn.Conv2d(stem, d, (PATCH_H, PATCH_W), stride=(PATCH_H_STEP, PATCH_W))
        self.pitch_emb = nn.Parameter(torch.randn(1, ROWS, 1, d) * 0.02)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(d)
        self.unpatch = nn.ConvTranspose2d(d, stem, (PATCH_H, PATCH_W), stride=(PATCH_H_STEP, PATCH_W))
        self.fuse = nn.Sequential(nn.Conv2d(2 * stem, stem, 3, padding=1), nn.GELU())
        self.head = nn.Conv2d(stem, cout, 1)
        if residual:                       # step 2 starts as the song hum itself and learns edits to it
            nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, x):                                   # (B, cin, 264, T) -> (B, cout, 264, T), >= 0
        B, _, Fb, T = x.shape
        s = self.stem(x)
        Tp = math.ceil(T / PATCH_W) * PATCH_W
        z = self.patch(F.pad(s, (0, Tp - T, 0, F_PAD - Fb))).permute(0, 2, 3, 1) + self.pitch_emb
        for b in self.blocks:
            z = b(z)
        u = self.unpatch(self.norm(z).permute(0, 3, 1, 2))[:, :, :Fb, :T]
        y = self.head(self.fuse(torch.cat([u, s], 1)))
        if self.residual:                  # input channel 0 is the song hum: output = song hum + learned edit
            return F.relu(x[:, :1] + y)
        return F.softplus(y)


def index(root):
    idx = collections.defaultdict(dict)
    for p in pathlib.Path(root).glob("*/*.npz"):
        if not p.name.endswith(".tmp.npz"):
            sid, w = p.stem.rsplit("_", 1); idx[sid][int(w)] = p
    return idx


class Units(Dataset):
    """Item: one unit -> a (1,264,T), b (RMAX,264,T), mask (RMAX).  Train: one random crop of a shared window."""

    def __init__(self, units, idx, train=True, crop=688, max_windows=4):
        self.units = [u for u in units if all(i in idx for i in u)]
        self.idx, self.train, self.crop, self.maxw = idx, train, crop, max_windows

    def __len__(self):
        return len(self.units)

    def windows(self, u):
        return sorted(set.intersection(*(set(self.idx[i]) for i in u)))

    def load(self, u, w, f0=0, L=None):
        refs = u[:RMAX]
        z0 = np.load(self.idx[refs[0]][w]); a = z0["a"]
        L = L or a.shape[1]
        A = np.zeros((1, NB, L), np.float32); Bm = np.zeros((RMAX, NB, L), np.float32)
        x = a[:, f0:f0 + L]; A[0, :, :x.shape[1]] = x
        for r, i in enumerate(refs):
            b = (z0 if r == 0 else np.load(self.idx[i][w]))["b"][:, f0:f0 + L]
            Bm[r, :, :b.shape[1]] = b
        mask = np.zeros(RMAX, np.float32); mask[:len(refs)] = 1
        return torch.from_numpy(A), torch.from_numpy(Bm), torch.from_numpy(mask)

    def __getitem__(self, j):
        u = list(self.units[j]); random.shuffle(u)
        w = random.choice(self.windows(u))
        T = np.load(self.idx[u[0]][w], mmap_mode=None)["a"].shape[1]
        f0 = random.randrange(0, max(1, T - self.crop))
        return self.load(u, w, f0, self.crop)


def stack(batch, dev):
    A = torch.stack([b[0] for b in batch]).to(dev); Bm = torch.stack([b[1] for b in batch]).to(dev)
    M = torch.stack([b[2] for b in batch]).to(dev)
    return to_log(A), to_log(Bm), M                          # (U,1,264,T), (U,R,264,T), (U,R)


def soft_on(x):
    """Soft "is this pitch sounding" (0..1), spread over a semitone and +-2 frames so near-misses still overlap."""
    on = torch.sigmoid((x - ACTIVE) / 0.03)
    sh = on.shape
    return F.max_pool2d(on.reshape(-1, 1, *sh[-2:]), (3, 5), 1, (1, 2)).reshape(sh)


def closest(pred, b, mask, active_w=0.0, dice_w=1.0):
    """Per-unit distance to the closest real piano hum: (U,), and which ref.  Pure hum-to-hum.

    Hums are ~94% silence and a song hum shares only ~10-15% of its sounding
    cells with the piano hum, so plain L1 is minimised by predicting SILENCE.
    The distance is therefore  L1 (weighted on sounding cells)  +  dice_w x
    (1 - soft Dice of where each hum is sounding): silence scores the worst
    possible Dice, so the model has to put sound where the piano hum has it."""
    e = (pred - b).abs()                                                   # (U,R,264,T)
    if active_w:
        e = e * (1 + active_w * ((b > ACTIVE) | (pred > ACTIVE)).float())
    l = e.mean((-1, -2))
    if dice_w:
        sp, sb = soft_on(pred), soft_on(b)
        dice = 2 * torch.minimum(sp, sb).sum((-1, -2)) / (sp.sum((-1, -2)) + sb.sum((-1, -2)) + 1e-6)   # fuzzy Dice: 1 for identical hums
        l = l + dice_w * (1 - dice)
    l = l + (1 - mask) * 1e9
    v, k = l.min(1)
    return v, k


RECIPE0 = dict(low=1.0, mid=1.0, high=1.0, drop=1.0, add=1.0, beta=0.0)


def recipe_weights(rc, dev):
    """(264,) pitch-band weights from a recipe: low < C3 (bin 81), mid C3-C6, high > C6 (bin 189)."""
    w = torch.ones(NB, device=dev)
    w[:81] = rc["low"]; w[81:189] = rc["mid"]; w[189:] = rc["high"]
    return w


def random_recipe(cur, rng):
    r = dict(cur)
    for k in ("low", "mid", "high", "drop", "add"):
        r[k] = float(np.clip(r[k] * math.exp(rng.gauss(0, 0.5)), 0.25, 4.0))
    r["beta"] = float(np.clip(r["beta"] + (abs(rng.gauss(0, 1.5)) if r["beta"] == 0 else rng.gauss(0, 1.0)), 0.0, 6.0))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", required=True); ap.add_argument("--units", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--d", type=int, default=192); ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--s1-pretrain", type=int, default=4000)
    ap.add_argument("--a-eval", type=int, default=500, help="phase A: evaluate every N steps")
    ap.add_argument("--a-patience", type=int, default=3); ap.add_argument("--a-max", type=int, default=12000)
    ap.add_argument("--b-steps", type=int, default=1500, help="step 1 study steps per candidate")
    ap.add_argument("--adapt", type=int, default=800, help="step 2 re-adapt steps per candidate")
    ap.add_argument("--candidates", type=int, default=3); ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--patience", type=int, default=2); ap.add_argument("--sideways", type=float, default=0.2)
    ap.add_argument("--hc-units", type=int, default=50, help="held-out units judging the hill climb")
    ap.add_argument("--active-weight", type=float, default=9.0, help="extra weight on sounding cells")
    ap.add_argument("--workers", type=int, default=7); ap.add_argument("--max-minutes", type=float, default=470)
    ap.add_argument("--limit-units", type=int, default=0)
    a = ap.parse_args()
    ck = pathlib.Path(os.path.expanduser(a.ckpt_dir)); ck.mkdir(parents=True, exist_ok=True)
    say = log_to(ck / "train.log"); t0 = time.time(); dev = "cuda"
    rng = random.Random(0)

    U = json.loads(pathlib.Path(a.units).read_text())
    idx = index(a.maps)
    lim = a.limit_units or None
    tr = Units(U["units"]["train"][:lim], idx, True)
    va_all = Units(U["units"]["validation"][:lim], idx, False)
    hc = Units(va_all.units[: a.hc_units], idx, False)          # judges the hill climb
    say(f"hum2hum: units {len(tr)} train, {len(va_all)} validation ({len(hc.units)} judge the hill climb); "
        f"multi-arrangement: {sum(len(u) > 1 for u in tr.units)} train / {sum(len(u) > 1 for u in va_all.units)} val")

    s1, s2 = HumNet(1, 2, a.d, a.depth).to(dev), HumNet(3, 1, a.d, a.depth, residual=True).to(dev)
    say(f"step 1 {sum(p.numel() for p in s1.parameters()) / 1e6:.2f}M, step 2 {sum(p.numel() for p in s2.parameters()) / 1e6:.2f}M params, from scratch")
    o1, o2 = torch.optim.AdamW(s1.parameters(), lr=a.lr), torch.optim.AdamW(s2.parameters(), lr=a.lr)
    dl = DataLoader(tr, batch_size=a.batch, shuffle=True, drop_last=True, collate_fn=lambda b: b,
                    num_workers=a.workers, persistent_workers=a.workers > 0)
    it = iter(dl)

    def batch():
        nonlocal it
        try:
            return stack(next(it), dev)
        except StopIteration:
            it = iter(dl); return stack(next(it), dev)

    state = dict(round=0, phase="s1_pretrain", recipe=RECIPE0, best=float("inf"), no_gain=0, history=[])
    if (ck / "state.pt").exists():
        c = torch.load(ck / "state.pt", map_location=dev, weights_only=False)
        s1.load_state_dict(c["s1"]); s2.load_state_dict(c["s2"]); o1.load_state_dict(c["o1"]); o2.load_state_dict(c["o2"])
        state = c["state"]; say(f"resumed: round {state['round']}, phase {state['phase']}")

    def save():
        t = ck / "state.tmp"
        torch.save({"s1": s1.state_dict(), "s2": s2.state_dict(), "o1": o1.state_dict(), "o2": o2.state_dict(),
                    "state": state, "args": vars(a)}, t)
        t.replace(ck / "state.pt")

    def out_of_time():
        if (time.time() - t0) / 60 > a.max_minutes:
            save(); say("time limit: saved state, resubmit to resume"); return True
        return False

    def s1_step(net1, opt, rc, net2=None):
        A, Bm, M = batch()
        with torch.no_grad():
            if net2 is not None and rc["beta"] > 0:          # where step 2's piano hum is still wrong
                d = net1(A); p = net2(torch.cat([A, d], 1))
                _, k = closest(p, Bm, M)
                b = Bm[torch.arange(len(k)), k][:, None]
                err = (p - b).abs(); err = err / (err.mean((-1, -2), keepdim=True) + 1e-6)
                wmap = 1 + rc["beta"] * err.clamp(max=5)
            else:
                d = net1(A); p = None
                k = torch.zeros(len(A), dtype=torch.long, device=dev); wmap = 1.0
            if p is None:                                    # no step 2 yet: closest ref by the song hum itself
                _, k = closest(A, Bm, M)
            b = Bm[torch.arange(len(k)), k][:, None]
            dt = torch.cat([F.relu(A - b), F.relu(b - A)], 1)            # true A-B, B-A for that ref
        band = recipe_weights(rc, dev)[None, None, :, None]
        ch = torch.tensor([rc["drop"], rc["add"]], device=dev)[None, :, None, None]
        act = 1 + a.active_weight * (dt > ACTIVE / 2).float()
        loss = ((net1(A) - dt).abs() * band * ch * wmap * act).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(net1.parameters(), 1.0); opt.step()
        return float(loss)

    def s2_step(net1, net2, opt):
        A, Bm, M = batch()
        with torch.no_grad():
            d = net1(A)                                      # step 1's PREDICTED differences only
        v, _ = closest(net2(torch.cat([A, d], 1)), Bm, M, a.active_weight)
        loss = v.mean()
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(net2.parameters(), 1.0); opt.step()
        return float(loss)

    @torch.no_grad()
    def evaluate(net1, net2, ds, full=False):
        net1.eval(); net2.eval(); R = collections.defaultdict(list)
        for u in ds.units:
            for w in ds.windows(u)[:: max(1, len(ds.windows(u)) // ds.maxw)][: ds.maxw]:
                A, Bm, M = ds.load(u, w)
                A, Bm, M = to_log(A[None].to(dev)), to_log(Bm[None].to(dev)), M[None].to(dev)
                d = net1(A); p = net2(torch.cat([A, d], 1))
                v, k = closest(p, Bm, M, a.active_weight); R["pred"].append(float(v))
                if not full:
                    continue
                R["song as-is"].append(float(closest(A, Bm, M, a.active_weight)[0]))
                R["no differences"].append(float(closest(net2(torch.cat([A, torch.zeros_like(d)], 1)), Bm, M, a.active_weight)[0]))
                b = Bm[0, k]
                R["diff error"].append(float((d - torch.cat([F.relu(A - b[:, None]), F.relu(b[:, None] - A)], 1)).abs().mean()))
                on_p, on_b = p[0, 0] > 0.15, b[0] > 0.15
                tp = float((on_p & on_b).sum()); R["activity F1"].append(2 * tp / max(1.0, float(on_p.sum() + on_b.sum())))
                nr = int(M.sum())
                if nr > 1:
                    R["human vs human"].append(float(np.mean([
                        closest(Bm[:, i:i + 1], Bm[:, j:j + 1], torch.ones(1, 1, device=dev), a.active_weight)[0].item()
                        for i in range(nr) for j in range(nr) if i != j])))
        net1.train(); net2.train()
        return {k: float(np.mean(v)) for k, v in R.items()}

    def report(tag):
        r = evaluate(s1, s2, va_all, full=True)
        say(f"VAL {tag}: distance to closest piano hum (lower = better)  song as-is {r['song as-is']:.4f} | "
            f"step 2 with step 1's differences {r['pred']:.4f} | step 2 with no differences {r['no differences']:.4f} | "
            f"human vs human {r.get('human vs human', float('nan')):.4f}  ·  step 1 differences error {r['diff error']:.4f}  ·  "
            f"activity F1 {r['activity F1']:.3f}")
        state["history"].append({"tag": tag, **r}); return r

    def phase_a():
        """Step 2 on step 1's differences until its score plateaus."""
        best, bad, n = float("inf"), 0, 0
        while n < a.a_max:
            for _ in range(a.a_eval):
                s2_step(s1, s2, o2); n += 1
            v = evaluate(s1, s2, hc)["pred"]
            say(f"  phase A step {n}: held-out distance {v:.4f}")
            if v < best * 0.995:
                best, bad = v, 0
            else:
                bad += 1
            if bad >= a.a_patience or out_of_time():
                break
        return best

    if state["phase"] == "s1_pretrain":
        for i in range(a.s1_pretrain):
            l = s1_step(s1, o1, RECIPE0)
            if (i + 1) % 500 == 0:
                say(f"  step 1 pretrain {i + 1}: loss {l:.4f}")
        state["phase"] = "A"; save()
    if state["phase"] == "A":
        state["best"] = phase_a(); r = report(f"round {state['round']}"); state["phase"] = "B"
        torch.save({"s1": s1.state_dict(), "s2": s2.state_dict(), "args": vars(a), "round": state["round"],
                    "recipe": state["recipe"], "val": r}, ck / "best.pt")
        save()
        if out_of_time():
            return
    while state["phase"] == "B" and state["round"] < a.rounds and state["no_gain"] < a.patience:
        state["round"] += 1
        cur = evaluate(s1, s2, hc)["pred"]
        cands = []
        for c in range(a.candidates):
            rc = random_recipe(state["recipe"], rng)
            n1, n2 = copy.deepcopy(s1), copy.deepcopy(s2)
            q1 = torch.optim.AdamW(n1.parameters(), lr=a.lr); q1.load_state_dict(o1.state_dict())
            q2 = torch.optim.AdamW(n2.parameters(), lr=a.lr); q2.load_state_dict(o2.state_dict())
            for _ in range(a.b_steps):
                s1_step(n1, q1, rc, s2)                      # step 1 studies the TRUE differences, recipe-weighted
            for _ in range(a.adapt):
                s2_step(n1, n2, q2)                          # step 2 re-adapts to the new differences
            v = evaluate(n1, n2, hc)["pred"]
            say(f"  round {state['round']} candidate {c + 1}: held-out distance {v:.4f} (current {cur:.4f})  recipe "
                + ", ".join(f"{k} {x:.2f}" for k, x in rc.items()))
            cands.append((v, rc, n1, n2, q1, q2))
            if out_of_time():
                return
        v, rc, n1, n2, q1, q2 = min(cands, key=lambda x: x[0])
        if v < cur or rng.random() < a.sideways:
            s1.load_state_dict(n1.state_dict()); s2.load_state_dict(n2.state_dict())
            o1.load_state_dict(q1.state_dict()); o2.load_state_dict(q2.state_dict())
            state["recipe"] = rc
            say(f"  round {state['round']}: kept candidate ({v:.4f} vs {cur:.4f}{', sideways step' if v >= cur else ''})")
        else:
            say(f"  round {state['round']}: no candidate beat the current model; kept it")
        b = phase_a()
        r = report(f"round {state['round']}")
        if b < state["best"] * 0.99:
            state["best"], state["no_gain"] = b, 0
            torch.save({"s1": s1.state_dict(), "s2": s2.state_dict(), "args": vars(a), "round": state["round"],
                        "recipe": state["recipe"], "val": r}, ck / "best.pt")
        else:
            state["no_gain"] += 1
        save()
        if out_of_time():
            return
    if not (ck / "best.pt").exists():
        torch.save({"s1": s1.state_dict(), "s2": s2.state_dict(), "args": vars(a), "round": state["round"],
                    "recipe": state["recipe"]}, ck / "best.pt")
    state["phase"] = "done"; save()
    say(f"done after round {state['round']}: best held-out distance {state['best']:.4f}; recipe {state['recipe']}")


if __name__ == "__main__":
    main()
