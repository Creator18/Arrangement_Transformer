"""Stage 6 of the modular design: a GATE that sets the effect adapters' switches from the audio.

Stage 5 (joint_refine.py) refined the 22 adapters with each clip's switches KNOWN (the render's effect list).
Real songs come without one, so this network listens and sets them:

  features (the transcriber's own input)  ->  conv front end, pitch axis collapsed
  ->  time tokens (~16 per second)  ->  attention layers ALTERNATING local (each token sees +-2 s) and
  global (the whole window)  ->  attention pooling  ->  one switch per adapter, in [0, 1]

Training, two phases:
  classify   switches = the render's effects (BCE).  Learns to NAME what it hears.
  e2e        the gate's own switches drive the frozen joint model; loss = transcription + a little BCE.
             Learns the switches that give the best NOTES, which is what matters on real songs (where
             "the right effect" has no ground truth).

Validation: per-effect detection (accuracy at 0.5, and how many clips each effect is in), and note F1 on
the combination clips with the gate's switches vs the known switches vs none.

    python -m app.ml.spectral.gate --joint <joint/best.pt> --renders /scratch/... --ckpt-dir <dir> --phase classify
    python -m app.ml.spectral.gate ... --phase e2e --init <dir>/best.pt
"""
from __future__ import annotations

import argparse
import math
import os
import pathlib
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from . import frontend as S
from .data import clips
from .ft_aux import trunk
from .joint_refine import Gate, GateClips, from_joint
from .train import log_to, note_f1


class AltBlock(nn.Module):
    """Pre-norm attention + MLP; `local` restricts attention to +-window tokens."""

    def __init__(self, d, heads, local, window):
        super().__init__()
        self.local, self.window = local, window
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, z):
        mask = None
        if self.local:
            n = z.shape[1]; i = torch.arange(n, device=z.device)
            mask = (i[None, :] - i[:, None]).abs() > self.window          # True = not allowed
        h = self.n1(z)
        z = z + self.att(h, h, h, attn_mask=mask, need_weights=False)[0]
        return z + self.ff(self.n2(z))


class GateNet(nn.Module):
    def __init__(self, K, d=192, layers=6, heads=4, window=32, in_ch=7):
        super().__init__()
        self.front = nn.Sequential(                                        # (B, 7, 264, T) -> (B, d, 1, T/4)
            nn.Conv2d(in_ch, 32, 3, stride=(2, 1), padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=(2, 2), padding=1), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=(2, 2), padding=1), nn.GELU(),
            nn.Conv2d(128, d, (33, 1)), nn.GELU())                          # 264/8 = 33 pitch rows -> 1
        self.blocks = nn.ModuleList([AltBlock(d, heads, local=(i % 2 == 0), window=window) for i in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.pool = nn.Linear(d, 1)                                        # attention pooling over time
        self.out = nn.Linear(d, K)
        self.d = d

    def forward(self, x):
        z = self.front(x).squeeze(2).transpose(1, 2)                       # (B, T', d)
        n = z.shape[1]; pos = torch.arange(n, device=z.device)[:, None]
        k = torch.exp(torch.arange(0, self.d, 2, device=z.device) * (-math.log(10000.0) / self.d))
        pe = torch.zeros(n, self.d, device=z.device); pe[:, 0::2] = torch.sin(pos * k); pe[:, 1::2] = torch.cos(pos * k)
        z = z + pe.to(z.dtype)
        for b in self.blocks:
            z = b(z)
        z = self.norm(z)
        w = torch.softmax(self.pool(z).squeeze(-1), dim=1)                 # (B, T')
        return self.out((w.unsqueeze(-1) * z).sum(1))                      # (B, K) logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint", required=True, help="stage-5 joint checkpoint (joint/best.pt)")
    ap.add_argument("--renders", required=True, help="holds mod_<m>/, mod_plain/, mod_combo/")
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--phase", choices=["classify", "e2e"], default="classify")
    ap.add_argument("--init", default="", help="e2e: start from this gate checkpoint (the classify phase)")
    ap.add_argument("--mix", default="0.5,0.4,0.1", help="share of combination / single-module / plain clips")
    ap.add_argument("--bce-weight", type=float, default=0.1, help="e2e: weight of the effect labels")
    ap.add_argument("--crop", type=float, default=8.0); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--total-steps", type=int, default=20000); ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--val-clips", type=int, default=400); ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--min-delta", type=float, default=0.001)
    ap.add_argument("--workers", type=int, default=max(2, int(os.environ.get("SLURM_CPUS_PER_TASK", "8")) - 1))
    ap.add_argument("--max-minutes", type=float, default=460)
    a = ap.parse_args()
    ckdir = pathlib.Path(os.path.expanduser(a.ckpt_dir)); ckdir.mkdir(parents=True, exist_ok=True)
    say = log_to(ckdir / "train.log"); t0 = time.time(); dev = "cuda"
    bf16 = torch.cuda.get_device_capability(0)[0] >= 8; amp = torch.bfloat16 if bf16 else torch.float16
    jck = torch.load(os.path.expanduser(a.joint), map_location="cpu", weights_only=False)
    modules = jck["modules"]; K = len(modules)
    joint = from_joint(jck).to(dev).eval()
    for p in joint.parameters():
        p.requires_grad_(False)
    fe = S.SpectralFrontend().to(dev).eval(); fe.mu.copy_(jck["fe_mu"].to(dev)); fe.sd.copy_(jck["fe_sd"].to(dev))
    gate = GateNet(K).to(dev)
    if a.init:
        gate.load_state_dict(torch.load(a.init, map_location=dev, weights_only=False)["gate"])
    say(f"gate: phase {a.phase}; {sum(p.numel() for p in gate.parameters()) / 1e6:.2f}M params; {K} adapters; "
        f"GPU {torch.cuda.get_device_name(0)}")

    R = pathlib.Path(a.renders)
    combo_tr, plain_tr = clips(R / "mod_combo", "train"), clips(R / "mod_plain", "train")
    single_tr = {m: clips(R / f"mod_{m}", "train") for m in modules}
    paths = combo_tr + plain_tr + [p for m in modules for p in single_tr[m]]
    sh = [float(x) for x in a.mix.split(",")]
    w = ([sh[0] / len(combo_tr)] * len(combo_tr) + [sh[2] / len(plain_tr)] * len(plain_tr)
         + [sh[1] / K / len(single_tr[m]) for m in modules for _ in single_tr[m]])
    sampler = WeightedRandomSampler(torch.tensor(w, dtype=torch.double), num_samples=len(paths), replacement=True)
    dl = DataLoader(GateClips(paths, modules, a.crop), batch_size=a.batch, sampler=sampler, num_workers=a.workers,
                    pin_memory=True, drop_last=True, persistent_workers=True, prefetch_factor=6)
    rv = random.Random(0)
    # validation: combination clips + an even spread of single-module and plain clips
    vc = rv.sample(clips(R / "mod_combo", "validation"), min(a.val_clips, len(clips(R / "mod_combo", "validation"))))
    vs = [p for m in modules for p in rv.sample(clips(R / f"mod_{m}", "validation"), 10)] + rv.sample(clips(R / "mod_plain", "validation"), 40)
    say(f"clips: combination {len(combo_tr)}, plain {len(plain_tr)}, single-module {sum(map(len, single_tr.values()))}; "
        f"validation {len(vc)} combination + {len(vs)} single/plain")

    def run_joint(x, g):
        Gate.g = g
        with torch.autocast("cuda", dtype=amp):
            o = joint.head(trunk(joint, x)).permute(1, 0, 3, 2)
        Gate.g = None
        return o

    def validate():
        gate.eval(); P = []; Y = []; F1 = {"gate": ([], []), "known": ([], []), "none": ([], [])}; Gt = ([], [])
        with torch.no_grad():
            for wav, on, fr, _v, _h, _t, _m, g in DataLoader(GateClips(vc + vs, modules, 20.0, train=False), batch_size=4,
                                                             num_workers=min(4, a.workers)):
                x = fe(wav.to(dev))
                with torch.autocast("cuda", dtype=amp):
                    lg = gate(x).float()
                P.append(torch.sigmoid(lg).cpu()); Y.append(g)
            Pc, Yc = torch.cat(P), torch.cat(Y)
            # transcription with each kind of switches, on the combination clips only
            for wav, on, fr, _v, _h, _t, _m, g in DataLoader(GateClips(vc[:200], modules, 20.0, train=False), batch_size=4,
                                                             num_workers=min(4, a.workers)):
                x = fe(wav.to(dev))
                with torch.autocast("cuda", dtype=amp):
                    pg = torch.sigmoid(gate(x).float())
                for name, sw in (("gate", pg), ("known", g.to(dev)), ("none", torch.zeros_like(pg))):
                    o = run_joint(x, sw)
                    F1[name][0].append(torch.sigmoid(o[0].float()).cpu().numpy()); F1[name][1].append(torch.sigmoid(o[1].float()).cpu().numpy())
                Gt[0].append(on.transpose(1, 2).numpy()); Gt[1].append(fr.transpose(1, 2).numpy())
        gate.train()
        hit = (Pc > 0.5).float()
        tpr = (hit * Yc).sum(0) / Yc.sum(0).clamp(min=1); tnr = ((1 - hit) * (1 - Yc)).sum(0) / (1 - Yc).sum(0).clamp(min=1)
        acc = (tpr + tnr) / 2                                                # balanced: always-off scores 0.5
        f1 = {k: note_f1(np.concatenate(v[0]), np.concatenate(v[1]), np.concatenate(Gt[0]), np.concatenate(Gt[1]))[0]["conp_f1"]
              for k, v in F1.items()}
        return acc, Yc.mean(0), f1

    params = list(gate.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / a.warmup, 0.5 * (
        1 + math.cos(math.pi * min(1.0, max(0, s - a.warmup) / max(1, a.total_steps - a.warmup))))))
    step, best, stale = 0, -1.0, 0; last = ckdir / "last.pt"
    if last.exists():
        c = torch.load(last, map_location=dev, weights_only=False)
        gate.load_state_dict(c["gate"]); opt.load_state_dict(c["opt"]); sched.load_state_dict(c["sched"])
        step, best, stale = c["step"], c["best"], c["stale"]; say(f"resumed at step {step}")

    def save(path):
        tmp = path.with_suffix(".tmp")
        torch.save({"gate": gate.state_dict(), "modules": modules, "joint": a.joint, "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "step": step, "best": best, "stale": stale, "args": vars(a),
                    "kind": "gate"}, tmp)
        tmp.replace(path)

    def report():
        acc, share, f1 = validate()
        score = float(acc.mean()) if a.phase == "classify" else f1["gate"]
        say(f"VAL step {step}: detection (balanced) {float(acc.mean()):.3f}  |  combination note F1: gate {f1['gate']:.3f}  "
            f"known {f1['known']:.3f}  none {f1['none']:.3f}")
        say("    " + "  ".join(f"{m} {float(acc[k]):.2f}/{float(share[k]):.2f}" for k, m in enumerate(modules)))
        return score

    if step == 0:
        best = report(); save(ckdir / "best.pt")
        say("    (per effect: balanced detection accuracy / share of validation clips that have it)")
    pw_on, pw_fr = torch.tensor(30.0, device=dev), torch.tensor(2.0, device=dev)
    # share of drawn clips that have each effect: combination clips carry ~3.5 of K, single clips one
    share = (sh[0] * 3.5 / K + sh[1] / K); pos_w = torch.full((K,), (1 - share) / share, device=dev)
    run, n, t_log, wait, t_fetch = np.zeros(2), 0, time.time(), 0.0, time.time(); gate.train()
    while step < a.total_steps:
        for wav, on, fr, vel, hv, _t, _m, g in dl:
            wait += time.time() - t_fetch
            with torch.no_grad():
                x = fe(wav.to(dev, non_blocking=True))
            g = g.to(dev)
            with torch.autocast("cuda", dtype=amp):
                lg = gate(x).float()
            l_bce = F.binary_cross_entropy_with_logits(lg, g, pos_weight=pos_w)     # each effect is in few clips
            if a.phase == "classify":
                loss, l_task = l_bce, torch.zeros((), device=dev)
            else:
                onT, frT = on.to(dev).transpose(1, 2), fr.to(dev).transpose(1, 2)
                o = run_joint(x, torch.sigmoid(lg))
                l_task = (F.binary_cross_entropy_with_logits(o[0].float(), onT, pos_weight=pw_on)
                          + F.binary_cross_entropy_with_logits(o[1].float(), frT, pos_weight=pw_fr))
                loss = l_task + a.bce_weight * l_bce
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); step += 1
            run += [float(l_bce), float(l_task)]; n += 1
            if step % 100 == 0:
                dt = time.time() - t_log
                say(f"step {step:>6}  bce {run[0] / n:.4f}  task {run[1] / n:.4f}  {100 / dt:.2f} steps/s  data wait {100 * wait / dt:.0f}%")
                run, n, t_log, wait = np.zeros(2), 0, time.time(), 0.0
            if step % a.val_every == 0 or step >= a.total_steps:
                score = report()
                stale = 0 if score > best + a.min_delta else stale + 1
                if score > best:
                    best = score; save(ckdir / "best.pt")
                save(last)
                if a.patience and stale >= a.patience:
                    say(f"stopping at step {step} (finished): plateaued for {stale} validations; best {best:.4f}")
                    return
            if (time.time() - t0) / 60 > a.max_minutes:
                save(last); say("time limit: saved last.pt, resubmit to resume"); return
            if step >= a.total_steps:
                break
            t_fetch = time.time()
    save(last); say(f"stopping at step {step} (finished); best {best:.4f}")


if __name__ == "__main__":
    main()
