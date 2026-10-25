"""Stage 5 of the modular design: JOINT REFINEMENT ("bundle adjustment") of the per-effect adapters.

Each module adapter (checkpoints/mod_<m>/best.pt) was trained alone on plain_base with exactly one effect.
Real songs have many effects at once, so here all of them sit side by side on the frozen plain_base and
are refined TOGETHER:

  W x  +  sum_k g_k * (alpha/r) B_k A_k x            every adapted linear layer
  head(h)  +  sum_k g_k * dhead_k(h)                  the output head (each module also tuned it)

g is a per-clip switch: 1 for each effect the clip was rendered with, 0 otherwise.  Here g is KNOWN (the
render says which effects it used); stage 6's gate learns to set it from the audio.

The training mix keeps each adapter honest on its own while teaching them to add up:
  combination renders (2-5 effects at once, eval/render_modular.py --module combo)   g = those effects
  each module's own renders                                                          g = that one module
  plain renders                                                                      g = 0 (= plain_base)
and a pull toward each adapter's separately trained weights (the prior; --prior-weight) so no module
drifts far from what it learned alone.  Validation, at step 0 and every --val-every steps: every module
on its own clips (one switch on) and the combination clips (their switches on), so a module the joint
fit hurts is visible against its own step-0 number.

    python -m app.ml.spectral.joint_refine --base plain_base.pt --modules-dir <checkpoints> \\
        --renders /scratch/.../ (mod_<m>/, mod_plain/, mod_combo/) --ckpt-dir <dir>
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
from torch.utils.data import DataLoader, WeightedRandomSampler

from . import frontend as S
from .data import clips
from .ft_aux import Clips, trunk
from .model import from_checkpoint
from .train import log_to, note_f1

MODULES = ["timbre", "reverb", "eq", "dynamics", "levels", "exposed", "articulation", "connected", "noteend",
           "decay", "expression", "late", "fast", "glides", "vibrato", "rolls", "unison", "drums", "heavydrums",
           "sidechain", "genre", "singer"]


class Gate:
    """The per-clip switches (B, K) for the forward pass in progress (set before each forward)."""
    g: torch.Tensor | None = None


def rows_gate(g, rows):
    """(B, K) -> (rows, K): the blocks fold the batch with time or pitch, always sample-major."""
    return g if rows == g.shape[0] else g.repeat_interleave(rows // g.shape[0], 0)


class MultiLoRALinear(nn.Module):
    """W x + scale * sum_k g_k B_k A_k x: K adapters side by side, frozen base."""

    def __init__(self, base: nn.Linear, A: torch.Tensor, B: torch.Tensor, scale: float):
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad_(False)
        self.K, self.r = A.shape[0], A.shape[1]
        self.A = nn.Parameter(A.clone())                 # (K, r, in)
        self.B = nn.Parameter(B.clone())                 # (K, out, r)
        self.scale = scale

    def forward(self, x):
        y = self.base(x)
        g = Gate.g
        if g is None:
            return y
        lead = x.shape[:-1]
        xa = x.reshape(lead[0], -1, x.shape[-1]) @ self.A.reshape(self.K * self.r, -1).t().to(x.dtype)   # (rows, n, K r)
        gw = rows_gate(g, lead[0]).to(x.dtype).repeat_interleave(self.r, 1).unsqueeze(1)                 # (rows, 1, K r)
        Bt = self.B.permute(0, 2, 1).reshape(self.K * self.r, -1).to(x.dtype)                           # (K r, out)
        return y + ((xa * gw) @ Bt * self.scale).reshape(*lead, -1)


class MultiHead(nn.Module):
    """head(h) + sum_k g_k dhead_k(h); dhead_k = module k's tuned head minus plain_base's."""

    def __init__(self, base: nn.Conv2d, dW: torch.Tensor, db: torch.Tensor):
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad_(False)
        self.K, self.c = dW.shape[0], dW.shape[1]
        self.dW = nn.Parameter(dW.clone())               # (K, 3, C, 3, 1)
        self.db = nn.Parameter(db.clone())               # (K, 3)

    def forward(self, h):
        y = self.base(h)
        g = Gate.g
        if g is None:
            return y
        d = F.conv2d(h, self.dW.reshape(self.K * self.c, *self.dW.shape[2:]).to(h.dtype), self.db.reshape(-1).to(h.dtype),
                     stride=self.base.stride)                                                  # (B, K c, F, T)
        d = d.view(h.shape[0], self.K, self.c, *d.shape[2:])
        return y + (d * g.to(h.dtype).view(h.shape[0], self.K, 1, 1, 1)).sum(1)


def build(base_ck, modules_dir, modules, dev):
    """plain_base with every module's adapter side by side (initialised to their separate training)."""
    net = from_checkpoint(base_ck)
    for p in net.parameters():
        p.requires_grad_(False)
    sds = []
    for m in modules:
        c = torch.load(pathlib.Path(modules_dir) / f"mod_{m}" / "best.pt", map_location="cpu", weights_only=False)
        assert c.get("lora"), f"{m}: not a LoRA checkpoint"
        sds.append((c["model"], c["lora"]))
    r = sds[0][1]
    targets = []
    for bi, blk in enumerate(net.blocks):
        for an in ("time", "pitch"):
            att = getattr(blk, an)
            targets += [(att, "qkv", f"blocks.{bi}.{an}.qkv"), (att, "out", f"blocks.{bi}.{an}.out")]
        targets += [(blk.ff, "0", f"blocks.{bi}.ff.0"), (blk.ff, "2", f"blocks.{bi}.ff.2")]
    for parent, name, key in targets:
        base = parent[int(name)] if isinstance(parent, nn.Sequential) else getattr(parent, name)
        A = torch.stack([sd[f"{key}.A"] for sd, _ in sds]); B = torch.stack([sd[f"{key}.B"] for sd, _ in sds])
        ml = MultiLoRALinear(base, A, B, 16.0 / r)       # add_lora's alpha 16
        if isinstance(parent, nn.Sequential):
            parent[int(name)] = ml
        else:
            setattr(parent, name, ml)
    hw, hb = base_ck["model"]["head.weight"], base_ck["model"]["head.bias"]
    dW = torch.stack([sd["head.weight"] - hw for sd, _ in sds]); db = torch.stack([sd["head.bias"] - hb for sd, _ in sds])
    net.head = MultiHead(net.head, dW, db)
    return net.to(dev)


def from_joint(ck):
    """A saved joint checkpoint on its own (no plain_base / module files needed): the side-by-side
    structure is rebuilt from the tensor shapes, then the weights are loaded."""
    from .model import SpectralTranscriber
    net = SpectralTranscriber(**ck["arch"]); sd = ck["model"]
    for bi, blk in enumerate(net.blocks):
        for an in ("time", "pitch"):
            att = getattr(blk, an)
            for nm in ("qkv", "out"):
                A, B = sd[f"blocks.{bi}.{an}.{nm}.A"], sd[f"blocks.{bi}.{an}.{nm}.B"]
                setattr(att, nm, MultiLoRALinear(getattr(att, nm), torch.zeros_like(A), torch.zeros_like(B), 16.0 / A.shape[1]))
        for k in (0, 2):
            A, B = sd[f"blocks.{bi}.ff.{k}.A"], sd[f"blocks.{bi}.ff.{k}.B"]
            blk.ff[k] = MultiLoRALinear(blk.ff[k], torch.zeros_like(A), torch.zeros_like(B), 16.0 / A.shape[1])
    net.head = MultiHead(net.head, torch.zeros_like(sd["head.dW"]), torch.zeros_like(sd["head.db"]))
    net.load_state_dict(sd)
    return net


def switches(modules, spec):
    """'all' | 'off' | 'a,b,c' -> (1, K) switches."""
    g = torch.zeros(1, len(modules))
    if spec == "all":
        g[:] = 1
    elif spec and spec != "off":
        for m in spec.split(","):
            g[0, modules.index(m.strip())] = 1
    return g


class GateClips(Clips):
    """Clips plus each clip's switches: the effects its render used (combo: "a+b+c"; plain: none)."""

    def __init__(self, paths, modules, crop_s=8.0, train=True):
        super().__init__(paths, crop_s, train)
        self.idx = {m: k for k, m in enumerate(modules)}

    def __getitem__(self, i):
        out = super().__getitem__(i)
        g = torch.zeros(len(self.idx))
        mod = json.loads(str(np.load(self.paths[i])["render"])).get("module", "plain")
        for m in mod.split("+"):
            if m in self.idx:
                g[self.idx[m]] = 1.0
        return (*out, g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="plain_base.pt")
    ap.add_argument("--modules-dir", required=True, help="holds mod_<m>/best.pt")
    ap.add_argument("--renders", required=True, help="holds mod_<m>/, mod_plain/, mod_combo/ (each train/ validation/)")
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--modules", default=",".join(MODULES))
    ap.add_argument("--mix", default="0.5,0.4,0.1", help="share of combination / single-module / plain clips")
    ap.add_argument("--prior-weight", type=float, default=10.0,
                    help="pull toward the separately trained adapters: weight x relative squared drift")
    ap.add_argument("--crop", type=float, default=8.0); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--total-steps", type=int, default=15000); ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--val-per-module", type=int, default=40); ap.add_argument("--val-combo", type=int, default=300)
    ap.add_argument("--patience", type=int, default=4); ap.add_argument("--min-delta", type=float, default=0.001)
    ap.add_argument("--workers", type=int, default=max(2, int(os.environ.get("SLURM_CPUS_PER_TASK", "8")) - 1))
    ap.add_argument("--max-minutes", type=float, default=460)
    a = ap.parse_args()
    modules = a.modules.split(","); K = len(modules)
    ckdir = pathlib.Path(os.path.expanduser(a.ckpt_dir)); ckdir.mkdir(parents=True, exist_ok=True)
    say = log_to(ckdir / "train.log"); t0 = time.time(); dev = "cuda"
    bf16 = torch.cuda.get_device_capability(0)[0] >= 8; amp = torch.bfloat16 if bf16 else torch.float16
    base_ck = torch.load(os.path.expanduser(a.base), map_location="cpu", weights_only=False)
    fe = S.SpectralFrontend().to(dev).eval(); fe.mu.copy_(base_ck["fe_mu"].to(dev)); fe.sd.copy_(base_ck["fe_sd"].to(dev))
    net = build(base_ck, os.path.expanduser(a.modules_dir), modules, dev)
    params = [p for p in net.parameters() if p.requires_grad]
    prior = [p.detach().clone() for p in params]
    prior_norm = sum(float((p ** 2).sum()) for p in prior)
    say(f"joint_refine: {K} adapters side by side on {a.base}; trainable {sum(p.numel() for p in params) / 1e6:.2f}M; "
        f"GPU {torch.cuda.get_device_name(0)}")

    R = pathlib.Path(a.renders)
    combo_tr, plain_tr = clips(R / "mod_combo", "train"), clips(R / "mod_plain", "train")
    single_tr = {m: clips(R / f"mod_{m}", "train") for m in modules}
    paths = combo_tr + plain_tr + [p for m in modules for p in single_tr[m]]
    sh = [float(x) for x in a.mix.split(",")]
    w = ([sh[0] / len(combo_tr)] * len(combo_tr) + [sh[2] / len(plain_tr)] * len(plain_tr)
         + [sh[1] / K / len(single_tr[m]) for m in modules for _ in single_tr[m]])
    say(f"clips: combination {len(combo_tr)}, plain {len(plain_tr)}, single-module {sum(map(len, single_tr.values()))}; "
        f"drawn {sh[0]:.0%} / {sh[2]:.0%} / {sh[1]:.0%}")
    sampler = WeightedRandomSampler(torch.tensor(w, dtype=torch.double), num_samples=len(paths), replacement=True)
    dl = DataLoader(GateClips(paths, modules, a.crop), batch_size=a.batch, sampler=sampler, num_workers=a.workers,
                    pin_memory=True, drop_last=True, persistent_workers=True, prefetch_factor=6)
    rv = random.Random(0)
    val_sets = {m: rv.sample(clips(R / f"mod_{m}", "validation"), a.val_per_module) for m in modules}
    vc = clips(R / "mod_combo", "validation"); val_sets["COMBO"] = rv.sample(vc, min(a.val_combo, len(vc)))

    def validate():
        net.eval(); res = {}
        with torch.no_grad():
            for name, vs in val_sets.items():
                P, G = ([], []), ([], [])
                for wav, on, fr, _v, _h, _t, _m, g in DataLoader(GateClips(vs, modules, 20.0, train=False), batch_size=4,
                                                                 num_workers=min(4, a.workers)):
                    x = fe(wav.to(dev)); Gate.g = g.to(dev)
                    with torch.autocast("cuda", dtype=amp):
                        o = net.head(trunk(net, x)).permute(1, 0, 3, 2)
                    P[0].append(torch.sigmoid(o[0].float()).cpu().numpy()); P[1].append(torch.sigmoid(o[1].float()).cpu().numpy())
                    G[0].append(on.transpose(1, 2).numpy()); G[1].append(fr.transpose(1, 2).numpy())
                r = note_f1(np.concatenate(P[0]), np.concatenate(P[1]), np.concatenate(G[0]), np.concatenate(G[1]))
                res[name] = r[0]["conp_f1"]
        Gate.g = None; net.train()
        score = 0.5 * float(np.mean([res[m] for m in modules])) + 0.5 * res["COMBO"]
        return score, res

    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)        # the prior term is the regulariser
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / a.warmup, 0.5 * (
        1 + math.cos(math.pi * min(1.0, max(0, s - a.warmup) / max(1, a.total_steps - a.warmup))))))
    scaler = torch.amp.GradScaler("cuda", enabled=not bf16)
    step, best, stale, ref = 0, -1.0, 0, None; last = ckdir / "last.pt"
    if last.exists():
        c = torch.load(last, map_location=dev, weights_only=False)
        net.load_state_dict(c["model"]); opt.load_state_dict(c["opt"]); sched.load_state_dict(c["sched"])
        step, best, stale, ref = c["step"], c["best"], c["stale"], c["ref"]; say(f"resumed at step {step}")

    def save(path):
        tmp = path.with_suffix(".tmp")
        torch.save({"model": net.state_dict(), "modules": modules, "base": a.base, "arch": net.arch,
                    "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step, "best": best, "stale": stale,
                    "ref": ref, "fe_mu": fe.mu.cpu(), "fe_sd": fe.sd.cpu(), "args": vars(a), "kind": "joint_refine"}, tmp)
        tmp.replace(path)

    def report(score, res):
        say(f"VAL step {step}: score {score:.4f}  combination F1 {res['COMBO']:.3f}"
            + (f" ({res['COMBO'] - ref['COMBO']:+.3f})" if ref else "")
            + f"  mean module F1 {np.mean([res[m] for m in modules]):.3f}")
        say("    " + "  ".join(f"{m} {res[m]:.3f}" + (f"({res[m] - ref[m]:+.3f})" if ref else "") for m in modules))

    if ref is None:                                                      # step 0: the separately trained adapters
        score, ref = validate(); report(score, ref); best = score; save(ckdir / "best.pt")
        say("    (step 0 = each adapter as trained alone; later lines show the change)")
    pw_on, pw_fr = torch.tensor(30.0, device=dev), torch.tensor(2.0, device=dev)
    run, n, t_log, wait, t_fetch = np.zeros(2), 0, time.time(), 0.0, time.time(); net.train()
    while step < a.total_steps:
        for wav, on, fr, vel, hv, _t, _m, g in dl:
            wait += time.time() - t_fetch
            with torch.no_grad():
                x = fe(wav.to(dev, non_blocking=True))
            onT, frT, vlT = on.to(dev).transpose(1, 2), fr.to(dev).transpose(1, 2), vel.to(dev).transpose(1, 2)
            hv = hv.to(dev); Gate.g = g.to(dev)
            with torch.autocast("cuda", dtype=amp):
                o = net.head(trunk(net, x)).permute(1, 0, 3, 2)
            a_, b_, c_ = (t.float() for t in (o[0], o[1], o[2]))
            mv = (onT > 0.5) * hv.view(-1, 1, 1)
            l_vel = (((torch.sigmoid(c_) - vlT) ** 2) * mv).sum() / mv.sum().clamp(min=1)
            l_task = (F.binary_cross_entropy_with_logits(a_, onT, pos_weight=pw_on)
                      + F.binary_cross_entropy_with_logits(b_, frT, pos_weight=pw_fr) + 0.5 * l_vel)
            drift = sum(((p - q) ** 2).sum() for p, q in zip(params, prior)) / prior_norm
            loss = l_task + a.prior_weight * drift
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0); scaler.step(opt); scaler.update(); sched.step(); step += 1
            run += [float(l_task), float(drift)]; n += 1
            if step % 100 == 0:
                dt = time.time() - t_log
                say(f"step {step:>6}  loss {run[0] / n:.4f}  drift {run[1] / n:.2e}  {100 / dt:.2f} steps/s  "
                    f"data wait {100 * wait / dt:.0f}%")
                run, n, t_log, wait = np.zeros(2), 0, time.time(), 0.0
            if step % a.val_every == 0 or step >= a.total_steps:
                score, res = validate(); report(score, res)
                stale = 0 if score > best + a.min_delta else stale + 1
                if score > best:
                    best = score; save(ckdir / "best.pt")
                save(last)
                if a.patience and stale >= a.patience:
                    say(f"stopping at step {step} (finished): score plateaued for {stale} validations; best {best:.4f}")
                    return
            if (time.time() - t0) / 60 > a.max_minutes:
                save(last); say("time limit: saved last.pt, resubmit to resume"); return
            if step >= a.total_steps:
                break
            t_fetch = time.time()
    save(last); say(f"stopping at step {step} (finished); best score {best:.4f}")


if __name__ == "__main__":
    main()
