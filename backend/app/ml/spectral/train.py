"""Train the spectral patch transformer on rendered multi-piano MAESTRO.

    python -m app.ml.spectral.train --data /scratch/basu.ang/maestro_render_multi \\
        --ckpt-dir ~/arrangement_transformer/checkpoints/spectral

Built around the cluster it runs on:

  RESUMABLE — the `gpu` partition has an 8-hour limit.  The run watches the
    clock, saves last.pt before the limit (--max-minutes), and a resubmitted
    job resumes from it: model, optimiser, schedule, step and best score.
  CHECKPOINTS IN HOME — /scratch is wiped monthly; the data regenerates in
    under an hour, a trained model does not.
  PRECISION — bfloat16 where the GPU supports it (A100/H200), float16 with a
    gradient scaler where it does not (V100).

Validation scores NOTE F1 (mir_eval, 50 ms onset tolerance) on fixed crops of
held-out pieces, through the same decoder the existing transcriber uses, so
the numbers are directly comparable.  The checkpoint is selected on it.

Front-end normalisation is calibrated once on a sample of TRAINING clips and
stored in the checkpoint.  It is per-band and position-specific, so it must
come from full-register data — calibrating on a narrow slice broke
transposition in Phase 2.
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
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import frontend as S
from .data import RenderedClips, clips
from .model import SpectralTranscriber


def log_to(path):
    f = open(path, "a", buffering=1, encoding="utf-8")

    def say(msg):
        line = f"{time.strftime('%m-%d %H:%M:%S')}  {msg}"
        print(line, flush=True)
        f.write(line + "\n")
    return say


def note_f1(po, pf, go, gf):
    """Mean note F1 / P / R and frame F1 over a batch of (T, 88) arrays.

    Both sides decode with the short-note filters OFF.  The first full run
    selected its checkpoint with them on, applied to the reference too, which
    removed 42% of true notes from the ground truth — so the score choosing
    the "best" model was blind to short notes and fast repeats.  Selection now
    rewards exactly the fine detail that was being lost.
    """
    from app.ml.transcriber.decode import decode_notes, note_scores
    agg = {"conp_f1": 0.0, "conp_p": 0.0, "conp_r": 0.0}
    tp = fp = fn = 0
    full = dict(min_frames=1, min_note_s=0.0, keep_restruck=True)
    for i in range(len(po)):
        est = decode_notes(po[i].T, pf[i].T, np.zeros_like(po[i].T), 0.5, 0.5, **full)
        ref = decode_notes(go[i].T, gf[i].T, np.zeros_like(go[i].T), 0.5, 0.5, **full)
        s = note_scores(ref, est)
        for k in agg:
            agg[k] += s[k]
        a, b = pf[i] >= 0.5, gf[i] >= 0.5
        tp += (a & b).sum(); fp += (a & ~b).sum(); fn += (~a & b).sum()
    return {k: v / max(1, len(po)) for k, v in agg.items()}, 2 * tp / max(1, 2 * tp + fp + fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="validation")
    ap.add_argument("--crop", type=float, default=8.0)
    ap.add_argument("--val-crop", type=float, default=20.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--total-steps", type=int, default=60000)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--val-clips", type=int, default=300)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--max-minutes", type=float, default=460.0,
                    help="save and exit before this wall-clock budget (8 h job limit)")
    ap.add_argument("--limit-train", type=int, default=0, help="trial runs only")
    ap.add_argument("--data-real", default="",
                    help="root of REAL-recording clips (eval/fetch_real_clips.py) to mix in")
    ap.add_argument("--real-frac", type=float, default=0.5,
                    help="expected fraction of each batch drawn from --data-real")
    ap.add_argument("--val-data", default="",
                    help="validate on this root instead of --data (e.g. real recordings)")
    ap.add_argument("--roles", action="store_true",
                    help="curriculum: melody/bass outputs, scope-masked note loss")
    ap.add_argument("--role-weight", type=float, default=1.0)
    ap.add_argument("--n-cond", type=int, default=0,
                    help="conditioning labels (4 for song-to-piano difficulty: 3 levels + unknown)")
    ap.add_argument("--mix-p", type=float, default=0.0,
                    help="probability of burying a second clip 12-24 dB under each example")
    ap.add_argument("--soft-weight", type=float, default=0.0,
                    help="extra onset-loss weight for soft notes (vel < 50), up to 1+w at vel 0")
    ap.add_argument("--augment", action="store_true",
                    help="room/noise/EQ/level augmentation on RENDERED clips")
    ap.add_argument("--init", default="",
                    help="start from this checkpoint's weights (fine-tune) when no last.pt exists")
    args = ap.parse_args()

    t_start = time.time()
    dev = torch.device("cuda")
    ckdir = pathlib.Path(os.path.expanduser(args.ckpt_dir))
    ckdir.mkdir(parents=True, exist_ok=True)
    say = log_to(ckdir / "train.log")

    # bfloat16 only on GPUs that do it in hardware (compute capability 8+:
    # A100, H200).  torch.cuda.is_bf16_supported() also returns True on a V100,
    # because bf16 can be EMULATED there — the first cluster trial ran on a
    # V100 in slow emulated bf16 because of exactly that.
    bf16 = torch.cuda.get_device_capability(0)[0] >= 8
    amp_dtype = torch.bfloat16 if bf16 else torch.float16
    say(f"GPU {torch.cuda.get_device_name(0)}, "
        f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB, "
        f"precision {'bfloat16' if bf16 else 'float16 + grad scaler'}")

    tr_paths = clips(args.data, args.train_split)
    real_paths = clips(args.data_real, args.train_split) if args.data_real else []
    va_paths = clips(args.val_data or args.data, args.val_split)
    if args.limit_train:
        tr_paths = tr_paths[: args.limit_train]
    random.Random(0).shuffle(va_paths)
    va_paths = sorted(va_paths[: args.val_clips])     # fixed subset, every evaluation
    say(f"data {args.data}: {len(tr_paths)} rendered train clips"
        + (f" + {len(real_paths)} real ({args.real_frac:.0%} of each batch)" if real_paths else "")
        + f", {len(va_paths)} validation clips from {args.val_data or args.data}"
        + (", augmenting rendered clips" if args.augment else ""))

    fe = S.SpectralFrontend().to(dev).eval()
    net = SpectralTranscriber(d=args.d, depth=args.depth, heads=args.heads,
                              n_cond=args.n_cond, roles=args.roles).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)

    def lr_at(step):
        if step < args.warmup:
            return (step + 1) / args.warmup
        p = (step - args.warmup) / max(1, args.total_steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    scaler = torch.amp.GradScaler("cuda", enabled=not bf16)

    step, best = 0, -1.0
    last = ckdir / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=dev, weights_only=False)
        net.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        if "scaler" in ck and not bf16:
            scaler.load_state_dict(ck["scaler"])
        fe.mu.copy_(ck["fe_mu"]); fe.sd.copy_(ck["fe_sd"])
        step, best = int(ck["step"]), float(ck["best"])
        say(f"resumed from step {step} (best note F1 {best:.3f})")
    elif args.init:
        # Fine-tune: weights AND front-end normalisation come from the source
        # model, so the features it was trained on mean the same thing here.
        ck = torch.load(os.path.expanduser(args.init), map_location=dev, weights_only=False)
        # a conditioned model adds cond_emb on top of a transcription checkpoint
        missing, unexpected = net.load_state_dict(ck["model"], strict=not (args.n_cond or args.roles))
        if missing or unexpected:
            say(f"new parameters (zero-initialised): {missing}; unexpected: {unexpected}")
        fe.mu.copy_(ck["fe_mu"]); fe.sd.copy_(ck["fe_sd"])
        say(f"initialised from {args.init} (step {ck['step']})")
    else:
        # normalisation from FULL-register training audio, never a narrow slice
        cal = RenderedClips(random.Random(1).sample(tr_paths, min(64, len(tr_paths))),
                            crop_s=None)
        fe.calibrate([cal[i][0][None].to(dev) for i in range(len(cal))])
        say(f"front end calibrated on {len(cal)} training clips")
    say(f"model {net.n_params() / 1e6:.2f}M params  d={args.d} depth={args.depth} "
        f"heads={args.heads}  batch {args.batch} x {args.crop:.0f} s  "
        f"steps {step}/{args.total_steps}")

    def save(path, extra=None):
        ck = {"model": net.state_dict(), "arch": net.arch, "opt": opt.state_dict(),
              "sched": sched.state_dict(), "scaler": scaler.state_dict(),
              "fe_mu": fe.mu.cpu(), "fe_sd": fe.sd.cpu(), "step": step, "best": best,
              "args": vars(args)}
        if extra:
            ck.update(extra)
        tmp = path.with_suffix(".tmp")
        torch.save(ck, tmp)
        tmp.replace(path)                       # atomic: never a half-written checkpoint

    tr_set = RenderedClips(tr_paths + real_paths, args.crop, random_crop=True,
                           augment=args.augment, mix_p=args.mix_p,
                           with_cond=bool(args.n_cond), with_roles=args.roles)
    if real_paths:
        # Weighted so a batch is ~real_frac real however many clips each side
        # has; drawn with replacement, an "epoch" is just len(tr_set) draws.
        from torch.utils.data import WeightedRandomSampler
        w = ([(1 - args.real_frac) / len(tr_paths)] * len(tr_paths)
             + [args.real_frac / len(real_paths)] * len(real_paths))
        sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
    else:
        sampler = None
    tr_dl = DataLoader(tr_set, batch_size=args.batch, shuffle=sampler is None, sampler=sampler,
                       num_workers=args.workers, pin_memory=True, drop_last=True,
                       persistent_workers=args.workers > 0)
    va_dl = DataLoader(RenderedClips(va_paths, args.val_crop, random_crop=False,
                                     with_cond=bool(args.n_cond), with_roles=args.roles),
                       batch_size=max(1, args.batch // 4), shuffle=False,
                       num_workers=args.workers)
    pw_on = torch.tensor(30.0, device=dev)
    pw_fr = torch.tensor(2.0, device=dev)

    def validate():
        net.eval()
        po, pf, go, gf = [], [], [], []
        verr, vn = 0.0, 0.0
        role_tp = np.zeros(2); role_fp = np.zeros(2); role_fn = np.zeros(2)
        with torch.no_grad():
            for batch in va_dl:
                wav, on, fr, vel, hv = batch[:5]
                cond = batch[5].to(dev) if len(batch) > 5 else None
                x = fe(wav.to(dev, non_blocking=True))
                with torch.autocast("cuda", dtype=amp_dtype):
                    outs = net(x, cond)
                a, b, c = outs[:3]
                if args.roles and len(batch) > 9:
                    rv = batch[9].numpy()[:, None, None] > 0
                    for r, (z, y) in enumerate(((outs[3], batch[6]), (outs[4], batch[7]))):
                        pr = (torch.sigmoid(z.float()).cpu().numpy() > 0.5) & rv
                        gt = (y.transpose(1, 2).numpy() > 0.5) & rv
                        role_tp[r] += (pr & gt).sum(); role_fp[r] += (pr & ~gt).sum()
                        role_fn[r] += (~pr & gt).sum()
                po.append(torch.sigmoid(a.float()).cpu().numpy())
                pf.append(torch.sigmoid(b.float()).cpu().numpy())
                onT = on.transpose(1, 2).numpy(); go.append(onT)
                gf.append(fr.transpose(1, 2).numpy())
                m = (onT > 0.5) * hv.numpy()[:, None, None]
                vp = torch.sigmoid(c.float()).cpu().numpy()
                verr += float((np.abs(vp - vel.transpose(1, 2).numpy()) * m).sum())
                vn += float(m.sum())
        net.train()
        validate.roles = 2 * role_tp / np.maximum(1, 2 * role_tp + role_fp + role_fn)
        s, ff = note_f1(np.concatenate(po), np.concatenate(pf),
                        np.concatenate(go), np.concatenate(gf))
        return s, ff, (verr / vn * 127 if vn else float("nan"))

    net.train()
    t_log, run_loss, n_loss = time.time(), 0.0, 0
    done = False
    while not done:
        for batch in tr_dl:
            wav, on, fr, vel, hv = batch[:5]
            cond = batch[5].to(dev, non_blocking=True) if len(batch) > 5 else None
            wav = wav.to(dev, non_blocking=True)
            onT = on.to(dev, non_blocking=True).transpose(1, 2)
            frT = fr.to(dev, non_blocking=True).transpose(1, 2)
            vlT = vel.to(dev, non_blocking=True).transpose(1, 2)
            hv = hv.to(dev)
            with torch.no_grad():
                x = fe(wav)
            with torch.autocast("cuda", dtype=amp_dtype):
                outs = net(x, cond)
            a, b, c = (t.float() for t in outs[:3])
            # velocity is scored only at true onsets, and only where a velocity
            # label exists — a regression over the silent 99% would learn zero
            m = (onT > 0.5) * hv.view(-1, 1, 1)
            l_vel = (((torch.sigmoid(c) - vlT) ** 2) * m).sum() / m.sum().clamp(min=1)
            # Soft onsets count extra: 61% of real-audio misses were vel < 50.
            # Weight rises linearly from 1 at vel 50 to 1 + soft_weight at vel 0.
            w_on = None
            if args.soft_weight:
                soft = ((50 / 127 - vlT) / (50 / 127)).clamp(0, 1) * m
                w_on = 1 + args.soft_weight * soft
            if args.roles:
                # Curriculum clips: onset/frame loss only where every pitched
                # note is labelled (scope "all"); melody/bass loss wherever the
                # clip has role rolls.  Per-clip means, then averaged over the
                # clips that qualify.
                nv = batch[8].to(dev).view(-1, 1, 1); rv = batch[9].to(dev).view(-1, 1, 1)
                mel = batch[6].to(dev).transpose(1, 2); bas = batch[7].to(dev).transpose(1, 2)
                # weighted mean: w is per clip (B,1,1) or per element (B,T,88)
                pc = lambda l, w: (l * w).sum() / torch.broadcast_to(w, l.shape).sum().clamp(min=1)
                bce = lambda z, y, pw: F.binary_cross_entropy_with_logits(z, y, pos_weight=pw, reduction="none")
                om = batch[10].to(dev).transpose(1, 2) if len(batch) > 10 else 1.0
                w_o = (nv if w_on is None else nv * w_on) * om      # unsure onsets are not taught
                l_notes = pc(bce(a, onT, pw_on), w_o) + pc(bce(b, frT, pw_fr), nv)
                l_role = (pc(bce(outs[3].float(), mel, pw_fr), rv)
                          + pc(bce(outs[4].float(), bas, pw_fr), rv))
                loss = l_notes + args.role_weight * l_role + 0.5 * l_vel
            else:
                loss = (F.binary_cross_entropy_with_logits(a, onT, weight=w_on, pos_weight=pw_on)
                        + F.binary_cross_entropy_with_logits(b, frT, pos_weight=pw_fr)
                        + 0.5 * l_vel)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            step += 1
            run_loss += float(loss); n_loss += 1

            if step % 100 == 0:
                dt = time.time() - t_log
                say(f"step {step:>6}  loss {run_loss / n_loss:.4f}  "
                    f"lr {sched.get_last_lr()[0]:.2e}  {100 / dt:.2f} steps/s")
                t_log, run_loss, n_loss = time.time(), 0.0, 0

            if step % args.val_every == 0 or step >= args.total_steps:
                s, ff, vmae = validate()
                improved = s["conp_f1"] > best
                if improved:
                    best = s["conp_f1"]
                    save(ckdir / "best.pt")
                save(last)
                say(f"VAL step {step}: note F1 {s['conp_f1']:.3f} (P {s['conp_p']:.3f} "
                    f"R {s['conp_r']:.3f})  frame F1 {ff:.3f}  velocity MAE {vmae:.1f}"
                    + (f"  melody F1 {validate.roles[0]:.3f}  bass F1 {validate.roles[1]:.3f}"
                       if args.roles else "")
                    + f"{'  * best' if improved else ''}")

            minutes = (time.time() - t_start) / 60
            if step >= args.total_steps or minutes > args.max_minutes:
                save(last)
                say(f"stopping at step {step} after {minutes:.0f} min "
                    f"({'finished' if step >= args.total_steps else 'time budget; resubmit to resume'})")
                done = True
                break


if __name__ == "__main__":
    main()
