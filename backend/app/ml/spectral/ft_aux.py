"""Fine-tune the render-trained arranger on new renders, with SIDE OUTPUTS for how each note was sounded.

The user's idea: the renders know, for every written note, how it was played — its attack, its
release, whether it was reached by a slide, whether its line was legato, whether it was sung
(eval/render_arrangements.py, note_meta).  Those are not given to the model (real songs never
come with them); they are extra things it must PREDICT at each note's onset, so its picture of
the sound has to tell a slow swell from a new strike, a long tail from a held note, a glide from
two notes.  Weighted lightly (--aux-weight) so the notes stay the main job.

Modes (all start from the render-trained model):
  lora        frozen model + LoRA adapters (+ output head)
  lora_stem   the same, plus the first layers (stem, patch) trainable — they read attacks and tails
  full        every layer trainable

Velocity follows the song's balance: on level renders a note's target velocity is scaled by its
level, so the arrangement plays its parts in the ratio the recording had them.

Level renders (eval/render_arrangements.py --levels) add each note's LEVEL: how loud it came out
against the loudest note sounding with it.  A 4-class side output (lead / a bit under / well
under / faint) makes the model notice quiet parts instead of dropping them.  Piano renders carry
only the level (their tone columns are masked).  --base may itself be an ft_aux checkpoint: its
side-output head is reused for the outputs it already had.

GPU use: every 100 steps the log shows the share of time spent waiting for data and the GPU's
own utilisation (nvidia-smi), so an input-bound run is visible; --workers defaults to the job's
CPUs.

Quick notes (--quick-weight, --oversample): onsets within ~144 ms of another onset, or re-striking the
same key, can weigh more in the onset loss, and clips with many of them can be drawn more often.

The saved model loads in the app like any spectral checkpoint (aux heads are kept apart).

    python -m app.ml.spectral.ft_aux --data <dir with train/ validation/> --base <arr_scratch/best.pt> \\
        --ckpt-dir <dir> --mode lora_stem --aux-weight 0.2
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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from . import frontend as S
from .data import clips
from .imagine import add_lora
from .model import SpectralTranscriber, from_checkpoint, PATCH_W, F_PAD
from .train import log_to, note_f1

N_AUX = 22        # attack (4) | release (3) | slid | legato | sung | level (4) | length (3) | ring after release (3) | faded | dip
LEVEL_EDGES = (-4.0, -14.0, -26.0)     # dB under the loudest sounding note: lead | a bit under | well under | faint


def aux_targets(meta, T, f0):
    """note_meta rows (onset s, pitch, attack_s, release_tc, decay_tc, sustain_cb, slid, legato, sung
    [, level_db, tone_valid [, length class]]) -> targets (N_AUX, 88, T) and masks (3, 88, T) at each
    note's onset frame: [0] the tone outputs, [1] the level output, [2] the length output (did the
    note SOUND as written, longer, or shorter: articulation both ways)."""
    tgt = np.zeros((N_AUX, 88, T), np.float32); mask = np.zeros((4, 88, T), np.float32)
    for row in meta:
        if len(row) < 9:                       # modular renders: onset, pitch, effect value only (no side outputs)
            continue
        on, p, att, rel, _dec, _sus, slid, leg, sung = row[:9]
        k = int(p) - 21; f = int(round(on * S.FPS)) - f0
        if not (0 <= k < 88 and 0 <= f < T):
            continue
        if len(row) < 11 or row[10] > 0:
            ac = 0 if att <= 0.005 else 1 if att <= 0.05 else 2 if att <= 0.15 else 3
            rc = 0 if rel < 0 else 1 if rel < 1800 else 2
            tgt[ac, k, f] = 1; tgt[4 + rc, k, f] = 1; tgt[7, k, f] = slid; tgt[8, k, f] = leg; tgt[9, k, f] = sung
            mask[0, k, f] = 1
        if len(row) >= 11:
            lc = sum(row[9] < e for e in LEVEL_EDGES)
            tgt[10:14, k, f] = 0; tgt[10 + lc, k, f] = 1; mask[1, k, f] = 1
        if len(row) >= 12:
            tgt[14:17, k, f] = 0; tgt[14 + int(row[11]), k, f] = 1; mask[2, k, f] = 1
        if len(row) >= 15:                     # sustain cases: ring after release, faded while held, mid-note dip
            tgt[17:20, k, f] = 0; tgt[17 + int(row[12]), k, f] = 1; tgt[20, k, f] = row[13]; tgt[21, k, f] = row[14]
            mask[3, k, f] = 1
    return tgt, mask


class Clips(Dataset):
    def __init__(self, paths, crop_s=8.0, train=True, level_velocity=True):
        self.paths, self.train, self.level_velocity = paths, train, level_velocity
        self.n = int(crop_s * S.SR) if crop_s else None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]; z = np.load(p)
        y = z["y"].astype(np.float32) / 32767; on, fr = z["on"].astype(np.float32), z["fr"].astype(np.float32)
        vp = p.with_name(p.stem + ".vel.npy")
        vel = np.load(vp).astype(np.float32) / 127 if vp.exists() else np.zeros_like(on)
        hv = 1.0 if vp.exists() else 0.0
        meta = z["note_meta"] if "note_meta" in z.files else np.zeros((0, 9), np.float32)
        if self.level_velocity and meta.shape[1] >= 11 and str(z["piano"]) == "multi" and hv:
            # Velocity follows the balance in the song: each note's target is scaled by how far
            # under the loudest sounding note it came out (velocity ~ amplitude^1/2, as the piano
            # renders, whose velocities already carry their levels).
            for on_s, p_, *_r in meta:
                k = int(p_) - 21; f = int(round(on_s * S.FPS))
                if 0 <= k < 88 and 0 <= f < vel.shape[1]:
                    vel[k, f] *= 10 ** (min(0.0, float(_r[7])) / 40)
        T = on.shape[1]; f0 = 0
        if self.n and len(y) > self.n:
            nT = S.n_frames(self.n)
            f0 = random.randrange(0, T - nT) if self.train else 0
            a = int(round(f0 / S.FPS * S.SR)); y = np.pad(y[a:a + self.n], (0, max(0, self.n - len(y[a:a + self.n]))))
            on, fr, vel = on[:, f0:f0 + nT], fr[:, f0:f0 + nT], vel[:, f0:f0 + nT]
        tgt, mask = aux_targets(meta, on.shape[1], f0)
        return (torch.from_numpy(y), torch.from_numpy(on), torch.from_numpy(fr), torch.from_numpy(vel),
                torch.tensor(hv), torch.from_numpy(tgt), torch.from_numpy(mask))


def quick_weight_map(onT, w, win, w_same=None, frT=None, w_oct=1.0):
    """(B, T, 88) onset targets -> onset-loss weights (1 elsewhere), the largest that applies:
      w       an onset within `win` frames after another onset on a DIFFERENT key (a quick note)
      w_same  a re-strike of the SAME key within `win` frames (default: w)
      w_oct   an onset with the same note sounding 1-3 octaves BELOW it (frame targets frT): the high
              note the bass tends to mask"""
    w_same = w if w_same is None else w_same
    on = (onT > 0.5).float()
    n_all = on.sum(-1, keepdim=True)                                     # onsets per frame, all keys
    cnt_all = torch.zeros_like(n_all); cnt_same = torch.zeros_like(on)
    for d in range(1, win + 1):                                          # onsets in the window before each frame
        cnt_all[:, d:] += n_all[:, :-d]; cnt_same[:, d:] += on[:, :-d]
    other = ((cnt_all - cnt_same) > 0).float()                           # ... on a DIFFERENT key
    same = (cnt_same > 0).float()
    wt = torch.ones_like(on)
    wt = torch.maximum(wt, 1.0 + (w - 1.0) * on * other)
    wt = torch.maximum(wt, 1.0 + (w_same - 1.0) * on * same)
    if w_oct != 1.0 and frT is not None:
        snd = torch.maximum((frT > 0.5).float(), on)
        below = torch.zeros_like(on)
        for o in (12, 24, 36):
            below[..., o:] = torch.maximum(below[..., o:], snd[..., :-o])
        wt = torch.maximum(wt, 1.0 + (w_oct - 1.0) * on * below)
    return wt


def _count_quick(args):
    path, win = args
    try:
        on = np.load(path)["on"] > 0                                     # (88, T)
        t = np.flatnonzero(on.any(0))
        gaps = np.diff(t)
        return path.name, int(on[:, t[1:][(gaps >= 1) & (gaps <= win)]].sum()) if len(t) > 1 else 0
    except Exception:                                                    # noqa: BLE001
        return path.name, 0


def quick_counts(root, paths, win, workers):
    """Quick onsets per clip (cached): onsets within `win` frames after the previous onset frame."""
    import json
    from multiprocessing import Pool
    cache = root / "quick_counts.json"
    have = json.loads(cache.read_text()) if cache.exists() else {}
    todo = [(p, win) for p in paths if p.name not in have]
    if todo:
        with Pool(max(1, workers)) as pool:
            for name, n in pool.imap_unordered(_count_quick, todo, chunksize=64):
                have[name] = n
        cache.write_text(json.dumps(have))
    return have


def trunk(net, x):
    """The transcriber's forward up to its fused features h (B, stem, 264, T)."""
    B, _, Fb, T = x.shape
    s = net.stem(x)
    Tp = math.ceil(T / PATCH_W) * PATCH_W
    z = net.patch(F.pad(s, (0, Tp - T, 0, F_PAD - Fb))).permute(0, 2, 3, 1) + net.pitch_emb
    for blk in net.blocks:
        z = blk(z)
    z = net.norm(z).permute(0, 3, 1, 2)
    u = net.unpatch(z)[:, :, :Fb, :T]
    return net.fuse(torch.cat([u, s], dim=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--base", required=True); ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--mode", choices=["lora", "lora_stem", "full"], default="lora_stem")
    ap.add_argument("--lora", type=int, default=16); ap.add_argument("--aux-weight", type=float, default=0.2)
    ap.add_argument("--crop", type=float, default=8.0); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--total-steps", type=int, default=20000); ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--val-clips", type=int, default=300)
    ap.add_argument("--patience", type=int, default=0,
                    help="stop when note F1 has not improved by --min-delta for this many validations (0 = run all steps)")
    ap.add_argument("--min-delta", type=float, default=0.002)
    ap.add_argument("--offset-weight", type=float, default=0.0,
                    help="train a NOTE-END output (key up here), loss weight; adds the head if the base has none")
    ap.add_argument("--quick-weight", type=float, default=1.0,
                    help="onset-loss weight of QUICK notes (within --quick-frames of another onset, or re-striking the same key)")
    ap.add_argument("--quick-frames", type=int, default=9, help="'quick' window, frames (9 = 144 ms)")
    ap.add_argument("--restrike-weight", type=float, default=None, help="weight of same-key re-strikes (default: --quick-weight)")
    ap.add_argument("--octave-weight", type=float, default=1.0,
                    help="onset-loss weight of notes with the same note sounding 1-3 octaves below (bass masking)")
    ap.add_argument("--oversample", action="store_true",
                    help="draw clips with many quick notes more often (up to 3x; counts cached in <data>/quick_counts.json)")
    ap.add_argument("--no-level-velocity", action="store_true",
                    help="keep the arrangement's own velocities on level renders (default: scale them by each note's level)")
    ap.add_argument("--workers", type=int, default=max(2, int(os.environ.get("SLURM_CPUS_PER_TASK", "8")) - 1))
    ap.add_argument("--max-minutes", type=float, default=460)
    a = ap.parse_args()
    ckdir = pathlib.Path(os.path.expanduser(a.ckpt_dir)); ckdir.mkdir(parents=True, exist_ok=True)
    say = log_to(ckdir / "train.log"); t0 = time.time(); dev = "cuda"
    bf16 = torch.cuda.get_device_capability(0)[0] >= 8; amp = torch.bfloat16 if bf16 else torch.float16
    ck = torch.load(os.path.expanduser(a.base), map_location=dev, weights_only=False)
    fe = S.SpectralFrontend().to(dev).eval(); fe.mu.copy_(ck["fe_mu"]); fe.sd.copy_(ck["fe_sd"])
    net = from_checkpoint(ck)
    if a.offset_weight > 0 and net.offset_head is None:
        net.add_offset_head()
    if a.mode != "full":
        add_lora(net, a.lora)
        for p in net.head.parameters():
            p.requires_grad_(True)
        if net.offset_head is not None:
            for p in net.offset_head.parameters():
                p.requires_grad_(True)
        if a.mode == "lora_stem":
            for mod in (net.stem, net.patch):
                for p in mod.parameters():
                    p.requires_grad_(True)
    net = net.to(dev)
    aux = nn.Conv2d(net.arch.get("stem", 32), N_AUX, (3, 1), stride=(3, 1)).to(dev)
    if "aux" in ck:                                                       # the base's side outputs carry over
        with torch.no_grad():
            w = ck["aux"]["weight"]; n0 = w.shape[0]
            aux.weight[:n0].copy_(w[:N_AUX]); aux.bias[:n0].copy_(ck["aux"]["bias"][:N_AUX])
    params = [p for p in net.parameters() if p.requires_grad] + list(aux.parameters())
    say(f"ft_aux: mode {a.mode}, aux weight {a.aux_weight}; trainable {sum(p.numel() for p in params) / 1e3:.0f}k "
        f"of {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M; GPU {torch.cuda.get_device_name(0)}")
    tr = clips(pathlib.Path(a.data), "train")
    va = random.Random(0).sample(clips(pathlib.Path(a.data), "validation"), min(a.val_clips, len(clips(pathlib.Path(a.data), "validation"))))
    say(f"clips: {len(tr)} train / {len(va)} val")
    sampler = None
    if a.oversample:                                                     # clips with many quick notes, more often
        qc = quick_counts(pathlib.Path(a.data), tr, a.quick_frames, a.workers)
        w = torch.tensor([1.0 + min(2.0, qc.get(p.name, 0) / 20.0) for p in tr], dtype=torch.double)
        sampler = WeightedRandomSampler(w, num_samples=len(tr), replacement=True)
        say(f"oversampling: {int((w > 1.5).sum())} of {len(tr)} clips drawn >1.5x (mean weight {float(w.mean()):.2f})")
    dl = DataLoader(Clips(tr, a.crop, level_velocity=not a.no_level_velocity), batch_size=a.batch,
                    shuffle=sampler is None, sampler=sampler, num_workers=a.workers, pin_memory=True,
                    drop_last=True, persistent_workers=a.workers > 0, prefetch_factor=6 if a.workers else None)
    vdl = DataLoader(Clips(va, 20.0, train=False), batch_size=2, num_workers=a.workers)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / a.warmup, 0.5 * (
        1 + math.cos(math.pi * min(1.0, max(0, s - a.warmup) / max(1, a.total_steps - a.warmup))))))
    scaler = torch.amp.GradScaler("cuda", enabled=not bf16)
    step, best = 0, -1.0; last = ckdir / "last.pt"; stale = 0
    if last.exists():
        c = torch.load(last, map_location=dev, weights_only=False)
        net.load_state_dict(c["model"]); aux.load_state_dict(c["aux"]); opt.load_state_dict(c["opt"]); sched.load_state_dict(c["sched"])
        step, best = c["step"], c["best"]; stale = c.get("stale", 0); say(f"resumed at step {step}")

    def save(path):
        tmp = path.with_suffix(".tmp")
        torch.save({"model": net.state_dict(), "aux": aux.state_dict(), "arch": net.arch,
                    "lora": a.lora if a.mode != "full" else 0, "mode": a.mode, "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "step": step, "best": best, "stale": stale, "fe_mu": fe.mu.cpu(), "fe_sd": fe.sd.cpu(),
                    "args": vars(a)}, tmp)
        tmp.replace(path)

    def validate():
        net.eval(); aux.eval(); P = ([], []); G = ([], []); acc = np.zeros(8); cnt = np.zeros(4)
        with torch.no_grad():
            for wav, on, fr, _v, _h, tgt, mask in vdl:
                x = fe(wav.to(dev))
                with torch.autocast("cuda", dtype=amp):
                    h = trunk(net, x); o = net.head(h).permute(1, 0, 3, 2); za = aux(h)
                P[0].append(torch.sigmoid(o[0].float()).cpu().numpy()); P[1].append(torch.sigmoid(o[1].float()).cpu().numpy())
                G[0].append(on.transpose(1, 2).numpy()); G[1].append(fr.transpose(1, 2).numpy())
                m = mask[:, 0].to(dev) > 0; ml = mask[:, 1].to(dev) > 0; za = za.float(); t = tgt.to(dev)
                if m.any():
                    acc[0] += (za[:, :4].argmax(1)[m] == t[:, :4].argmax(1)[m]).float().sum().item()
                    acc[1] += (za[:, 4:7].argmax(1)[m] == t[:, 4:7].argmax(1)[m]).float().sum().item()
                    acc[2] += ((za[:, 7] > 0)[m] == (t[:, 7] > .5)[m]).float().sum().item()
                    acc[3] += ((za[:, 9] > 0)[m] == (t[:, 9] > .5)[m]).float().sum().item()
                    cnt[0] += int(m.sum())
                if ml.any():
                    acc[4] += (za[:, 10:14].argmax(1)[ml] == t[:, 10:14].argmax(1)[ml]).float().sum().item()
                    cnt[1] += int(ml.sum())
                mn = mask[:, 2].to(dev) > 0
                if mn.any():
                    acc[5] += (za[:, 14:17].argmax(1)[mn] == t[:, 14:17].argmax(1)[mn]).float().sum().item()
                    cnt[2] += int(mn.sum())
                ms = mask[:, 3].to(dev) > 0
                if ms.any():
                    acc[6] += (za[:, 17:20].argmax(1)[ms] == t[:, 17:20].argmax(1)[ms]).float().sum().item()
                    acc[7] += ((za[:, 21] > 0)[ms] == (t[:, 21] > .5)[ms]).float().sum().item()
                    cnt[3] += int(ms.sum())
        net.train(); aux.train()
        r = note_f1(np.concatenate(P[0]), np.concatenate(P[1]), np.concatenate(G[0]), np.concatenate(G[1]))
        return r, (acc / np.maximum(1, cnt[[0, 0, 0, 0, 1, 2, 3, 3]])).round(3).tolist()

    import subprocess, threading
    util = []                                                             # GPU utilisation samples (nvidia-smi, 1/s)

    def sample_gpu():
        while True:
            try:
                q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0].split(",")
                util.append((float(q[0]), float(q[1])))      # a job sees only its own GPU
            except Exception:                                             # noqa: BLE001
                pass
            time.sleep(1.0)
    threading.Thread(target=sample_gpu, daemon=True).start()
    pw_on, pw_fr = torch.tensor(30.0, device=dev), torch.tensor(2.0, device=dev)
    run = np.zeros(2); n = 0; t_log = time.time(); net.train(); wait = 0.0; t_fetch = time.time()
    while step < a.total_steps:
        for wav, on, fr, vel, hv, tgt, mask in dl:
            wait += time.time() - t_fetch
            with torch.no_grad():
                x = fe(wav.to(dev, non_blocking=True))
            onT, frT, vlT = on.to(dev).transpose(1, 2), fr.to(dev).transpose(1, 2), vel.to(dev).transpose(1, 2)
            hv = hv.to(dev); tgt, mask = tgt.to(dev), mask.to(dev)
            with torch.autocast("cuda", dtype=amp):
                h = trunk(net, x); o = net.head(h).permute(1, 0, 3, 2); za = aux(h)
            a_, b_, c_ = (t.float() for t in (o[0], o[1], o[2]))
            l_off = torch.zeros((), device=dev)
            if a.offset_weight > 0:                                       # key up: where a sustain run ends
                offT = torch.zeros_like(frT); offT[:, 1:] = ((frT[:, :-1] > .5) & (frT[:, 1:] <= .5)).float()
                with torch.autocast("cuda", dtype=amp):
                    zo = net.offset_head(h)[:, 0].permute(0, 2, 1)
                l_off = F.binary_cross_entropy_with_logits(zo.float(), offT, pos_weight=pw_on)
            mv = (onT > 0.5) * hv.view(-1, 1, 1)
            l_vel = (((torch.sigmoid(c_) - vlT) ** 2) * mv).sum() / mv.sum().clamp(min=1)
            wq = (quick_weight_map(onT, a.quick_weight, a.quick_frames, a.restrike_weight, frT, a.octave_weight)
                  if (a.quick_weight != 1.0 or a.octave_weight != 1.0 or a.restrike_weight not in (None, 1.0)) else None)
            loss = (a.offset_weight * l_off + F.binary_cross_entropy_with_logits(a_, onT, pos_weight=pw_on, weight=wq)
                    + F.binary_cross_entropy_with_logits(b_, frT, pos_weight=pw_fr) + 0.5 * l_vel)
            l_aux = torch.zeros((), device=dev)
            if a.aux_weight and mask.sum() > 0:
                za = za.float(); m = mask[:, 0] > 0; ml = mask[:, 1] > 0
                ce = lambda lo, hi, mm: F.cross_entropy(za[:, lo:hi].permute(0, 2, 3, 1)[mm], tgt[:, lo:hi].permute(0, 2, 3, 1)[mm].argmax(-1))
                bce = lambda c: F.binary_cross_entropy_with_logits(za[:, c][m], tgt[:, c][m])
                if m.any():
                    l_aux = l_aux + ce(0, 4, m) + ce(4, 7, m) + bce(7) + bce(8) + bce(9)
                if ml.any():
                    l_aux = l_aux + ce(10, 14, ml)
                mn = mask[:, 2] > 0
                if mn.any():
                    l_aux = l_aux + ce(14, 17, mn)
                ms = mask[:, 3] > 0
                if ms.any():
                    l_aux = (l_aux + ce(17, 20, ms) + F.binary_cross_entropy_with_logits(za[:, 20][ms], tgt[:, 20][ms])
                             + F.binary_cross_entropy_with_logits(za[:, 21][ms], tgt[:, 21][ms]))
                loss = loss + a.aux_weight * l_aux
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0); scaler.step(opt); scaler.update(); sched.step(); step += 1
            run += [float(loss), float(l_aux)]; n += 1
            if step % 100 == 0:
                dt = time.time() - t_log; u = util[-max(1, int(dt)):] if util else []
                gu = f"  GPU {np.mean([x[0] for x in u]):.0f}% ({max(x[1] for x in u) / 1024:.1f} GB)" if u else ""
                say(f"step {step:>6}  loss {run[0] / n:.4f}  aux {run[1] / n:.4f}  {100 / dt:.2f} steps/s  "
                    f"data wait {100 * wait / dt:.0f}%{gu}")
                run, n, t_log, wait = np.zeros(2), 0, time.time(), 0.0
            if step % a.val_every == 0 or step >= a.total_steps:
                r, aa = validate()
                stale = 0 if r[0]["conp_f1"] > best + a.min_delta else stale + 1
                if r[0]["conp_f1"] > best:
                    best = r[0]["conp_f1"]; save(ckdir / "best.pt")
                save(last)
                say(f"VAL step {step}: note F1 {r[0]['conp_f1']:.3f} (P {r[0]['conp_p']:.3f} R {r[0]['conp_r']:.3f})  "
                    f"frame F1 {r[1]:.3f}  |  side outputs: attack {aa[0]:.2f}  release {aa[1]:.2f}  slide {aa[2]:.2f}  sung {aa[3]:.2f}  "
                    f"level {aa[4]:.2f}  length {aa[5]:.2f}  ring {aa[6]:.2f}  dip {aa[7]:.2f}")
                if a.patience and stale >= a.patience:              # plateau: no gain for `patience` validations
                    save(last); say(f"stopping at step {step} (finished): note F1 plateaued for {stale} validations; best {best:.3f}")
                    return
            if step % 50 == 0 and (ckdir / "HANDOVER").exists():
                save(last); say(f"handing over at step {step}: saved last.pt for the faster GPU"); return
            if (time.time() - t0) / 60 > a.max_minutes:
                save(last); say("time limit: saved last.pt, resubmit to resume"); return
            if step >= a.total_steps:
                break
            t_fetch = time.time()
    save(last); say(f"stopping at step {step} (finished); best note F1 {best:.3f}")


if __name__ == "__main__":
    main()
