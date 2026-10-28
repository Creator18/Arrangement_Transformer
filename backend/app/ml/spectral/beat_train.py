"""Our own beat / downbeat / tempo model (2026-10-28; the user: madmom was a stopgap, build our own learned one).

Data: eval/beat_labels.py -> /scratch/basu.ang/beat_data: POP909 render clips (exact beats and downbeats from POP909's
beat_midi.txt, speed recovered per song render), 64-band log-mel at 100 fps, 20 s each.  728 training songs (115k
clips), 90 held-out songs (14k clips).

Model: a multi-task temporal convolutional network (after Boeck, Davies & Knees 2019): a small 2-D conv front end
over the mel bands, then dilated 1-D residual blocks (dilations 1 .. 1024: ~40 s of context both ways), per-frame
beat and downbeat outputs, and a tempo output (30-300 BPM, one class per BPM) from the pooled skip connections.
Targets are widened +-1 frame at half weight.  Augmentation: random time-scaling 0.8-1.25x (the labels and tempo
move with it), crops, level shifts, band masking.

Decoding is ours too (app/ml/spectral/beat_decode.py).  Validation: mir_eval beat F-measure (+-70 ms), downbeat F,
and the share of clips at the right tempo (within 4%) on held-out songs.

    python -m app.ml.spectral.beat_train --data /scratch/basu.ang/beat_data --out checkpoints/beat1 --steps 40000
"""
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

FPS, NMEL, NFR = 100, 64, 2000
BPM_LO, BPM_HI = 30, 300
NT = BPM_HI - BPM_LO + 1


class Block(nn.Module):
    def __init__(self, ch, dil, k=5, drop=0.1):
        super().__init__()
        self.a = nn.Conv1d(ch, ch, k, dilation=dil, padding=dil * (k - 1) // 2)
        self.b = nn.Conv1d(ch, ch, k, dilation=dil * 2, padding=dil * 2 * (k - 1) // 2)   # two rates per block
        self.mix = nn.Conv1d(2 * ch, ch, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        h = torch.cat([F.elu(self.a(x)), F.elu(self.b(x))], 1)
        h = self.drop(self.mix(h))
        return x + h, h


class BeatNet(nn.Module):
    def __init__(self, ch=128, n_blocks=11):
        super().__init__()
        self.front = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), padding=(1, 0)), nn.ELU(), nn.MaxPool2d((1, 3)), nn.Dropout(0.1),   # 64 -> 62 -> 20
            nn.Conv2d(32, 64, (3, 3), padding=(1, 0)), nn.ELU(), nn.MaxPool2d((1, 3)), nn.Dropout(0.1),  # 20 -> 18 -> 6
            nn.Conv2d(64, ch, (1, 6)), nn.ELU(), nn.Dropout(0.1))                                        # 6 -> 1
        self.blocks = nn.ModuleList([Block(ch, 2 ** i) for i in range(n_blocks)])
        self.beat = nn.Conv1d(ch, 2, 1)                          # beat, downbeat
        self.tempo = nn.Linear(ch, NT)

    def forward(self, mel):                                      # (B, T, NMEL) -> (B, 2, T), (B, NT)
        x = self.front(mel.unsqueeze(1)).squeeze(-1)             # (B, ch, T)
        skip = 0
        for b in self.blocks:
            x, h = b(x)
            skip = skip + h
        return self.beat(F.elu(x)), self.tempo(F.elu(skip).mean(-1))


def load(data, split):
    mels, rows = [], []
    for j in sorted(pathlib.Path(data).glob(f"{split}_*.json"), key=lambda p: int(p.stem.split("_")[-1])):
        d = json.loads(j.read_text())
        m = np.load(j.parent / f"{j.stem}.mel.npy", mmap_mode="r")
        for i, r in enumerate(d["rows"]):
            rows.append(dict(r, shard=len(mels), i=i))
        mels.append(m)
    return mels, rows


def song_of(clip):
    return clip.split("_")[0]


def make_batch(mels, rows, idx, dev, train, crop=1500):
    B = len(idx)
    X = np.stack([np.asarray(mels[rows[i]["shard"]][rows[i]["i"]], np.float32) for i in idx])   # (B, NFR, NMEL)
    X = torch.from_numpy(X).to(dev)
    scale = np.ones(B)
    if train:
        scale = np.exp(np.random.uniform(np.log(0.8), np.log(1.25), B))     # >1 = faster music
    L = crop if train else NFR
    xs, yb, yd, tb = [], [], [], []
    for k, i in enumerate(idx):
        x = X[k]
        n = int(round(NFR / scale[k]))
        if n != NFR:
            x = F.interpolate(x.T[None], size=n, mode="linear", align_corners=False)[0].T
        o = random.randint(0, max(0, n - L)) if train else 0
        x = x[o:o + L]
        if len(x) < L:
            x = F.pad(x, (0, 0, 0, L - len(x)))
        b = np.asarray(rows[i]["beats"]) / scale[k] * FPS - o
        d = np.asarray(rows[i]["downbeats"]) / scale[k] * FPS - o
        yb.append(_target(b, L)); yd.append(_target(d, L))
        bb = np.asarray(rows[i]["beats"])
        bpm = 60.0 / np.median(np.diff(bb)) * scale[k] if len(bb) > 1 else 0
        tb.append(bpm)
        xs.append(x)
    x = torch.stack(xs)
    if train:
        x = x + torch.empty(B, 1, 1, device=dev).uniform_(-0.5, 0.5)          # level
        for k in range(B):                                                    # one band mask
            if random.random() < 0.5:
                f0 = random.randint(0, NMEL - 8); x[k, :, f0:f0 + random.randint(2, 8)] = 0
    return x, torch.from_numpy(np.stack(yb)).to(dev), torch.from_numpy(np.stack(yd)).to(dev), np.array(tb)


def _target(t, L):
    y = np.zeros(L, np.float32)
    f = np.round(t).astype(int)
    f = f[(f >= 0) & (f < L)]
    for dlt, w in ((-1, .5), (1, .5), (0, 1.0)):
        g = np.clip(f + dlt, 0, L - 1)
        y[g] = np.maximum(y[g], w)
    return y


def tempo_target(bpm, dev):
    """soft tempo targets around each clip's BPM (and none when unknown)."""
    ax = np.arange(BPM_LO, BPM_HI + 1)[None]
    t = np.exp(-0.5 * ((ax - np.asarray(bpm)[:, None]) / 1.5) ** 2)
    t = t / np.maximum(t.sum(1, keepdims=True), 1e-9)
    return torch.from_numpy(t.astype(np.float32)).to(dev)


@torch.no_grad()
def evaluate(net, mels, rows, idx, dev, bs=16):
    import mir_eval
    from app.ml.spectral import beat_decode as D
    net.eval()
    fb, fd, tok, tdbl = [], [], [], []
    for s in range(0, len(idx), bs):
        part = idx[s:s + bs]
        x, _, _, _ = make_batch(mels, rows, part, dev, train=False)
        o, tl = net(x)
        p = torch.sigmoid(o.float()).cpu().numpy()
        tp = tl.float().argmax(-1).cpu().numpy() + BPM_LO
        for k, i in enumerate(part):
            ref, refd = np.asarray(rows[i]["beats"]), np.asarray(rows[i]["downbeats"])
            est = D.beats(p[k, 0])
            fb.append(mir_eval.beat.f_measure(mir_eval.beat.trim_beats(ref, 0), mir_eval.beat.trim_beats(est, 0)))
            ed, _ = D.downbeats(est, p[k, 1])
            fd.append(mir_eval.beat.f_measure(refd, ed) if len(refd) > 1 else np.nan)
            bpm = 60 / np.median(np.diff(ref))
            tok.append(abs(tp[k] / bpm - 1) < .04)
            eb = 60 / np.median(np.diff(est)) if len(est) > 2 else 0
            tdbl.append(abs(eb / bpm - 1) < .04)
    net.train()
    return dict(beat_f=round(float(np.mean(fb)), 4), beat_f_above_0_9=round(float(np.mean(np.array(fb) > .9)), 4),
                downbeat_f=round(float(np.nanmean(fd)), 4), tempo_head_right=round(float(np.mean(tok)), 4),
                decoded_tempo_right=round(float(np.mean(tdbl)), 4), clips=len(idx))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--ch", type=int, default=128)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--val-clips", type=int, default=600)
    ap.add_argument("--budget-min", type=float, default=0)
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    t_load = time.time()
    tm, tr = load(a.data, "train")
    vm, vr = load(a.data, "validation")
    print(f"train {len(tr)} clips / {len({song_of(r['clip']) for r in tr})} songs, validation {len(vr)} clips / "
          f"{len({song_of(r['clip']) for r in vr})} songs ({time.time() - t_load:.0f} s)", flush=True)
    vsub = random.Random(1).sample(range(len(vr)), min(a.val_clips, len(vr)))
    net = BeatNet(a.ch).to(dev)
    print(f"{sum(p.numel() for p in net.parameters()) / 1e6:.2f}M params on {dev}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = lambda s: min(1.0, s / 1000) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps)))
    step, best = 0, -1.0
    if (out / "last.pt").exists():
        s = torch.load(out / "last.pt", map_location=dev, weights_only=False)
        net.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); step, best = s["step"], s["best"]
        print(f"resumed at step {step} (best {best:.4f})", flush=True)
    if step >= a.steps:
        print("already finished"); return
    t0, run = time.time(), dict(beat=0.0, down=0.0, tempo=0.0, n=0)
    pos_w = torch.tensor([3.0, 10.0], device=dev)[None, :, None]     # beats ~1 frame in 50, downbeats ~1 in 200
    while step < a.steps:
        for g in opt.param_groups:
            g["lr"] = a.lr * sched(step)
        idx = np.random.randint(0, len(tr), a.bs)
        x, yb, yd, bpm = make_batch(tm, tr, idx, dev, train=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            o, tl = net(x)
        y = torch.stack([yb, yd], 1)
        bce = F.binary_cross_entropy_with_logits(o.float(), y, pos_weight=pos_w, reduction="none").mean((0, 2))
        lt = -(tempo_target(bpm, dev) * F.log_softmax(tl.float(), -1)).sum(-1).mean()
        loss = bce[0] + bce[1] + 0.5 * lt
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        step += 1
        run["beat"] += bce[0].item(); run["down"] += bce[1].item(); run["tempo"] += lt.item(); run["n"] += 1
        if step % 200 == 0:
            n = run.pop("n")
            print(f"step {step} " + " ".join(f"{k} {v / n:.4f}" for k, v in run.items()) +
                  f" {(time.time() - t0) / 60:.1f} min", flush=True)
            run = dict(beat=0.0, down=0.0, tempo=0.0, n=0)
        final = step >= a.steps
        if step % a.val_every == 0 or final:
            v = evaluate(net, vm, vr, list(range(len(vr))) if final else vsub, dev)
            print(f"VAL step {step}{' (all held-out clips)' if final else ''} {json.dumps(v)}", flush=True)
            if v["beat_f"] + v["downbeat_f"] > best:
                best = v["beat_f"] + v["downbeat_f"]
                torch.save(dict(model=net.state_dict(), cfg=dict(ch=a.ch), step=step, val=v), out / "best.pt")
            torch.save(dict(model=net.state_dict(), opt=opt.state_dict(), step=step, best=best), out / "last.pt")
        if a.budget_min and (time.time() - t0) / 60 > a.budget_min and not final:
            torch.save(dict(model=net.state_dict(), opt=opt.state_dict(), step=step, best=best), out / "last.pt")
            print(f"budget reached at step {step}: saved", flush=True)
            return
    print("done", flush=True)


if __name__ == "__main__":
    main()
