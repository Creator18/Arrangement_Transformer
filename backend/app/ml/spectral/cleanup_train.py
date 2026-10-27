"""The CLEAN-UP model (the user's design, 2026-10-27): the base's messy 0.8 output in, the clean written arrangement
as the answer.  It only KEEPS or DROPS the base's notes (it never writes one: "keep, don't write").

Data: eval/cleanup_data.py on the base_render validation clips (the base never trained on them).  Each note is a
token: pitch, start, length, loudness, peak onset probability, and a small window of the base's onset / frame
probabilities around it (its own key, the semitones / tones next to it and the octaves / twelfths above and below,
+-3 frames) -- enough to see a double press, an octave or overtone ghost or a neighbour.  A transformer over all
the notes of a 20 s clip (a start-time embedding gives it the order) says keep or drop per note.  Split by SONG.

    python -m app.ml.spectral.cleanup_train --data /scratch/basu.ang/cleanup_data --out checkpoints/cleanup1
"""
import argparse
import json
import math
import pathlib
import random
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import torch.nn as nn

FPS = 62.5
OFFS = (-24, -19, -12, -2, -1, 0, 1, 2, 12, 19, 24)          # keys looked at, relative to the note
WIN = 3                                                       # frames each side
NCTX = len(OFFS) * (2 * WIN + 1) * 2
NFEAT = 4 + NCTX                                              # length, velocity, peak, held + the window
MAXN = 1024


def song_of(name):
    return name.split("_")[0]


def featurize(path):
    z = np.load(path)
    n, on, fr = z["notes"], z["on"].astype(np.float32), z["fr"].astype(np.float32)
    if not len(n):
        return None
    x = note_features(n, on, fr)
    return dict(name=pathlib.Path(path).stem, pitch=n[:, 2].astype(np.int16), t=n[:, 0].astype(np.float32),
                x=x.astype(np.float16), y=n[:, 5].astype(np.uint8), peak=n[:, 4].astype(np.float32),
                nwritten=len(z["gt"]))


def note_features(n, on, fr):
    """(N, NFEAT) per note.  n: (N, >=5) onset s, offset s, pitch, velocity, peak onset probability; on / fr: the
    base's (88, T) probabilities (a clip, or a whole song: the app, app/services/cleanup.py)."""
    T = on.shape[1]
    onp = np.pad(on, ((24, 24), (WIN, WIN)))
    frp = np.pad(fr, ((24, 24), (WIN, WIN)))
    k = n[:, 2].astype(int) - 21
    f = np.clip(np.round(n[:, 0] * FPS).astype(int), 0, T - 1)
    rows = (k[:, None] + 24 + np.array(OFFS)[None]).astype(int)                     # (N, 11)
    cols = f[:, None] + np.arange(2 * WIN + 1)[None]                                # (N, 7) in padded frames
    ctx = np.concatenate([onp[rows[:, :, None], cols[:, None, :]].reshape(len(n), -1),
                          frp[rows[:, :, None], cols[:, None, :]].reshape(len(n), -1)], 1)
    held = fr[k, np.clip(f - 2, 0, T - 1)]                                          # key already sounding before
    return np.concatenate([np.c_[np.log1p((n[:, 1] - n[:, 0]) * 10), n[:, 3] / 127, n[:, 4], held], ctx], 1)


def load(data, workers):
    cache = pathlib.Path(data) / "_features.pt"
    if cache.exists():
        return torch.load(cache, weights_only=False)
    files = sorted(str(p) for p in pathlib.Path(data).glob("*.npz") if ".tmp" not in p.name)
    with ProcessPoolExecutor(workers) as ex:
        clips = [c for c in ex.map(featurize, files, chunksize=64) if c is not None]
    torch.save(clips, cache)
    return clips


class Cleanup(nn.Module):
    def __init__(self, d=256, depth=6, heads=4):
        super().__init__()
        self.pitch = nn.Embedding(128, d)
        self.feat = nn.Sequential(nn.Linear(NFEAT, d), nn.GELU(), nn.Linear(d, d))
        self.d = d
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, 0.1, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, depth)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def time_emb(self, t):                                    # sinusoidal start time, 16 ms .. ~60 s scales
        i = torch.arange(self.d // 2, device=t.device)
        w = 1.0 / (0.016 * (4000 ** (i / (self.d // 2))))
        a = t[..., None] * w * 2 * math.pi
        return torch.cat([a.sin(), a.cos()], -1)

    def forward(self, pitch, t, x, pad):
        h = self.pitch(pitch) + self.feat(x) + self.time_emb(t)
        return self.head(self.enc(h, src_key_padding_mask=pad))[..., 0]


def batch(clips, dev):
    B, N = len(clips), min(MAXN, max(len(c["y"]) for c in clips))
    P = torch.zeros(B, N, dtype=torch.long); T = torch.zeros(B, N); X = torch.zeros(B, N, NFEAT)
    Y = torch.zeros(B, N); pad = torch.ones(B, N, dtype=torch.bool)
    for i, c in enumerate(clips):
        n = min(N, len(c["y"]))
        P[i, :n] = torch.from_numpy(c["pitch"][:n].astype(np.int64)); T[i, :n] = torch.from_numpy(c["t"][:n])
        X[i, :n] = torch.from_numpy(c["x"][:n].astype(np.float32)); Y[i, :n] = torch.from_numpy(c["y"][:n].astype(np.float32))
        pad[i, :n] = False
    return P.to(dev), T.to(dev), X.to(dev), Y.to(dev), pad.to(dev)


@torch.no_grad()
def evaluate(net, clips, dev, bs=32):
    """F1 against the WRITTEN notes: as the base gives them (0.8), the 0.9 bar (peak >= 0.9), and kept by the model
    at a few keep bars."""
    net.eval()
    keep_p, ys, peaks, nw = [], [], [], 0
    for i in range(0, len(clips), bs):
        cs = clips[i:i + bs]
        P, T, X, Y, pad = batch(cs, dev)
        p = torch.sigmoid(net(P, T, X, pad).float())
        for j, c in enumerate(cs):
            n = min(MAXN, len(c["y"]))
            keep_p.append(p[j, :n].cpu().numpy()); ys.append(c["y"][:n]); peaks.append(c["peak"][:n])
            nw += c["nwritten"]
    kp, y, pk = np.concatenate(keep_p), np.concatenate(ys).astype(bool), np.concatenate(peaks)

    def f1(sel):
        r, p = (y & sel).sum() / max(1, nw), (y & sel).sum() / max(1, sel.sum())
        return dict(R=round(float(r), 4), P=round(float(p), 4), F1=round(float(2 * r * p / max(1e-9, r + p)), 4),
                    kept=round(float(sel.mean()), 3))
    out = {"base 0.8": f1(np.ones_like(y)), "base 0.9": f1(pk >= 0.9)}
    for b in (0.3, 0.4, 0.5, 0.6, 0.7):
        out[f"keep {b}"] = f1(kp >= b)
    net.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--val-songs", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    clips = load(a.data, a.workers)
    songs = sorted({song_of(c["name"]) for c in clips})
    val_s = set(random.Random(0).sample(songs, max(1, int(len(songs) * a.val_songs))))
    train = [c for c in clips if song_of(c["name"]) not in val_s]
    val = [c for c in clips if song_of(c["name"]) in val_s]
    print(f"{len(clips)} clips, {len(songs)} songs: train {len(train)} clips, val {len(val)} clips ({len(val_s)} songs); "
          f"notes right {np.mean(np.concatenate([c['y'] for c in train])):.3f}", flush=True)
    (out / "split.json").write_text(json.dumps(sorted(val_s)))
    net = Cleanup(a.d, a.depth).to(dev)
    print(f"{sum(p.numel() for p in net.parameters()) / 1e6:.2f}M params", flush=True)
    opt = torch.optim.AdamW(net.parameters(), a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, s / 1000) * 0.5 * (1 + math.cos(math.pi * min(1, s / a.steps))))
    step, best = 0, -1
    ck = out / "last.pt"
    if ck.exists():
        s = torch.load(ck, map_location=dev, weights_only=False)
        net.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"])
        step, best = s["step"], s["best"]
        print(f"resumed at step {step}", flush=True)
    rng = random.Random(step)
    t0 = time.time()
    loss_avg = None
    while step < a.steps:
        P, T, X, Y, pad = batch(rng.sample(train, a.bs), dev)
        with torch.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
            logit = net(P, T, X, pad)
        loss = nn.functional.binary_cross_entropy_with_logits(logit.float()[~pad], Y[~pad])
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        step += 1
        loss_avg = loss.item() if loss_avg is None else 0.98 * loss_avg + 0.02 * loss.item()
        if step % 200 == 0:
            print(f"step {step} loss {loss_avg:.4f} {(time.time() - t0) / 60:.1f} min", flush=True)
        if step % 2000 == 0 or step == a.steps:
            ev = evaluate(net, val, dev)
            score = max(v["F1"] for k, v in ev.items() if k.startswith("keep"))
            print(f"VAL step {step} " + json.dumps(ev), flush=True)
            state = dict(model=net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), step=step, best=max(best, score),
                         cfg=dict(d=a.d, depth=a.depth))
            if score > best:
                best = score
                torch.save(dict(model=net.state_dict(), step=step, score=score, cfg=dict(d=a.d, depth=a.depth), val=ev), out / "best.pt")
            torch.save(state, ck)
    print("done", flush=True)


if __name__ == "__main__":
    main()
