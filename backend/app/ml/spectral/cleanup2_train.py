"""Clean-up v2 (2026-10-27): keep / drop each of the base's 0.8 notes AND where it sits in the beat (learned
quantizing: the user's point that notes sit at many positions of a beat, so no naive nearest-grid snap).

v1 (cleanup_train.py, by ear "amazing" at keep 0.4) overfit after 4k steps on 228 songs, and the user still hears
double presses on some songs.  v2:
  songs      + 40 clips of each of the 1,522 TRAINING songs (eval/cleanup_data.py --list), with the 34k validation
             clips; held out: the same 22 validation songs as v1 (checkpoints/cleanup1/split.json)
  same key   time since the last strike of this key and whether it still sounds, that note's length, time to the next
             strike, notes starting within 30 ms (a chord) -- a ghost re-strike 50-300 ms after a held note
  beats      madmom's beat grid on the clip's audio (eval/clip_beats.py): phase in the beat at 1, 1/2, 1/3, 1/4 periods,
             the local beat length, signed distance to each grid slot, beat / downbeat activations, phase in the bar
  position   a second output per note: which slot of the beat its WRITTEN start is in (beat, 1/4, 1/3, 1/2, 2/3, 3/4,
             next beat) or "off" (leave it where it is), relative to the madmom beat holding the base's note start.
             Trained on POP909 clips only: their arrangements sit on a 16th / triplet grid (94%, eval/grid_check.py);
             Music2MIDI's are performances (22% exact on their estimated beats), so they teach keep / drop only.

    python -m app.ml.spectral.cleanup2_train --data /scratch/basu.ang/cleanup_data,/scratch/basu.ang/cleanup_data_tr \
        --beats /scratch/basu.ang/clip_beats --split checkpoints/cleanup1/split.json --out checkpoints/cleanup2
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

from app.ml.spectral import cleanup_train as C1

FPS = C1.FPS
SLOTS = np.array([0, 1 / 4, 1 / 3, 1 / 2, 2 / 3, 3 / 4, 1.0])
OFF = len(SLOTS)                                              # class "leave it where it is"
NPOS = OFF + 1
NKEY = 6
NBEAT = 8 + 1 + len(SLOTS) + 2 + 2 + 1
NFEAT = C1.NFEAT + NKEY + NBEAT
MAXN = C1.MAXN
BEATS_DIR = [None]


def key_features(n):
    """(N, NKEY) same-key context from the note list itself."""
    on, off, p = n[:, 0], n[:, 1], n[:, 2].astype(int)
    out = np.zeros((len(n), NKEY), np.float32)
    last = {}
    for i in np.argsort(on, kind="stable"):
        j = last.get(p[i])
        if j is not None:
            out[i, 0] = np.log1p(min(2.0, on[i] - on[j]) * 20)        # since this key's last strike
            out[i, 1] = float(off[j] > on[i] + 0.01)                  # ... which still sounds
            out[i, 2] = np.log1p(min(2.0, off[j] - on[j]) * 10)       # its length
            out[j, 3] = np.log1p(min(2.0, on[i] - on[j]) * 20)        # (for j: time to the next strike)
        else:
            out[i, 0] = out[i, 3] = np.log1p(40)
        last[p[i]] = i
    out[:, 3] = np.where(out[:, 3] == 0, np.log1p(40), out[:, 3])
    srt = np.sort(on)
    out[:, 4] = np.log1p(np.searchsorted(srt, on + 0.03) - np.searchsorted(srt, on - 0.03))   # chord size
    out[:, 5] = 1.0
    return out


def beat_context(onsets, b):
    """per onset: containing beat k (or -1), its start, its length.  b: beat times (s)."""
    if len(b) < 3:
        return np.full(len(onsets), -1), np.zeros(len(onsets)), np.ones(len(onsets))
    ibi = np.diff(b)
    ext = np.r_[b[0] - ibi[0], b, b[-1] + ibi[-1]]                 # one extrapolated beat each side
    k = np.searchsorted(ext, onsets, side="right") - 1
    ok = (k >= 0) & (k < len(ext) - 1)
    kk = np.clip(k, 0, len(ext) - 2)
    return np.where(ok, kk, -1), ext[kk], ext[kk + 1] - ext[kk]


def beat_features(onsets, z):
    """(N, NBEAT) from a clip_beats record (beats, dbeats, dbnum, act, dact) or None."""
    out = np.zeros((len(onsets), NBEAT), np.float32)
    if z is None or len(z["beats"]) < 3:
        return out
    k, b0, ibi = beat_context(onsets, z["beats"].astype(float))
    ok = k >= 0
    ph = (onsets - b0) / ibi
    cols = []
    for per in (1, 1 / 2, 1 / 3, 1 / 4):
        a = 2 * np.pi * ph / per
        cols += [np.sin(a), np.cos(a)]
    cols.append(np.log(np.clip(ibi, .1, 3)))
    cols += [np.clip((ph - s) * ibi * 10, -3, 3) for s in SLOTS]     # signed distance to each slot, in 100 ms
    act, dact = z["act"].astype(np.float32), z["dact"].astype(np.float32)
    fi = np.clip((onsets * 100).astype(int), 0, max(0, len(act) - 1))
    cols += [act[fi] if len(act) else np.zeros(len(onsets)), dact[fi] if len(dact) else np.zeros(len(onsets))]
    db, num = z["dbeats"].astype(float), z["dbnum"].astype(int)
    if len(db) >= 3 and (num == 1).sum() >= 2:
        bar = db[num == 1]
        kb, s0, L = beat_context(onsets, bar)
        a = 2 * np.pi * (onsets - s0) / L
        cols += [np.where(kb >= 0, np.sin(a), 0), np.where(kb >= 0, np.cos(a), 0)]
    else:
        cols += [np.zeros(len(onsets)), np.zeros(len(onsets))]
    cols.append(np.ones(len(onsets)))
    out[:] = np.stack(cols, 1)
    out[~ok] = 0
    return out


def match_written(n, gt):
    """index of the written note each base note matches (same rule as cleanup_data: same key, 48 ms, nearest)."""
    starts = {}
    for j, (o, _, p) in enumerate(gt):
        starts.setdefault(int(p), []).append((o, j))
    cand = [(abs(o - g[0]), i, j) for i, g in enumerate(n) for o, j in starts.get(int(g[2]), []) if abs(o - g[0]) <= 3 / FPS + 1e-6]
    m = np.full(len(n), -1)
    used = set()
    for _, i, j in sorted(cand):
        if m[i] < 0 and j not in used:
            m[i] = j
            used.add(j)
    return m


def pos_labels(n, m, gt, z, use):
    """slot class of each matched note's WRITTEN start in the madmom beat holding the BASE start; -1 = no label."""
    y = np.full(len(n), -1)
    if not use or z is None or len(z["beats"]) < 3:
        return y
    k, b0, ibi = beat_context(n[:, 0], z["beats"].astype(float))
    for i in np.nonzero((m >= 0) & (k >= 0))[0]:
        ph = (gt[m[i], 0] - b0[i]) / ibi[i]
        d = np.abs(ph - SLOTS) * ibi[i]
        j = int(np.argmin(d))
        y[i] = j if d[j] < min(0.03, 0.08 * ibi[i]) else OFF
    return y


def featurize(path):
    z = np.load(path)
    n, on, fr, gt = z["notes"], z["on"].astype(np.float32), z["fr"].astype(np.float32), z["gt"]
    if not len(n):
        return None
    bf = pathlib.Path(BEATS_DIR[0]) / pathlib.Path(path).name
    zb = dict(np.load(bf)) if bf.exists() else None
    x = np.concatenate([C1.note_features(n, on, fr), key_features(n), beat_features(n[:, 0], zb)], 1)
    m = match_written(n, gt)
    name = pathlib.Path(path).stem
    return dict(name=name, pitch=n[:, 2].astype(np.int16), t=n[:, 0].astype(np.float32), x=x.astype(np.float16),
                y=n[:, 5].astype(np.uint8), pos=pos_labels(n, m, gt, zb, name.startswith("pop")).astype(np.int8),
                peak=n[:, 4].astype(np.float32), nwritten=len(gt), beats=zb is not None,
                err=(np.where(m >= 0, gt[np.clip(m, 0, None), 0] - n[:, 0], np.nan) if len(gt)
                     else np.full(len(n), np.nan)).astype(np.float32),
                snap=snap_targets(n[:, 0], zb).astype(np.float32))


def snap_targets(onsets, zb):
    """(N, NPOS) the start time each slot class would give (OFF = where it is)."""
    out = np.repeat(onsets[:, None], NPOS, 1).astype(np.float64)
    if zb is None or len(zb["beats"]) < 3:
        return out
    k, b0, ibi = beat_context(onsets, zb["beats"].astype(float))
    ok = k >= 0
    out[ok, :OFF] = b0[ok, None] + SLOTS[None] * ibi[ok, None]
    return out


def init_worker(beats):
    BEATS_DIR[0] = beats


def load(dirs, beats, workers):
    clips = []
    for d in dirs:
        cache = pathlib.Path(d) / "_features_v2.pt"
        if cache.exists():
            clips += torch.load(cache, weights_only=False)
            continue
        files = sorted(str(p) for p in pathlib.Path(d).glob("*.npz") if ".tmp" not in p.name)
        with ProcessPoolExecutor(workers, initializer=init_worker, initargs=(beats,)) as ex:
            cs = [c for c in ex.map(featurize, files, chunksize=64) if c is not None]
        torch.save(cs, cache)
        clips += cs
    return clips


class Cleanup2(C1.Cleanup):
    def __init__(self, d=256, depth=6, heads=4):
        nn.Module.__init__(self)
        self.pitch = nn.Embedding(128, d)
        self.feat = nn.Sequential(nn.Linear(NFEAT, d), nn.GELU(), nn.Linear(d, d))
        self.d = d
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, 0.1, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, depth)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1 + NPOS))

    def forward(self, pitch, t, x, pad):
        h = self.pitch(pitch) + self.feat(x) + self.time_emb(t)
        o = self.head(self.enc(h, src_key_padding_mask=pad))
        return o[..., 0], o[..., 1:]


def batch(clips, dev):
    B, N = len(clips), min(MAXN, max(len(c["y"]) for c in clips))
    P = torch.zeros(B, N, dtype=torch.long); T = torch.zeros(B, N); X = torch.zeros(B, N, NFEAT)
    Y = torch.zeros(B, N); Q = torch.full((B, N), -1, dtype=torch.long); pad = torch.ones(B, N, dtype=torch.bool)
    for i, c in enumerate(clips):
        n = min(N, len(c["y"]))
        P[i, :n] = torch.from_numpy(c["pitch"][:n].astype(np.int64)); T[i, :n] = torch.from_numpy(c["t"][:n])
        X[i, :n] = torch.from_numpy(c["x"][:n].astype(np.float32)); Y[i, :n] = torch.from_numpy(c["y"][:n].astype(np.float32))
        Q[i, :n] = torch.from_numpy(c["pos"][:n].astype(np.int64)); pad[i, :n] = False
    return P.to(dev), T.to(dev), X.to(dev), Y.to(dev), Q.to(dev), pad.to(dev)


@torch.no_grad()
def evaluate(net, clips, dev, bs=32):
    """keep / drop against the written notes (as v1), and the start-time error of right notes: as the base gives
    them, snapped by the position output, and snapped to the nearest slot (the naive snap)."""
    net.eval()
    kp, ys, pk, pp, pos, err, snap, nw = [], [], [], [], [], [], [], 0
    for i in range(0, len(clips), bs):
        cs = clips[i:i + bs]
        P, T, X, Y, Q, pad = batch(cs, dev)
        lk, lp = net(P, T, X, pad)
        k_, p_ = torch.sigmoid(lk.float()), torch.softmax(lp.float(), -1)
        for j, c in enumerate(cs):
            n = min(MAXN, len(c["y"]))
            kp.append(k_[j, :n].cpu().numpy()); pp.append(p_[j, :n].cpu().numpy()); ys.append(c["y"][:n])
            pk.append(c["peak"][:n]); pos.append(c["pos"][:n]); err.append(c["err"][:n]); snap.append(c["snap"][:n] - c["t"][:n, None])
            nw += c["nwritten"]
    kp, y, pk = np.concatenate(kp), np.concatenate(ys).astype(bool), np.concatenate(pk)
    pp, pos, err, snap = np.concatenate(pp), np.concatenate(pos), np.concatenate(err), np.concatenate(snap)

    def f1(sel):
        r, p = (y & sel).sum() / max(1, nw), (y & sel).sum() / max(1, sel.sum())
        return dict(R=round(float(r), 4), P=round(float(p), 4), F1=round(float(2 * r * p / max(1e-9, r + p)), 4),
                    kept=round(float(sel.mean()), 3))
    out = {"base 0.8": f1(np.ones_like(y)), "base 0.9": f1(pk >= 0.9)}
    for b in (0.3, 0.4, 0.5, 0.6):
        out[f"keep {b}"] = f1(kp >= b)
    lab = pos >= 0
    if lab.any():
        pred = pp.argmax(-1)
        moved = np.take_along_axis(snap, pred[:, None], 1)[:, 0]            # start shift the position output gives
        near = np.argmin(np.abs(snap[:, :OFF]), 1)
        naive = np.take_along_axis(snap, near[:, None], 1)[:, 0]
        e0, e1, e2 = np.abs(err[lab]), np.abs(err[lab] - moved[lab]), np.abs(err[lab] - naive[lab])
        out["position"] = dict(notes=int(lab.sum()), acc=round(float((pred[lab] == pos[lab]).mean()), 4),
                               off_share=round(float((pos[lab] == OFF).mean()), 3),
                               err_ms_median={"base": round(float(np.median(e0)) * 1000, 1), "learned": round(float(np.median(e1)) * 1000, 1),
                                              "nearest slot": round(float(np.median(e2)) * 1000, 1)},
                               within_10ms={"base": round(float((e0 < .01).mean()), 3), "learned": round(float((e1 < .01).mean()), 3),
                                            "nearest slot": round(float((e2 < .01).mean()), 3)},
                               worse_than_50ms={"base": round(float((e0 > .05).mean()), 4), "learned": round(float((e1 > .05).mean()), 4),
                                                "nearest slot": round(float((e2 > .05).mean()), 4)})
    net.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="comma-separated cleanup_data dirs")
    ap.add_argument("--beats", required=True)
    ap.add_argument("--split", required=True, help="held-out song list (v1's)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--pos-weight", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    clips = load(a.data.split(","), a.beats, a.workers)
    val_s = set(json.loads(pathlib.Path(a.split).read_text()))
    train = [c for c in clips if C1.song_of(c["name"]) not in val_s]
    val = [c for c in clips if C1.song_of(c["name"]) in val_s]
    songs = {C1.song_of(c["name"]) for c in train}
    print(f"{len(clips)} clips: train {len(train)} clips ({len(songs)} songs), val {len(val)} clips ({len(val_s)} songs); "
          f"with beats {np.mean([c['beats'] for c in clips]):.3f}; position labels "
          f"{sum(int((c['pos'] >= 0).sum()) for c in train)}", flush=True)
    (out / "split.json").write_text(json.dumps(sorted(val_s)))
    net = Cleanup2(a.d, a.depth).to(dev)
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
    la = lp_ = None
    cfg = dict(d=a.d, depth=a.depth, version=2)
    while step < a.steps:
        P, T, X, Y, Q, pad = batch(rng.sample(train, a.bs), dev)
        with torch.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
            lk, lp = net(P, T, X, pad)
        lk = nn.functional.binary_cross_entropy_with_logits(lk.float()[~pad], Y[~pad])
        m = Q >= 0
        lq = nn.functional.cross_entropy(lp.float()[m], Q[m]) if m.any() else torch.zeros((), device=dev)
        loss = lk + a.pos_weight * lq
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        step += 1
        la = lk.item() if la is None else 0.98 * la + 0.02 * lk.item()
        lp_ = lq.item() if lp_ is None else 0.98 * lp_ + 0.02 * lq.item()
        if step % 200 == 0:
            print(f"step {step} loss {la:.4f} pos {lp_:.4f} {(time.time() - t0) / 60:.1f} min", flush=True)
        if step % 2000 == 0 or step == a.steps:
            ev = evaluate(net, val, dev)
            score = max(v["F1"] for k, v in ev.items() if k.startswith("keep"))
            print(f"VAL step {step} " + json.dumps(ev), flush=True)
            state = dict(model=net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), step=step, best=max(best, score), cfg=cfg)
            if score > best:
                best = score
                torch.save(dict(model=net.state_dict(), step=step, score=score, cfg=cfg, val=ev), out / "best.pt")
            torch.save(state, ck)
    print("done", flush=True)


if __name__ == "__main__":
    main()
