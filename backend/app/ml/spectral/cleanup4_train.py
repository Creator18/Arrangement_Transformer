"""Clean-up v4 (2026-10-29): v3 + a "no human keeps this note" output (the user's idea: learn from the human
arrangements only which of our notes humans NEVER keep, not what or how they play).

eval/never_kept.py measured it on 925 Music2MIDI songs: 32% of our notes are in no arrangement of the song vs 25%
for one human's notes judged the same way (arrangers disagree); the excess sits where the voice sounds, and the
quick out-of-key slide notes are 84.5% never kept.

Training mixes two kinds of batch every step:
    renders     (as v3)  keep / merge / beat-position from the written arrangement      -> keeps v3's skills
    real songs  (new)    "never kept" = no arrangement of the song has the pitch class within 150 ms
                          (eval/human_data.py), weighted by how many arrangements the song has (cleaner labels)
The real songs train this output only (the base stays renders-only; the clean-up may learn from real songs, like
the solo detector).  Held out: the songs touching the Music2MIDI test split (one song audio = one unit).
In the app the new output only ever DROPS a note v3 kept (app/services/cleanup.py, ARRANGEMENT_CLEANUP_NK).

    python -m app.ml.spectral.cleanup4_train --data /scratch/basu.ang/cleanup_data,/scratch/basu.ang/cleanup_data_tr \
        --beats /scratch/basu.ang/clip_beats --labels /scratch/basu.ang/merge_labels --human /scratch/basu.ang/human_data \
        --songs /scratch/basu.ang/never_kept/songs.json --init checkpoints/cleanup3/best.pt \
        --split checkpoints/cleanup1/split.json --out checkpoints/cleanup4 --budget-min 105
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
from app.ml.spectral import cleanup3_train as C3
from app.ml.spectral import cleanup_train as C1

MAXN = C2.MAXN
WIN_S = 20.0
NK_PRIOR = 0.35


class Cleanup4(C3.Cleanup3):
    def __init__(self, d=256, depth=6, heads=4, no_beats=False):
        super().__init__(d, depth, heads)
        self.nk = nn.Linear(d, 1)
        nn.init.zeros_(self.nk.weight)
        nn.init.constant_(self.nk.bias, math.log(NK_PRIOR / (1 - NK_PRIOR)))
        self.no_beats = no_beats                                # the user's v4 without madmom: beat inputs held at 0

    def forward(self, pitch, t, x, pad, with_nk=False):
        if self.no_beats:
            x = torch.cat([x[..., :-C2.NBEAT], torch.zeros_like(x[..., -C2.NBEAT:])], -1)
        h = self.pitch(pitch) + self.feat(x) + self.time_emb(t)
        e = self.enc(h, src_key_padding_mask=pad)
        o = self.head(e)
        e0 = self.head[0](e)
        out = (o[..., 0], o[..., 1:], self.merge(e0)[..., 0])
        return out + (self.nk(e0)[..., 0],) if with_nk else out


def weight(nref):
    """labels from more arrangements are cleaner (never kept 37% with 1, 19% with 3+, 9% with 9+)."""
    return min(3, nref) / 2.0


def load_human(d, songs_json):
    held = {s["audio"] for s in json.loads(pathlib.Path(songs_json).read_text())["songs"] if s["split"] == "test"}
    songs = []
    for f in sorted(pathlib.Path(d).glob("*.npz")):
        if ".tmp" in f.name:
            continue
        z = np.load(f)
        songs.append(dict(name=f.stem, notes=z["notes"], x=z["x"], kept=z["kept"], nref=int(z["nref"]), win=z["win"],
                          held=f.stem in held))
    clips = []
    for si, s in enumerate(songs):
        on = s["notes"][:, 0]
        for w0 in s["win"]:
            idx = np.nonzero((on >= w0) & (on < w0 + WIN_S))[0][:MAXN]
            if len(idx):
                clips.append(dict(song=si, idx=idx, w0=float(w0)))
    return songs, clips


def human_batch(songs, clips, dev):
    B, N = len(clips), max(len(c["idx"]) for c in clips)
    P = torch.zeros(B, N, dtype=torch.long); T = torch.zeros(B, N); X = torch.zeros(B, N, C2.NFEAT)
    Y = torch.zeros(B, N); W = torch.zeros(B, N); pad = torch.ones(B, N, dtype=torch.bool)
    for i, c in enumerate(clips):
        s, idx = songs[c["song"]], c["idx"]
        n = len(idx)
        P[i, :n] = torch.from_numpy(s["notes"][idx, 2].astype(np.int64))
        T[i, :n] = torch.from_numpy((s["notes"][idx, 0] - c["w0"]).astype(np.float32))
        X[i, :n] = torch.from_numpy(s["x"][idx].astype(np.float32))
        Y[i, :n] = torch.from_numpy(1.0 - s["kept"][idx].astype(np.float32))
        W[i, :n] = weight(s["nref"]); pad[i, :n] = False
    return P.to(dev), T.to(dev), X.to(dev), Y.to(dev), W.to(dev), pad.to(dev)


@torch.no_grad()
def song_probs(net, s, dev):
    """keep / merge / never-kept per note over the whole song, the app's way: 20 s windows every 10 s, centre-weighted."""
    n, N = s["notes"], len(s["notes"])
    acc = np.zeros((N, 3)); ws = np.zeros(N)
    for w0 in s["win"]:
        idx = np.nonzero((n[:, 0] >= w0) & (n[:, 0] < w0 + WIN_S))[0]
        for part in np.array_split(idx, max(1, int(np.ceil(len(idx) / MAXN)))) if len(idx) else []:
            P = torch.from_numpy(n[part, 2].astype(np.int64))[None].to(dev)
            Tt = torch.from_numpy((n[part, 0] - w0).astype(np.float32))[None].to(dev)
            X = torch.from_numpy(s["x"][part].astype(np.float32))[None].to(dev)
            lk, _, lm, ln = net(P, Tt, X, torch.zeros(1, len(part), dtype=torch.bool, device=dev), with_nk=True)
            p = torch.sigmoid(torch.stack([lk, lm, ln], -1).float())[0].cpu().numpy()
            w = 1.0 - np.abs(n[part, 0] - (w0 + WIN_S / 2)) / (WIN_S / 2) + 1e-3
            acc[part] += w[:, None] * p; ws[part] += w
    return acc / np.maximum(ws, 1e-9)[:, None]


def decide_idx(n, pk, pm, keep_bar=0.3, merge_bar=0.5):
    """v3's decision (cleanup3_train.decide), returning the indices of the base notes that start an output note."""
    keep = []
    for p in np.unique(n[:, 2]).astype(int):
        idx = np.nonzero(n[:, 2] == p)[0]
        idx = idx[np.argsort(n[idx, 0], kind="stable")]
        alive = False
        for i in idx:
            if alive and pm[i] >= merge_bar:
                continue
            if pk[i] >= keep_bar:
                keep.append(i); alive = True
            else:
                alive = False
    return np.array(sorted(keep), int)


@torch.no_grad()
def evaluate_human(net, songs, dev, bars=(0.5, 0.6, 0.7, 0.8, 0.9)):
    """on held-out real songs, among the notes v3's decision outputs: AUC of the new output for "never kept", and per
    drop bar: notes/s, share never kept, share of the human-kept notes lost."""
    from sklearn.metrics import roc_auc_score
    if not songs:
        return dict(auc=-1.0)
    net.eval()
    P_, Y_, secs = [], [], 0.0
    for s in songs:
        pr = song_probs(net, s, dev)
        k = decide_idx(s["notes"], pr[:, 0], pr[:, 1])
        P_.append(pr[k, 2]); Y_.append(1 - s["kept"][k].astype(int))
        secs += float(s["notes"][:, 0].max())
    net.train()
    p, y = np.concatenate(P_), np.concatenate(Y_)
    out = dict(auc=round(float(roc_auc_score(y, p)), 4) if 0 < y.mean() < 1 else -1.0, v3=dict(nps=round(len(p) / secs, 2), never=round(float(y.mean()), 4)))
    for b in bars:
        m = p < b
        out[f"nk {b}"] = dict(nps=round(m.sum() / secs, 2), never=round(float(y[m].mean()), 4),
                              kept_lost=round(float(((~m) & (y == 0)).sum() / max(1, (y == 0).sum())), 4),
                              dropped=round(float((~m).mean()), 4))
    return out


def main():
    ap = argparse.ArgumentParser()
    for k in ("data", "beats", "labels", "human", "songs", "init", "split", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--nk-weight", type=float, default=1.0)
    ap.add_argument("--merge-weight", type=float, default=1.0)
    ap.add_argument("--pos-weight", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--val-clips", type=int, default=600)
    ap.add_argument("--budget-min", type=float, default=0)
    ap.add_argument("--no-beats", action="store_true", help="beat inputs held at 0 (no madmom); no position loss")
    a = ap.parse_args()
    if a.no_beats:
        a.pos_weight = 0.0                                      # the slot labels come from madmom's beats
    t_start = time.time()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ck = out / "last.pt"
    if ck.exists() and torch.load(ck, map_location="cpu", weights_only=False)["step"] >= a.steps:
        print("already finished", flush=True)
        return
    dev = "cuda"
    rclips = C3.load(a)
    val_s = set(json.loads(pathlib.Path(a.split).read_text()))
    rtrain = [c for c in rclips if C1.song_of(c["name"]) not in val_s]
    rval = random.Random(0).sample([c for c in rclips if C1.song_of(c["name"]) in val_s], a.val_clips)
    songs, hclips = load_human(a.human, a.songs)
    htrain = [c for c in hclips if not songs[c["song"]]["held"]]
    hval = [s for s in songs if s["held"]]
    ntr = np.concatenate([songs[i]["kept"] for i in {c["song"] for c in htrain}])
    print(f"renders: {len(rtrain)} train clips;  real songs: {len(songs)} ({len(hval)} held out), {len(htrain)} train "
          f"windows, never kept {1 - ntr.mean():.3f}", flush=True)
    s = torch.load(a.init, map_location="cpu", weights_only=False)
    cfg = dict(s["cfg"]); cfg.pop("version", None)
    if a.no_beats:
        cfg["no_beats"] = True
    net = Cleanup4(**cfg)
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
        print("VAL step 0 renders " + json.dumps(C3.evaluate(net, rval, a.data, dev)), flush=True)
        print("VAL step 0 human " + json.dumps(evaluate_human(net, hval, dev)), flush=True)
    cfg = dict(cfg, version=4)
    rng = random.Random(step)
    t0 = time.time()
    la = None

    def save_last():
        torch.save(dict(model=net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), step=step, best=best,
                        cfg=cfg), ck)
    amp = dict(device_type="cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    while step < a.steps:
        P, T, X, Y, Q, pad, A = C3.batch(rng.sample(rtrain, a.bs), dev)
        with torch.autocast(**amp):
            lk, lp, lm = net(P, T, X, pad)
        m = A >= 0
        l_keep = nn.functional.binary_cross_entropy_with_logits(lk.float()[m], (A[m] == 1).float())
        l_merge = nn.functional.binary_cross_entropy_with_logits(lm.float()[m], (A[m] == 2).float())
        mq = Q >= 0
        l_pos = nn.functional.cross_entropy(lp.float()[mq], Q[mq]) if mq.any() else torch.zeros((), device=dev)
        hP, hT, hX, hY, hW, hpad = human_batch(songs, rng.sample(htrain, a.bs), dev)
        with torch.autocast(**amp):
            ln = net(hP, hT, hX, hpad, with_nk=True)[3]
        hm = ~hpad
        l_nk = (nn.functional.binary_cross_entropy_with_logits(ln.float()[hm], hY[hm], reduction="none") * hW[hm]).sum() / hW[hm].sum()
        loss = l_keep + a.merge_weight * l_merge + a.pos_weight * l_pos + a.nk_weight * l_nk
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        step += 1
        cur = np.array([l_keep.item(), l_merge.item(), l_nk.item()])
        la = cur if la is None else 0.98 * la + 0.02 * cur
        if step % 200 == 0:
            print(f"step {step} keep {la[0]:.4f} merge {la[1]:.4f} never-kept {la[2]:.4f} {(time.time() - t0) / 60:.1f} min", flush=True)
        if step % 2000 == 0 or step == a.steps:
            ev = evaluate_human(net, hval, dev)
            er = C3.evaluate(net, rval, a.data, dev)
            print(f"VAL step {step} human " + json.dumps(ev), flush=True)
            print(f"VAL step {step} renders " + json.dumps(er), flush=True)
            if ev["auc"] > best:
                best = ev["auc"]
                torch.save(dict(model=net.state_dict(), step=step, score=best, cfg=cfg, val=dict(human=ev, renders=er)),
                           out / "best.pt")
            save_last()
        if a.budget_min and (time.time() - t_start) / 60 > a.budget_min:
            save_last()
            print(f"budget reached at step {step}: saved, exiting for the next piece", flush=True)
            return
    print("done", flush=True)


if __name__ == "__main__":
    main()
