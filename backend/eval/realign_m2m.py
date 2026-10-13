"""Re-align Music2MIDI arrangements to their song audio.

The arrangements are MuseScore scores warped onto the recordings by the
dataset's coarse alignment: only 35.3% of their note onsets fall within 25 ms
of an attack in the song (median 43 ms off), against ~63% for our models'
own notes (eval/timing_offset.py).  Every arranger trained on them learned
from targets that are often 40+ ms off the music.

Two steps, per song:
  coarse  every 2 bars (~4 s), the time shift in +-150 ms that best lines the
          arrangement's onset train up with the song's onset envelope;
          shifts are median-smoothed over neighbouring windows and applied to
          every note in the window (offsets move with their onsets)
  fine    each note onset then snaps to an attack peak within +-35 ms when one
          exists; notes with no attack nearby keep the coarse time

Reported: share of onsets within 25 / 50 ms of an attack, before and after.
Output: <out>/<id>.npy, same [on, off, pitch, vel] format as midi_numpy.

    python eval/realign_m2m.py --m2m /scratch/.../m2m --out /scratch/.../m2m/midi_numpy_aligned [--shard i/n]
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

SR, HOP = 22050, 128
DT = HOP / SR
CONV = 0.026            # onset-envelope peak -> attack, measured on real MAESTRO


def attacks(wav):
    import librosa
    from scipy.signal import find_peaks
    y, _ = librosa.load(str(wav), sr=SR, mono=True)
    e = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP, lag=1, max_size=1)
    e = (e - e.mean()) / (e.std() + 1e-9)
    pk, _ = find_peaks(e, height=1.0, distance=int(0.04 / DT))
    return e, pk * DT - CONV


def realign(notes, env, tp, win=4.0, max_shift=0.15):
    n = notes[np.argsort(notes[:, 0], kind="stable")].copy()
    T = len(env)
    t_env = np.arange(T) * DT - CONV
    starts = np.arange(n[0, 0], n[-1, 0] + win, win)
    shifts = []
    L = int(max_shift / DT)
    for s in starts:
        sel = n[(n[:, 0] >= s) & (n[:, 0] < s + win), 0]
        if len(sel) < 4:
            shifts.append(np.nan); continue
        best, bk = -1e9, 0
        for k in range(-L, L + 1):
            idx = np.round((sel + k * DT + CONV) / DT).astype(int)
            idx = idx[(idx >= 0) & (idx < T)]
            sc = env[idx].sum() if len(idx) else -1e9
            if sc > best:
                best, bk = sc, k
        shifts.append(bk * DT)
    sh = np.array(shifts)
    good = np.isfinite(sh)
    if not good.any():
        return n, 0.0
    sh[~good] = np.interp(np.flatnonzero(~good), np.flatnonzero(good), sh[good])
    sm = np.array([np.median(sh[max(0, i - 2):i + 3]) for i in range(len(sh))])      # smooth over ~5 windows
    per = np.interp(n[:, 0], starts + win / 2, sm)
    n[:, :2] += per[:, None]
    # fine: snap to an attack within 35 ms
    j = np.clip(np.searchsorted(tp, n[:, 0]), 1, len(tp) - 1)
    near = np.where(np.abs(tp[j] - n[:, 0]) < np.abs(tp[j - 1] - n[:, 0]), tp[j], tp[j - 1])
    d = near - n[:, 0]
    snap = np.abs(d) <= 0.035
    n[snap, 0] += d[snap]; n[snap, 1] += d[snap]
    n[:, 0] = np.maximum(n[:, 0], 0); n[:, 1] = np.maximum(n[:, 1], n[:, 0] + 0.03)
    return n, float(np.median(np.abs(sm)))


def on_attack(notes, tp):
    on = np.unique(np.round(notes[:, 0], 3))
    j = np.clip(np.searchsorted(tp, on), 1, len(tp) - 1)
    return np.minimum(np.abs(on - tp[j - 1]), np.abs(on - tp[j]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2m", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()
    m2m, out = pathlib.Path(args.m2m), pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sp = json.loads((m2m / "m2m_splits.json").read_text())
    ids = sorted(sp["train"] + sp["validation"] + sp["test"])
    si, sn = (int(v) for v in args.shard.split("/"))
    before, after, shifts = [], [], []
    for sid in ids[si::sn]:
        notes = np.load(m2m / "midi_numpy" / f"{sid}.npy")
        if len(notes) < 10:
            np.save(out / f"{sid}.npy", notes); continue
        env, tp = attacks(m2m / "audio" / f"{sid}.wav")
        new, med = realign(notes, env, tp)
        np.save(out / f"{sid}.npy", new.astype(notes.dtype))
        b, a = on_attack(notes, tp), on_attack(new, tp)
        before.append(b); after.append(a); shifts.append(med)
        print(f"  {sid}: on-attack(25ms) {np.mean(b <= .025):.0%} -> {np.mean(a <= .025):.0%}, "
              f"median window shift {med * 1000:.0f} ms", flush=True)
    b, a = np.concatenate(before), np.concatenate(after)
    print(f"\n{len(before)} songs: onsets within 25 ms of an attack {np.mean(b <= .025):.1%} -> {np.mean(a <= .025):.1%}; "
          f"within 50 ms {np.mean(b <= .05):.1%} -> {np.mean(a <= .05):.1%}; median distance "
          f"{np.median(b) * 1000:.1f} -> {np.median(a) * 1000:.1f} ms; median coarse shift {np.median(shifts) * 1000:.0f} ms")


if __name__ == "__main__":
    main()
