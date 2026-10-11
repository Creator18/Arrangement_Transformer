"""Real MAESTRO recordings -> training clips, in the rendered clips' exact format.

The spectral model scores F1 0.968 on rendered audio and 0.900 on real
recordings, and the loss is concentrated in soft notes (recall 0.655 against
ByteDance's 0.838, which trained on real recordings).  The remedy is real
audio in training.

Runs ON THE CLUSTER: each member of the 108 GB MAESTRO archive is read by HTTP
range requests straight into memory (see fetch_maestro_real.HttpRangeFile),
converted to 24 kHz mono, cut into 20 s windows and labelled from the MIDI the
Disklavier captured — aligned to the audio within ~3 ms, so unlike rendered
clips there is no sampler latency to correct for.

Output matches render_multi_full.py: <i>_<w>.npz (int16 audio, onset and frame
rolls, piano="real") plus <i>_<w>.vel.npy, so data.RenderedClips reads both
kinds unchanged.  Resumable and atomic, like the renderer.

    python eval/fetch_real_clips.py --split train --shard 3/48 --out /scratch/.../maestro_real_clips
"""
from __future__ import annotations

import argparse
import io
import math
import pathlib
import sys
import warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
warnings.filterwarnings("ignore")

SECONDS = 20.0

# Label times are shifted to where the ATTACK reaches half its peak — the
# convention every rendered piano uses (render.latencies(), 7.5-15.5 ms).  The
# Disklavier's MIDI marks the key event, and the tone builds a little later:
# measured with the same rule (1 ms RMS envelope, band-passed around the note)
# on 55 isolated notes in 12 real recordings, +8.6 ms (IQR 4.6-12.7).  Without
# the shift, onset alignment on real clips peaked a frame early, and mixing the
# two sources would have taught two different ideas of "the note starts".
REAL_LATENCY_S = 0.0086


def labels(pm, t0: float, seconds: float):
    """(onset, frame, velocity) rolls for one window, on the half-peak convention."""
    from app.ml.spectral import frontend as S
    T = S.n_frames(int(seconds * S.SR))
    on = np.zeros((S.N_KEYS, T), np.uint8)
    fr = np.zeros((S.N_KEYS, T), np.uint8)
    vel = np.zeros((S.N_KEYS, T), np.uint8)
    f = lambda t: int(round((t + REAL_LATENCY_S) * S.FPS))   # centred frames: round, not floor
    for inst in pm.instruments:
        for n in inst.notes:
            s, e = n.start - t0, n.end - t0
            k = n.pitch - 21
            if e <= 0 or s >= seconds or not 0 <= k < S.N_KEYS:
                continue
            a, b = f(max(0.0, s)), f(min(seconds, e))
            fr[k, max(0, a):min(T, max(b, a + 1))] = 1
            if s >= 0 and a < T:
                on[k, a] = 1
                vel[k, a] = max(vel[k, a], int(n.velocity))
    return on, fr, vel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "validation", "test"])
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import csv, zipfile, pretty_midi, soundfile as sf
    from scipy.signal import resample_poly
    from fetch_maestro_real import HttpRangeFile, URL, MAESTRO
    from app.ml.spectral import frontend as S

    rows = [r for r in csv.DictReader(open(MAESTRO / "maestro-v3.0.0.csv", encoding="utf-8"))
            if r["split"] == args.split]
    si, sn = (int(v) for v in args.shard.split("/"))
    jobs = list(enumerate(rows))[si::sn]
    out = pathlib.Path(args.out) / args.split
    out.mkdir(parents=True, exist_ok=True)

    zf = zipfile.ZipFile(io.BufferedReader(HttpRangeFile(URL), buffer_size=1 << 20))
    names = {n.split("/", 1)[-1]: n for n in zf.namelist()}
    n_win = 0
    for i, r in jobs:
        windows = max(1, math.ceil(float(r["duration"]) / SECONDS))
        if all((out / f"{i:04d}_{w:03d}.npz").exists() for w in range(windows)):
            continue
        y, sr = sf.read(io.BytesIO(zf.read(names[r["audio_filename"]])), dtype="float32",
                        always_2d=True)
        y = y.mean(axis=1)
        g = math.gcd(S.SR, sr)
        y = resample_poly(y, S.SR // g, sr // g).astype(np.float32)
        pm = pretty_midi.PrettyMIDI(str(MAESTRO / r["midi_filename"]))
        seg = int(SECONDS * S.SR)
        for w in range(windows):
            dst = out / f"{i:04d}_{w:03d}.npz"
            if dst.exists():
                continue
            clip = y[w * seg:(w + 1) * seg]
            clip = np.pad(clip, (0, seg - len(clip)))
            on, fr, vel = labels(pm, w * SECONDS, SECONDS)
            tmp = dst.with_name(dst.stem + ".tmp.npz")
            np.savez_compressed(tmp, y=(np.clip(clip, -1, 1) * 32767).astype(np.int16),
                                on=on, fr=fr, piano="real")
            tmp.replace(dst)
            vt = dst.with_name(dst.stem + ".vel.tmp.npy")
            np.save(vt, vel)
            vt.replace(dst.with_name(dst.stem + ".vel.npy"))
            n_win += 1
        print(f"  {args.split} {i:4d}  {float(r['duration']) / 60:5.1f} min  {windows} windows", flush=True)
    print(f"done: {n_win} windows written")


if __name__ == "__main__":
    main()
