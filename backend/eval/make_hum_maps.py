"""Hums -> their per-pitch loudness maps: the native form for the hum-to-hum model (CPU).

Each hum (eval/make_pair_hums.py) is a sum of sines, one per Basic Pitch bin
(3 per semitone from A0, 264 bins).  A CQT on exactly those bins (36 per
octave from 27.5 Hz, hop 256 at 22.05 kHz = the hum's own frame grid) reads
each sine's loudness back out, so the map IS the hum in the form it was built
from, and renders back to audio with the same sine renderer.  Nothing is
thresholded or snapped.

Per window, one file holding both sides:
    <out>/<split>/<id>_<w>.npz   a = song-hum map, b = piano-hum map   (264, T) float16, difficulty

    python eval/make_hum_maps.py --hums /scratch/.../pair_hums --out <dir> --shard i/n
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np

SR, HOP, FMIN, BINS, BPO = 22050, 256, 27.5, 264, 36


def hum_map(y24: np.ndarray) -> np.ndarray:
    import librosa
    from scipy.signal import resample_poly
    y = resample_poly(y24.astype(np.float32), 147, 160).astype(np.float32)          # 24 kHz -> 22.05 kHz
    return np.abs(librosa.cqt(y, sr=SR, hop_length=HOP, fmin=FMIN, n_bins=BINS, bins_per_octave=BPO))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hums", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1"); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    si, sn = (int(v) for v in a.shard.split("/"))
    root = pathlib.Path(a.hums)
    done = 0
    for split in ("validation", "test", "train"):
        paths = sorted(p for p in (root / "song" / split).glob("*.npz") if not p.name.endswith(".tmp.npz"))
        out = pathlib.Path(a.out) / split; out.mkdir(parents=True, exist_ok=True)
        for ps in paths[si::sn][: a.limit or None]:
            pp = root / "piano" / split / ps.name
            dst = out / ps.name
            if dst.exists() or not pp.exists():
                continue
            zs, zp = np.load(ps), np.load(pp)
            A = hum_map(zs["y"].astype(np.float32) / 32767); B = hum_map(zp["y"].astype(np.float32) / 32767)
            n = min(A.shape[1], B.shape[1])
            t = dst.with_name(dst.stem + ".tmp.npz")
            np.savez_compressed(t, a=A[:, :n].astype(np.float16), b=B[:, :n].astype(np.float16),
                                difficulty=zs["difficulty"])
            t.replace(dst); done += 1
            if done % 200 == 0:
                print(f"  {done} maps", flush=True)
    print(f"{done} windows mapped", flush=True)


if __name__ == "__main__":
    main()
