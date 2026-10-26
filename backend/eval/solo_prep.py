"""Audio for the solo-voice detector (app/ml/spectral/solo_train.py), at the front end's 24 kHz, mono, float16.

  musdb/<split>/<song>/{vocals,other,bassdrums}.npy   true stems, so solo passages can be made by turning the band down
  m2m/<id>.npy                                        Music2MIDI mixes (labels come from eval/m2m_vocal_curves.py)

    python eval/solo_prep.py --musdb /scratch/basu.ang/datasets/musdb18hq --m2m /scratch/basu.ang/m2m/audio \
        --out /scratch/basu.ang/solo_det [--shard i --nshards n]
"""
import argparse
import pathlib

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SR = 24000


def to24(x, sr):
    x = x.mean(1) if x.ndim > 1 else x
    if sr == 44100:
        return resample_poly(x, 80, 147)
    if sr == 16000:
        return resample_poly(x, 3, 2)
    from math import gcd
    g = gcd(SR, sr)
    return resample_poly(x, SR // g, sr // g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--musdb", required=True)
    ap.add_argument("--m2m", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    jobs = []
    for split in ("train", "test"):
        for d in sorted((pathlib.Path(a.musdb) / split).iterdir()):
            jobs.append(("musdb", split, d))
    for f in sorted(pathlib.Path(a.m2m).glob("*.wav")):
        jobs.append(("m2m", None, f))
    for i, (kind, split, p) in enumerate(jobs):
        if i % a.nshards != a.shard:
            continue
        if kind == "musdb":
            dst = out / "musdb" / split / p.name
            if (dst / "bassdrums.npy").exists():
                continue
            dst.mkdir(parents=True, exist_ok=True)
            st = {k: to24(*sf.read(p / f"{k}.wav", dtype="float32")) for k in ("vocals", "other", "bass", "drums")}
            np.save(dst / "vocals.npy", st["vocals"].astype(np.float16))
            np.save(dst / "other.npy", st["other"].astype(np.float16))
            np.save(dst / "bassdrums.npy", (st["bass"] + st["drums"]).astype(np.float16))
        else:
            dst = out / "m2m" / (p.stem + ".npy")
            if dst.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            np.save(dst, to24(*sf.read(p, dtype="float32")).astype(np.float16))
        print(f"{i + 1}/{len(jobs)} {kind} {p.name}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
