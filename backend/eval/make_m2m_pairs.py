"""Music2MIDI song windows paired with the SAME window of the human arrangement, played on a piano.

The "imagination" data: A = the song (as heard), B = what a pianist would play
for it, as AUDIO, so the two can be compared in the same pitch space (HCQT).
B is the human arrangement (re-aligned notes, midi_numpy_aligned) rendered
through one of the sampled pianos, chosen per song so every piano is seen.
Each piano's onset latency is subtracted before rendering, so B's attacks land
on the note times — the same times the song's attacks were aligned to.

Reads the existing song clips (<split>/<id>_<w>.npz, 20 s @ 24 kHz) and writes
<out>/<split>/<id>_<w>.npz with
    y   song audio (int16)          yp  rendered arrangement (int16)
    on, fr  aligned arrangement rolls (88, T)     piano, difficulty

    python eval/make_m2m_pairs.py --m2m /scratch/.../m2m --out <dir> --shard i/n
"""
from __future__ import annotations

import argparse
import pathlib
import random
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

SECONDS = 20.0
PRE = 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2m", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1"); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    from app.ml.spectral import frontend as S, render as R
    from make_m2m_clips import rolls
    names = sorted(R.pianos()); lat = R.latencies()
    print("pianos:", names, flush=True)
    m2m = pathlib.Path(a.m2m)
    si, sn = (int(v) for v in a.shard.split("/"))
    n_samp = int(SECONDS * S.SR); T = S.n_frames(n_samp)
    done = 0
    for split in ("train", "validation", "test"):
        src = sorted((m2m / "clips" / split).glob("*_[0-9][0-9][0-9].npz"))
        songs = sorted({p.stem.rsplit("_", 1)[0] for p in src})[si::sn]
        out = pathlib.Path(a.out) / split; out.mkdir(parents=True, exist_ok=True)
        for sid in songs[: a.limit or None]:
            notes = np.load(m2m / "midi_numpy_aligned" / f"{sid}.npy").astype(np.float64)
            piano = random.Random(sid).choice(names)
            for p in sorted((m2m / "clips" / split).glob(f"{sid}_[0-9][0-9][0-9].npz")):
                dst = out / p.name
                if dst.exists():
                    continue
                w = int(p.stem.rsplit("_", 1)[1]); t0 = w * SECONDS
                base = max(0.0, t0 - PRE); crop = t0 - base; span = crop + SECONDS
                sel = notes[(notes[:, 1] > base) & (notes[:, 0] < t0 + SECONDS)]
                ev = [{"pitch": int(q), "velocity": int(np.clip(v, 1, 127)),
                       "onset": max(0.0, s - base - lat[piano]),
                       "offset": max(0.01, min(span, e - base - lat[piano]))} for s, e, q, v in sel]
                y = R.render_with(piano, ev) if ev else np.zeros(int(span * S.SR), np.float32)
                k = int(round(crop * S.SR)); y = y[k:k + n_samp]
                y = np.pad(y, (0, max(0, n_samp - len(y))))
                pk = float(np.abs(y).max()) if len(y) else 0.0
                if pk > 0.95:
                    y = y * (0.95 / pk)
                lab = sel.copy(); lab[:, :2] -= t0
                on, fr, _ = rolls(lab, T, S.FPS)
                z = np.load(p)
                tmp = dst.with_name(dst.stem + ".tmp.npz")
                np.savez_compressed(tmp, y=z["y"], yp=(np.clip(y, -1, 1) * 32767).astype(np.int16),
                                    on=on, fr=fr, piano=piano, difficulty=z["difficulty"])
                tmp.replace(dst); done += 1
            print(f"  {split} {sid} ({piano})", flush=True)
    print(f"{done} pairs written", flush=True)


if __name__ == "__main__":
    main()
