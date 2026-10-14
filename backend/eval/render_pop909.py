"""POP909 piano arrangements -> rendered piano clips for fine-tuning the solo-piano transcriber (CPU).

All three POP909 tracks (MELODY, BRIDGE, PIANO) are one piano arrangement of
the song, so they are rendered together, through one of the sampled pianos
drawn per song.  Same format and the same rendering path as the MAESTRO set
(app.ml.spectral.render.render_window: 2 s pre-roll so held notes ring in,
per-piano latency corrected in the labels, pedal not rendered so notes end
at key-up), so data.RenderedClips reads it unchanged:

    <out>/<split>/pop<NNN>_<w>.npz  (y int16 24 kHz, on, fr, piano)  + .vel.npy

Split BY SONG, fixed: song number % 10 == 0 -> validation, == 1 -> test, else train.

    python eval/render_pop909.py --pop909 /scratch/.../POP909 --out <dir> --shard i/n
"""
from __future__ import annotations

import argparse
import math
import pathlib
import random
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SECONDS = 20.0


def split_of(n: int) -> str:
    return "validation" if n % 10 == 0 else "test" if n % 10 == 1 else "train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop909", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1"); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    import pretty_midi
    from app.ml.spectral import render as R
    names = sorted(R.pianos())
    si, sn = (int(v) for v in a.shard.split("/"))
    songs = sorted(p for p in pathlib.Path(a.pop909).iterdir() if p.is_dir() and p.name.isdigit())[si::sn]
    done = 0
    for d in songs[: a.limit or None]:
        n = int(d.name); split = split_of(n)
        out = pathlib.Path(a.out) / split; out.mkdir(parents=True, exist_ok=True)
        pm = pretty_midi.PrettyMIDI(str(d / f"{d.name}.mid"))
        for inst in pm.instruments:
            inst.is_drum = False
        piano = random.Random(f"pop909-{n}").choice(names)
        end = max((nt.end for i in pm.instruments for nt in i.notes), default=0.0)
        for w in range(max(1, math.ceil(end / SECONDS))):
            dst = out / f"pop{d.name}_{w:03d}.npz"
            if dst.exists():
                continue
            y, on, fr = R.render_window(pm, w * SECONDS, SECONDS, piano)
            vel = R.velocity_roll(pm, w * SECONDS, SECONDS, piano)
            t = dst.with_name(dst.stem + ".tmp.npz")
            np.savez_compressed(t, y=(np.clip(y, -1, 1) * 32767).astype(np.int16), on=on, fr=fr, piano=piano)
            t.replace(dst)
            vt = dst.with_name(dst.stem + ".vel.tmp.npy"); np.save(vt, vel); vt.replace(dst.with_name(dst.stem + ".vel.npy"))
            done += 1
        print(f"  {split} {d.name} ({piano})", flush=True)
    print(f"{done} clips written", flush=True)


if __name__ == "__main__":
    main()
