"""Slim Salamander for CPU hosting: 8 of the 16 velocity layers (each kept layer takes its neighbour's range), samples
as 44.1 kHz 16-bit FLAC (the previews render at 44.1 kHz).  Release, harmonic and pedal samples are kept.

    python deploy/slim_salamander.py <SalamanderGrandPiano dir> deploy/piano
"""
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

KEEP = [2, 4, 6, 8, 10, 12, 14, 16]
SR = 44100


def convert(job):
    src, dst = job
    y, sr = sf.read(src, dtype="float32", always_2d=True)
    if sr != SR:
        g = gcd(SR, sr)
        y = resample_poly(y, SR // g, sr // g, axis=0)
    sf.write(dst, np.clip(y, -1, 1), SR, subtype="PCM_16", format="FLAC")


def main(src, dst):
    src, dst = Path(src), Path(dst)
    (dst / "Samples").mkdir(parents=True, exist_ok=True)
    shutil.copytree(src / "Data", dst / "Data", dirs_exist_ok=True)
    shutil.copy(src / "Salamander Grand Piano V3.sfz", dst)
    # notes.txt: one group per kept layer, covering its own range and the dropped layer below it
    lines = (src / "Data" / "notes.txt").read_text().splitlines()
    groups = [l for l in lines if l.startswith("<group>")]
    rng = [tuple(map(int, re.search(r"lovel=(\d+) hivel=(\d+)", g).groups())) for g in groups]
    out = [l for l in lines if not l.startswith("<group>")]
    for k in KEEP:
        lo, hi = rng[k - 2][0], rng[k - 1][1]
        out.append(f'<group> #include "Data/vel_{k:02d}.txt" lovel={lo} hivel={hi} #include "Data/region.txt"')
    (dst / "Data" / "notes.txt").write_text("\n".join(out) + "\n")
    keep_v = {f"v{k}" for k in KEEP}
    jobs = []
    for f in sorted((src / "Samples").glob("*.flac")):
        m = re.search(r"(v\d+)\.flac$", f.name)
        if m and m.group(1) not in keep_v:
            continue
        jobs.append((str(f), str(dst / "Samples" / f.name)))
    with ProcessPoolExecutor() as ex:
        list(ex.map(convert, jobs, chunksize=8))
    print(f"{len(jobs)} samples -> {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
