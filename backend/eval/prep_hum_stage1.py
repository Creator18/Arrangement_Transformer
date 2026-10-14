"""Data layout for stage 1 FROM SCRATCH: piano hum -> notes, trained on classical AND pop-arrangement hums.

  <out>/m2m/{train,validation,test}/   the Music2MIDI PIANO hums (eval/make_pair_hums.py), re-sorted
                                       BY SONG with eval/m2m_bysong.py's units — no song on both sides
  <out>/val_mixed/validation/          the validation set the trainer scores on: --n-each MAESTRO hums
                                       + --n-each Music2MIDI hums, so one number covers both styles
                                       (prefixed mae_ / m2m_; eval/hum_s1_checks.py reports them apart)

Symlinks only (velocity files linked alongside when they exist), nothing is copied.
Then: python -m app.ml.spectral.train --data <maestro piano_hums> --data-real <out>/m2m
      --real-frac 0.5 --val-data <out>/val_mixed ...

    python eval/prep_hum_stage1.py --maestro /scratch/.../piano_hums --pair /scratch/.../pair_hums \\
        --units m2m_units.json --out /scratch/.../hum_s1_data
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import random


def link(src: pathlib.Path, dst: pathlib.Path):
    if not dst.exists():
        os.symlink(src, dst)
    v = src.with_name(src.stem + ".vel.npy")
    if v.exists() and not (dst.with_name(dst.stem + ".vel.npy")).exists():
        os.symlink(v, dst.with_name(dst.stem + ".vel.npy"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maestro", required=True); ap.add_argument("--pair", required=True)
    ap.add_argument("--units", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n-each", type=int, default=150)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    U = json.loads(pathlib.Path(a.units).read_text())
    split_of = {i: k for k, us in U["units"].items() for u in us for i in u}
    piano = collections.defaultdict(list)
    for p in (pathlib.Path(a.pair) / "piano").glob("*/*.npz"):
        if not p.name.endswith(".tmp.npz"):
            piano[p.stem.rsplit("_", 1)[0]].append(p)
    n = collections.Counter()
    for sid, ps in piano.items():
        k = split_of.get(sid)
        if k is None:
            continue
        d = out / "m2m" / k; d.mkdir(parents=True, exist_ok=True)
        for p in ps:
            link(p, d / p.name); n[k] += 1
    print("Music2MIDI piano hums by song:", dict(n))
    vm = out / "val_mixed" / "validation"; vm.mkdir(parents=True, exist_ok=True)
    mae = sorted(p for p in (pathlib.Path(a.maestro) / "validation").glob("*.npz") if not p.name.endswith(".tmp.npz"))
    m2m = sorted((out / "m2m" / "validation").glob("*.npz"))
    for tag, ps in (("mae", mae), ("m2m", m2m)):
        for p in random.Random(0).sample(ps, min(a.n_each, len(ps))):
            link(p.resolve(), vm / f"{tag}_{p.name}")
    print(f"val_mixed: {len(list(vm.glob('*.npz')))} clips ({a.n_each} MAESTRO + {a.n_each} Music2MIDI)")


if __name__ == "__main__":
    main()
