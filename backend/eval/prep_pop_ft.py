"""Data layout for the POP909 LoRA fine-tune of the solo-piano transcriber (symlinks only).

  <out>/train/        every POP909 train clip (pop_) + MAESTRO so that MAESTRO is --mae-frac
                      of the set, half real recordings (maer_), half rendered (maes_) —
                      the adapters learn pop textures without drifting on classical
  <out>/validation/   --n-val POP909 validation clips (pop_) + --n-val REAL MAESTRO
                      validation recordings (mae_): one number in training, reported
                      apart by eval/hum_s1_checks.py

    python eval/prep_pop_ft.py --pop /scratch/.../pop909_render --mae-real /scratch/.../maestro_real_clips \\
        --mae-render /scratch/.../maestro_render_multi --out /scratch/.../pop_ft
"""
from __future__ import annotations

import argparse
import os
import pathlib
import random


def clips(d):
    return sorted(p for p in pathlib.Path(d).glob("*.npz") if not p.name.endswith(".tmp.npz"))


def link(src, dst):
    if not dst.exists():
        os.symlink(src, dst)
    v = src.with_name(src.stem + ".vel.npy")
    dv = dst.with_name(dst.stem + ".vel.npy")
    if v.exists() and not dv.exists():
        os.symlink(v, dv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", required=True); ap.add_argument("--mae-real", required=True)
    ap.add_argument("--mae-render", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--mae-frac", type=float, default=0.25); ap.add_argument("--n-val", type=int, default=150)
    a = ap.parse_args()
    rng = random.Random(0)
    tr = pathlib.Path(a.out) / "train"; va = pathlib.Path(a.out) / "validation"
    tr.mkdir(parents=True, exist_ok=True); va.mkdir(parents=True, exist_ok=True)
    pop = clips(pathlib.Path(a.pop) / "train")
    for p in pop:
        link(p, tr / f"pop_{p.name}")
    n_mae = int(len(pop) * a.mae_frac / (1 - a.mae_frac))
    for tag, root in (("maer", a.mae_real), ("maes", a.mae_render)):
        for p in rng.sample(clips(pathlib.Path(root) / "train"), n_mae // 2):
            link(p, tr / f"{tag}_{p.name}")
    for tag, root in (("pop", pathlib.Path(a.pop) / "validation"), ("mae", pathlib.Path(a.mae_real) / "validation")):
        for p in rng.sample(clips(root), min(a.n_val, len(clips(root)))):
            link(p, va / f"{tag}_{p.name}")
    print(f"train: {len(pop)} POP909 + {2 * (n_mae // 2)} MAESTRO ({a.mae_frac:.0%}); "
          f"validation: {len(clips(va))} ({a.n_val} POP909 + {a.n_val} real MAESTRO)")


if __name__ == "__main__":
    main()
