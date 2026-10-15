"""Fine-tuning set: a NEW render batch + a share of the original renders, so adapters learn the new
material without the model forgetting the old (symlinks only).

    python eval/prep_ft_mix.py --new /scratch/.../arr_tone,/scratch/.../arr_vocal --old /scratch/.../arr_render --out <dir> [--old-frac 0.33]
<out>/train: every new train clip + old train clips equal to --old-frac of the new count
<out>/validation: --n-val new + --n-val old validation clips (prefixes new_ / old_)
"""
import argparse, os, pathlib, random
ap = argparse.ArgumentParser(); ap.add_argument("--new", required=True); ap.add_argument("--old", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--old-frac", type=float, default=0.33); ap.add_argument("--n-val", type=int, default=150)
a = ap.parse_args(); rng = random.Random(0)
clips = lambda d: sorted(p for p in pathlib.Path(d).glob("*_[0-9][0-9][0-9].npz") if ".tmp" not in p.name)
def link(p, dst):
    for src, d in ((p, dst), (p.with_name(p.stem + ".vel.npy"), dst.with_name(dst.stem + ".vel.npy"))):
        if src.exists() and not d.exists():
            os.symlink(src, d)
tr = pathlib.Path(a.out) / "train"; va = pathlib.Path(a.out) / "validation"; tr.mkdir(parents=True, exist_ok=True); va.mkdir(parents=True, exist_ok=True)
new = [p for d in a.new.split(",") for p in clips(pathlib.Path(d) / "train")]
for p in new:
    link(p, tr / f"new_{p.parent.parent.name}_{p.name}")
old = clips(pathlib.Path(a.old) / "train")
for p in rng.sample(old, min(len(old), int(len(new) * a.old_frac))):
    link(p, tr / f"old_{p.name}")
for tag, root in [("new", d) for d in a.new.split(",")] + [("old", a.old)]:
    v = clips(pathlib.Path(root) / "validation")
    for p in rng.sample(v, min(a.n_val, len(v))):
        link(p, va / f"{tag}_{pathlib.Path(root).name}_{p.name}")
print(len(list(tr.glob("*.npz"))), "train,", len(list(va.glob("*.npz"))), "validation")
