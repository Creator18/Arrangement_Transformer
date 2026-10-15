"""Validation set for the render-trained model, grouped by render type (symlinks only, CPU).

    <out>/validation/pia_<clip>   pure piano renders
    <out>/validation/l<k>_<clip>  multi-instrument renders at layering level k (0 none .. 3 heavy)
--per clips each, drawn at random from <renders>/validation; eval/hum_s1_checks.py reports each group.
"""
import argparse, json, os, pathlib, random
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--renders", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--per", type=int, default=100); a = ap.parse_args()
paths = sorted(p for p in (pathlib.Path(a.renders) / "validation").glob("*_r*_[0-9][0-9][0-9].npz"))
random.Random(0).shuffle(paths)
want = {"pia": [], "l0": [], "l1": [], "l2": [], "l3": []}
for p in paths:
    if all(len(v) >= a.per for v in want.values()):
        break
    d = json.loads(str(np.load(p)["render"]))
    tag = "pia" if "piano" in d else f"l{d['layering']}"
    if len(want[tag]) < a.per:
        want[tag].append(p)
out = pathlib.Path(a.out) / "validation"; out.mkdir(parents=True, exist_ok=True)
for tag, ps in want.items():
    for p in ps:
        for src, dst in ((p, out / f"{tag}_{p.name}"), (p.with_name(p.stem + ".vel.npy"), out / f"{tag}_{p.stem}.vel.npy")):
            if src.exists() and not dst.exists():
                os.symlink(src, dst)
print({k: len(v) for k, v in want.items()})
