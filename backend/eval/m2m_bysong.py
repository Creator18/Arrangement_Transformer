"""Music2MIDI by SONG: a leak-free split, style units, and human-vs-human agreement (CPU).

The carved split was by arrangement, so 16/106 test and 23/106 validation
arrangements share their song audio with a training arrangement.  Here songs
are grouped by YouTube audio and every group goes to ONE split: a group that
touches test goes to test, else one that touches validation goes to
validation, else train — so held-out sets only grow, and train loses the
leaking arrangements.

A "unit" is one song audio at one difficulty: the arrangements a style-code
model can be asked to match with that song and that level (the difficulty
input differs across levels, so different-level arrangements are not rival
answers to the same question).

Human agreement: for every pair of arrangements of the same audio, note F1
(strict 50 ms / tolerant 150 ms / pitch-class) one against the other — the
realistic ceiling for matching any single human arrangement.

    python eval/m2m_bysong.py --splits m2m_splits.json --out m2m_units.json
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "eval"))
DATA = ROOT.parent / "data" / "music2midi" / "music2midi-dataset"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import yaml
    from s2p_evaluate import f1s
    sp = json.loads(pathlib.Path(a.splits).read_text())
    old = {i: k for k in ("train", "validation", "test") for i in sp[k]}
    meta = {i: yaml.safe_load(open(DATA / "metadata" / f"{i}.yaml", encoding="utf-8")) for i in old}
    groups = collections.defaultdict(list)
    for i, m in meta.items():
        groups[m["youtube"]["url"]].append(i)
    new, units = {"train": [], "validation": [], "test": []}, {"train": [], "validation": [], "test": []}
    for url, ids in sorted(groups.items()):
        ks = {old[i] for i in ids}
        k = "test" if "test" in ks else "validation" if "validation" in ks else "train"
        new[k] += sorted(ids)
        by_lv = collections.defaultdict(list)
        for i in ids:
            by_lv[meta[i]["piano"]["difficulty"]].append(i)
        units[k] += [sorted(v) for _, v in sorted(by_lv.items())]
    for k in new:
        moved = sum(old[i] != k for i in new[k])
        multi = sum(len(u) > 1 for u in units[k])
        print(f"{k:<11} {len(new[k]):>4} arrangements ({moved:>2} moved in)  {len(units[k]):>4} units, "
              f"{multi} with >1 arrangement")

    rows = collections.defaultdict(list)
    for ids in groups.values():
        for x, y in itertools.combinations(sorted(ids), 2):
            nx = np.load(DATA / "midi_numpy" / f"{x}.npy"); ny = np.load(DATA / "midi_numpy" / f"{y}.npy")
            same = meta[x]["piano"]["difficulty"] == meta[y]["piano"]["difficulty"]
            f, p, r, ft, fp = f1s(nx, ny)
            rows["same level" if same else "different level"].append((f, ft, fp))
    print("\nhuman vs human, same song audio (note F1: strict 50 ms / tolerant 150 ms / pitch-class):")
    for k, v in rows.items():
        v = np.array(v)
        print(f"  {k:<16} {len(v):>3} pairs   strict {np.mean(v[:, 0]):.3f} (median {np.median(v[:, 0]):.3f})  "
              f"tolerant {np.mean(v[:, 1]):.3f}  pc {np.mean(v[:, 2]):.3f}")
    pathlib.Path(a.out).write_text(json.dumps(dict(splits=new, units=units,
                                                   difficulty={i: meta[i]["piano"]["difficulty"] for i in old})))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
