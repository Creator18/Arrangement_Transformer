"""CPU producer for the random-note curriculum: random clips made on the fly, never pre-rendered (the user, 2026-11-02).

Many producers run as a CPU job array on `courses`, started one after another, and feed the GPU trainer through a
shared spool folder on scratch:
    control.json   {"level": k, "stop": false}: written by the trainer, re-read before every bundle
    spool/*.npz    bundles of clips, written under a temporary name and renamed (the trainer never sees half a file);
                   the trainer loads and DELETES them, so every clip is used once
    back-pressure  a producer pauses while the spool already holds --max-files bundles
Each bundle records exactly what went in (notes, tags, every sound quality) next to the audio and the targets.
A share of clips re-voice a recent clip's notes with fresh qualities (same notes, different sound); a share
reviews lower levels so earlier skills are kept.

    python eval/random_producer.py --spool DIR --control FILE --sf2s a,b,c --sf-table T --seed K
    python eval/random_producer.py --evalset DIR --per-level 48 ...       (the fixed, seeded test clips per level)
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "eval"))
import random_notes as RN                 # noqa: E402


def targets(notes, tags, seconds):
    """onset / frame / velocity rolls (written times) and a TEACH mask: 0 over inaudible notes (neither taught nor
    scored)."""
    on, fr, vel = RN.rolls(notes, seconds)
    teach = np.ones_like(on)
    from app.ml.spectral import frontend as S
    for (s, e, p, v, k), t in zip(notes, tags):
        if t[RN.TAGS.index("inaudible")]:
            a, b = int(round(s * S.FPS)), int(round(e * S.FPS))
            teach[int(p) - 21, max(0, a - 1):min(on.shape[1], max(b, a + 1) + 1)] = 0
    return on, fr, vel, teach


def bundle(clips, seconds):
    ys, ons, frs, vels, teach, recs = [], [], [], [], [], []
    for lv, y, n, T, meta, revoiced in clips:
        on, fr, vel, te = targets(n, T, seconds)
        L = int(seconds * RN.SR)
        yy = np.zeros(L, np.float32); yy[:min(L, len(y))] = y[:L]
        ys.append((np.clip(yy, -1, 1) * 32767).astype(np.int16)); ons.append(on); frs.append(fr); vels.append(vel)
        teach.append(te)
        recs.append(dict(meta, level=lv, revoiced=revoiced, notes=n.tolist(), tags=T.astype(int).tolist()))
    return dict(y=np.stack(ys), on=np.stack(ons), fr=np.stack(frs), vel=np.stack(vels), teach=np.stack(teach),
                level=np.array([r["level"] for r in recs]), record=json.dumps(recs, default=float))


def write(path, d):
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **d)
    tmp.replace(path)


def produce(a):
    spool = pathlib.Path(a.spool); spool.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1000 + a.seed)
    time.sleep(a.seed * a.stagger)                            # started one after another
    recent, n, t_start = {}, 0, time.time()
    print(f"producer {a.seed} up", flush=True)
    while True:
        try:
            ctl = json.loads(pathlib.Path(a.control).read_text())
        except Exception:                                     # noqa: BLE001  (being rewritten: try again)
            time.sleep(1); continue
        if ctl.get("stop"):
            if n == 0 and time.time() - t_start < 1800:          # a new run: the trainer has not cleared the flag yet
                time.sleep(5); continue
            print(f"producer {a.seed}: stop flag, {n} bundles made", flush=True); return
        if len(list(spool.glob("*.npz"))) >= a.max_files:
            time.sleep(2); continue
        level, clips = int(ctl["level"]), []
        for _ in range(a.bundle):
            lv = int(rng.integers(1, level)) if level > 1 and rng.random() < a.review_p else level
            pool = recent.get(lv, [])
            revoice = bool(pool) and rng.random() < a.revoice_p
            notes = pool[int(rng.integers(len(pool)))] if revoice else None
            y, nt, T, meta = RN.make_clip(lv, rng, a.seconds, notes=notes)
            if not revoice:
                recent.setdefault(lv, []).append(nt); recent[lv] = recent[lv][-20:]
            clips.append((lv, y, nt, T, meta, revoice))
        write(spool / f"p{a.seed:03d}_{n:06d}.npz", bundle(clips, a.seconds))
        n += 1
        if n % 20 == 0:
            print(f"producer {a.seed}: {n} bundles, level {level}, {(time.time() - t_start) / n:.1f} s per bundle", flush=True)
        if a.max_minutes and (time.time() - t_start) / 60 > a.max_minutes:
            print(f"producer {a.seed}: time budget, {n} bundles", flush=True); return


def evalset(a):
    """fixed, seeded test clips per level: the same clips at every check, so progress is comparable."""
    out = pathlib.Path(a.evalset); out.mkdir(parents=True, exist_ok=True)
    for lv in RN.LEVELS:
        f = out / f"level{lv}.npz"
        if f.exists():
            continue
        rng = np.random.default_rng(777 + lv)
        clips = []
        for i in range(a.per_level):
            y, nt, T, meta = RN.make_clip(lv, rng, a.seconds)
            clips.append((lv, y, nt, T, meta, False))
        write(f, bundle(clips, a.seconds))
        print(f"eval level {lv}: {a.per_level} clips", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sf2s", required=True); ap.add_argument("--sf-table", required=True)
    ap.add_argument("--spool"); ap.add_argument("--control"); ap.add_argument("--evalset")
    ap.add_argument("--seed", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--bundle", type=int, default=16); ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--max-files", type=int, default=300); ap.add_argument("--stagger", type=float, default=3.0)
    ap.add_argument("--review-p", type=float, default=0.2); ap.add_argument("--revoice-p", type=float, default=0.2)
    ap.add_argument("--per-level", type=int, default=48); ap.add_argument("--max-minutes", type=float, default=0)
    ap.add_argument("--presets", default="", help="eval/random_calib.py output: presets that sound off the key")
    a = ap.parse_args()
    RN.init(a.sf2s.split(","), a.sf_table, seed=a.seed, presets=a.presets)
    evalset(a) if a.evalset else produce(a)


if __name__ == "__main__":
    main()
