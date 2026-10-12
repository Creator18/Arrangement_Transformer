"""Music2MIDI songs -> spectral training clips: song audio in, human piano cover out.

Song-to-piano v1.  The spectral transcriber hears piano at note F1 0.955 on
real recordings; fine-tuning it so the ANSWER becomes "the notes a human put in
their piano cover of this song" keeps the hearing and spends the small dataset
on what to keep.  The same move as the September arranger (old mel transcriber
-> Music2MIDI) and AMT-APC (arXiv 2409.14086), on a far better starting model.

Output matches render_multi_full.py / fetch_real_clips.py: 20 s windows,
<split>/<id>_<w>.npz (int16 audio @ 24 kHz, onset + frame rolls at 62.5 fps,
piano="m2m") and <id>_<w>.vel.npy only when the cover has dynamics (a third of
MuseScore exports carry one default velocity: missing data, not quiet playing;
no file -> the clip's velocity loss is masked).  data.RenderedClips reads it
unchanged.

Splits are arranger_data.usable_splits(): alignment-verified songs, carved and
pinned by song (853 / 106 / 106) — so v1 numbers compare with the September
arranger's on identical held-out songs.

The audio was fetched at 16 kHz, so it carries nothing above 8 kHz; the model
was trained at 24 kHz.  Resampled up here; fine-tuning absorbs the missing top.

    python eval/make_m2m_clips.py --out /scratch/.../m2m_clips [--audio-dir ...]
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SECONDS = 20.0
DIFF = {"beginner": 0, "intermediate": 1, "advanced": 2}


def rolls(notes: np.ndarray, T: int, fps: float):
    on = np.zeros((88, T), np.uint8); fr = np.zeros((88, T), np.uint8)
    vel = np.zeros((88, T), np.uint8)
    for s, e, p, v in notes:
        k = int(p) - 21
        if not 0 <= k < 88:
            continue
        a, b = int(round(s * fps)), int(round(e * fps))      # centred frames: round
        if a >= T or b <= 0:
            continue
        fr[k, max(0, a):min(T, max(b, a + 1))] = 1
        if a >= 0:
            on[k, a] = 1
            vel[k, a] = max(vel[k, a], int(np.clip(v, 1, 127)))
    return on, fr, vel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--audio-dir", default="")
    ap.add_argument("--data-dir", default="")
    ap.add_argument("--splits", default="", help="json {train,validation,test,difficulty}; "
                    "written by --write-splits so the cluster needs no metadata")
    ap.add_argument("--write-splits", default="")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0, help="smoke runs only")
    args = ap.parse_args()
    import soundfile as sf
    from scipy.signal import resample_poly
    from app.ml.spectral import frontend as S
    from app.ml.transcriber import arranger_data as AD
    if args.audio_dir:
        AD.AUDIO = pathlib.Path(args.audio_dir)
    if args.data_dir:
        AD.DATA = pathlib.Path(args.data_dir)

    import json
    if args.splits:
        d = json.loads(pathlib.Path(args.splits).read_text())
        tr, va, te, how, diff = d["train"], d["validation"], d["test"], d["how"], d["difficulty"]
    else:
        import yaml
        tr, va, te, how = AD.usable_splits(AD.AUDIO)
        diff = {s: yaml.safe_load(open(AD.DATA / "metadata" / f"{s}.yaml", encoding="utf-8"))
                ["piano"]["difficulty"] for s in tr + va + te}
    if args.write_splits:
        pathlib.Path(args.write_splits).write_text(json.dumps(
            {"train": tr, "validation": va, "test": te, "how": how, "difficulty": diff}))
        print(f"wrote {args.write_splits}"); return
    print(f"split: {len(tr)} / {len(va)} / {len(te)}  ({how})", flush=True)
    seg = int(SECONDS * S.SR)
    T = S.n_frames(seg)
    for split, ids in (("train", tr), ("validation", va), ("test", te)):
        out = pathlib.Path(args.out) / split
        out.mkdir(parents=True, exist_ok=True)
        n = 0
        si, sn = (int(v) for v in args.shard.split("/"))
        for sid in ids[si::sn][: args.limit or None]:
            y, sr = sf.read(str(AD.AUDIO / f"{sid}.wav"), dtype="float32", always_2d=True)
            y = y.mean(1)
            g = math.gcd(S.SR, sr)
            y = resample_poly(y, S.SR // g, sr // g).astype(np.float32)
            notes = np.load(AD.DATA / "midi_numpy" / f"{sid}.npy")    # [on, off, pitch, vel]
            has_dyn = len(np.unique(notes[:, 3])) >= AD.MIN_VELOCITIES
            for w in range(max(1, math.ceil(len(y) / seg))):
                dst = out / f"{sid}_{w:03d}.npz"
                if dst.exists():
                    continue
                clip = np.pad(y[w * seg:(w + 1) * seg], (0, max(0, seg - len(y[w * seg:(w + 1) * seg]))))
                sel = notes[(notes[:, 1] > w * SECONDS) & (notes[:, 0] < (w + 1) * SECONDS)].copy()
                sel[:, :2] -= w * SECONDS
                on, fr, vel = rolls(sel, T, S.FPS)
                tmp = dst.with_name(dst.stem + ".tmp.npz")
                np.savez_compressed(tmp, y=(np.clip(clip, -1, 1) * 32767).astype(np.int16),
                                    on=on, fr=fr, piano="m2m",
                                    difficulty=DIFF[diff[sid]])
                tmp.replace(dst)
                if has_dyn:
                    vt = dst.with_name(dst.stem + ".vel.tmp.npy")
                    np.save(vt, vel); vt.replace(dst.with_name(dst.stem + ".vel.npy"))
                n += 1
        print(f"  {split}: {n} windows written", flush=True)


if __name__ == "__main__":
    main()
