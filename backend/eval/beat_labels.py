"""Training data for OUR beat / downbeat / tempo model (app/ml/spectral/beat_train.py; the user, 2026-10-27/28: replace
madmom, which was a stopgap).  (2026-10-28)

POP909 render clips only (base_render train / validation): their beats and downbeats are exact (POP909 beat_midi.txt,
the same grid the renders' drums were placed on).  Music2MIDI renders are left out: their beats were only estimated
from the notes (render_arrangements.estimate_beats).  The renderer may have sped a song up and doesn't store the
factor, so it is recovered once per song render (song + render tag) the way eval/beat_check.py does it: the clip's
written starts are matched against the song's own starts / speed.

Per clip: a 64-band log-mel at 100 fps (2000 frames, float16) and the clip's beat / downbeat times.  One shard writes
    <out>/<split>_<shard>.mel.npy   (N, 2000, 64) float16
    <out>/<split>_<shard>.json      clip names, beats, downbeats, speed, match

    python eval/beat_labels.py --pop909 /scratch/basu.ang/datasets/pop909_tmp/POP909 \
        --clips /scratch/basu.ang/base_render --split train --out /scratch/basu.ang/beat_data --shard 0 --nshards 40
"""
import argparse
import json
import pathlib
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import beat_check as BC  # noqa: E402
import render_arrangements as RA  # noqa: E402

SR, SEC, FPS, NMEL = 24000, 20.0, 100, 64
NFR = int(SEC * FPS)


def logmel(y):
    import librosa
    m = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=2048, hop_length=SR // FPS, n_mels=NMEL, fmin=30, fmax=11000,
                                       power=1.0, center=True)
    m = np.log1p(100.0 * m)[:, :NFR]
    return np.pad(m, ((0, 0), (0, NFR - m.shape[1]))).T.astype(np.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop909", required=True)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in (pathlib.Path(a.clips) / a.split).glob("pop*.npz") if ".tmp" not in p.name)
    songs = sorted({f.stem.split("_")[0] for f in files})[a.shard::a.nshards]
    keep = set(songs)
    groups = defaultdict(list)                                  # (song, render tag) -> clips in window order
    for f in files:
        s = f.stem.split("_")
        if s[0] in keep:
            groups[(s[0], s[1])].append(f)
    n_all = sum(len(v) for v in groups.values())
    mel = np.lib.format.open_memmap(out / f"{a.split}_{a.shard}.mel.npy", mode="w+", dtype=np.float16,
                                    shape=(n_all, NFR, NMEL))
    rows, i, skipped = [], 0, 0
    for (sid, tag), fl in sorted(groups.items()):
        notes, _, beats, flags = RA.load_pop909(pathlib.Path(a.pop909) / sid[3:])[:4]
        song_on, beats = np.sort(notes[:, 0]), np.asarray(beats, float)
        downs = np.asarray(RA.downbeats(beats, flags), float)
        fl = sorted(fl, key=lambda p: int(p.stem.split("_")[-1]))
        best = None                                             # the song render's speed: best match over a few clips
        for f in fl[:4]:
            w = int(f.stem.split("_")[-1])
            r = BC.clip_beats(np.load(f), song_on, beats, w * SEC)
            if r is not None and (best is None or r[2] > best[1]):
                best = (r[1], r[2])
        if best is None:
            skipped += len(fl)
            print(f"{sid} {tag}: no speed match, {len(fl)} clips skipped", flush=True)
            continue
        sp, m = best
        for f in fl:
            w = int(f.stem.split("_")[-1])
            b = beats / sp - w * SEC
            d = downs / sp - w * SEC
            b, d = b[(b >= 0) & (b < SEC)], d[(d >= 0) & (d < SEC)]
            if len(b) < 4:
                skipped += 1
                continue
            y = np.load(f)["y"].astype(np.float32) / 32767
            mel[i] = logmel(y)
            rows.append(dict(clip=f.stem, beats=b.round(4).tolist(), downbeats=d.round(4).tolist(),
                             speed=round(float(sp), 4), match=round(float(m), 3)))
            i += 1
        print(f"{sid} {tag}: speed {sp:.3f} (match {m:.2f}), {len(fl)} clips", flush=True)
    mel.flush()
    (out / f"{a.split}_{a.shard}.json").write_text(json.dumps(dict(n=i, rows=rows)))
    print(f"done: {i} clips written ({skipped} skipped); mel rows {i} of {n_all} used", flush=True)


if __name__ == "__main__":
    main()
