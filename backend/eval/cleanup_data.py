"""Data for the CLEAN-UP model (the user's design, 2026-10-27): the base's messy 0.8 output in, the clean written
arrangement as the answer.  First run on the base_render VALIDATION clips (34,262 clips, 228 songs): the base never
trained on them, so its mistakes there are the kind it makes on new songs.  v2 adds 40 clips of each of the 1,522
TRAINING songs (--list): the base has heard those, so its mistakes there are fewer, but the clean-up needs songs.

Per clip, one npz in --out:
    notes   (N, 7) float32  the base's notes at the 0.8 bar, the app's decoding (spectral_transcription.transcribe):
                            onset s, offset s, pitch, velocity, peak onset probability (+-2 frames), label, clip time 0
            label: 1 = a written start on that key within 48 ms (one-to-one, nearest first), 0 = a mistake
    on, fr  (88, T) float16 the base's onset / frame probabilities (context for the model; decode at any bar later)
    gt      (M, 3) float32  the written notes: onset s, offset s, pitch (from the clip's label rolls)

    python eval/cleanup_data.py --ckpt <base best.pt> --clips /scratch/basu.ang/base_render/validation \
        --out /scratch/basu.ang/cleanup_data --shard k --nshards n
"""
import argparse
import pathlib
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from app.ml.spectral import frontend as S  # noqa: E402
from app.services import spectral_transcription as ST  # noqa: E402

BAR = 0.8
TOL = 3                                                    # frames (48 ms)


def written(on, fr):
    """(onset s, offset s, pitch) of every written note: a start in the onset roll, held while the frame roll is."""
    out = []
    T = on.shape[1]
    for k, f in zip(*np.nonzero(on)):
        e = f + 1
        while e < T and fr[k, e] and not on[k, e]:
            e += 1
        out.append((f / S.FPS, e / S.FPS, k + 21))
    return np.array(sorted(out), np.float32).reshape(-1, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--list", default="", help="only these clip file names (one per line), e.g. a sample of the TRAINING renders")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.list:
        files = sorted(pathlib.Path(a.clips) / n.strip() for n in open(a.list) if n.strip())[a.shard::a.nshards]
    else:
        files = sorted(p for p in pathlib.Path(a.clips).glob("*.npz") if ".tmp" not in p.name)[a.shard::a.nshards]
    tmp = out / f"_tmp{a.shard}.wav"
    real = ST.rolls
    cache = {}
    ST.rolls = lambda path, cond=None, with_offset=False: cache.setdefault("r", real(path, cond=cond, with_offset=with_offset))
    for i, f in enumerate(files):
        dst = out / f.name
        if dst.exists():
            continue
        z = np.load(f)
        sf.write(tmp, z["y"].astype(np.float32) / 32767, S.SR)
        cache.clear()
        notes = ST.transcribe(tmp, onset_thr=BAR, ckpt=pathlib.Path(a.ckpt), frame_thr=0.5)
        on, fr = cache["r"][0], cache["r"][1]
        won, wfr = z["on"] > 0, z["fr"] > 0
        T = min(won.shape[1], on.shape[1])
        gt = written(won[:, :T], wfr[:, :T])
        starts = {}
        for j, (o, _, p) in enumerate(gt):
            starts.setdefault(int(p), []).append((int(round(o * S.FPS)), j))
        cand = []                                          # (distance, note index, written index)
        for i_n, g in enumerate(notes):
            fq = int(round(g["onset"] * S.FPS))
            cand += [(abs(fw - fq), i_n, j) for fw, j in starts.get(g["pitch"], []) if abs(fw - fq) <= TOL]
        label = np.zeros(len(notes), np.float32)
        used = set()
        for _, i_n, j in sorted(cand):
            if not label[i_n] and j not in used:
                label[i_n] = 1
                used.add(j)
        rows = []
        for g, l in zip(notes, label):
            k, fq = g["pitch"] - 21, int(round(g["onset"] * S.FPS))
            rows.append([g["onset"], g["offset"], g["pitch"], g["velocity"], float(on[k, max(0, fq - 2):fq + 3].max()), l, 0])
        tmpz = out / (f.stem + ".tmp.npz")
        np.savez_compressed(tmpz, notes=np.array(rows, np.float32).reshape(-1, 7), on=on[:, :T].astype(np.float16),
                            fr=fr[:, :T].astype(np.float16), gt=gt)
        tmpz.rename(dst)
        if i % 200 == 0:
            print(f"{i + 1}/{len(files)} {f.name}: {len(notes)} notes, {int(label.sum())} right, {len(gt)} written", flush=True)
    tmp.unlink(missing_ok=True)
    print("done")


if __name__ == "__main__":
    main()
