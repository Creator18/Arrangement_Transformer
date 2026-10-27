"""Can we read the beat from AUDIO well enough to quantize the arranger's notes?  (2026-10-27)

On held-out base renders of POP909 songs (base_render validation, true beats from beat_midi.txt).  The clips don't
store their beats, and the renderer may have sped a song up (--tempo), so each clip's grid is recovered: the clip's
written starts (its onset label roll) are matched against the song's own note starts / speed (speed searched
0.5-2.5x), and the song's beats are mapped the same way.  Clips whose match is weak are skipped.

Trackers: madmom (RNN + DBN beats; RNN + DBN downbeats).  (librosa's tracker segfaults in numba on some courses
nodes, so it is left out.)  Scored with mir_eval: F-measure
(+-70 ms), and the share of clips at the right tempo / double / half tempo.  Real songs have no beat labels; their
tempo agreement between madmom and the human arrangement's own estimated beats is a sanity check (--real).

    python eval/beat_check.py --pop909 <dir> --clips /scratch/basu.ang/base_render/validation --n 300 --out <json>
"""
import argparse
import json
import pathlib
import random
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import render_arrangements as RA  # noqa: E402

FPS, SR, SEC = 62.5, 24000, 20.0


def clip_beats(z, song_on, beats, t0):
    """(beats inside the clip, speed, match share) or None: fit the speed s so song starts / s land on the clip's."""
    won = z["on"] > 0
    t = np.sort(np.nonzero(won)[1] / FPS) + t0
    if len(t) < 8:
        return None
    def score(s):
        src = song_on / s
        src = src[(src >= t0 - .1) & (src < t0 + SEC + .1)]
        if not len(src):
            return 0.0
        i = np.clip(np.searchsorted(src, t), 1, len(src) - 1)
        return float((np.minimum(np.abs(src[i] - t), np.abs(src[i - 1] - t)) < .02).mean())
    best = max((score(s), s) for s in np.arange(0.5, 2.5, 0.005))             # coarse, then fine around the best
    best = max([best] + [(score(s), s) for s in np.arange(best[1] - .006, best[1] + .006, 0.0002)])
    m, s = best
    if m < .8:
        return None
    b = beats / s - t0
    return b[(b >= 0) & (b < SEC)], s, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop909", required=True)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    import librosa
    import mir_eval
    from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
    from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
    rnn, dbn = RNNBeatProcessor(), DBNBeatTrackingProcessor(fps=100)
    drnn, ddbn = RNNDownBeatProcessor(), DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
    files = sorted(p for p in pathlib.Path(a.clips).glob("pop*.npz") if ".tmp" not in p.name)
    files = random.Random(0).sample(files, min(4 * a.n, len(files)))[a.shard::a.nshards]
    a.n //= a.nshards
    songs, rows = {}, []
    for f in files:
        if len(rows) >= a.n:
            break
        sid, w = f.stem.split("_")[0][3:], int(f.stem.split("_")[-1])
        if sid not in songs:
            notes, _, beats, flags = RA.load_pop909(pathlib.Path(a.pop909) / sid)[:4]
            songs[sid] = (np.sort(notes[:, 0]), np.asarray(beats, float), flags)
        song_on, beats, flags = songs[sid]
        z = np.load(f)
        r = clip_beats(z, song_on, beats, w * SEC)
        if r is None or len(r[0]) < 8:
            continue
        ref, speed, match = r
        y = z["y"].astype(np.float32) / 32767
        y44 = librosa.resample(y, orig_sr=SR, target_sr=44100)
        est = {"madmom": dbn(rnn(y44))}
        db = ddbn(drnn(y44))
        est["madmom downbeat model"] = db[:, 0] if len(db) else np.zeros(0)
        row = dict(clip=f.stem, speed=round(speed, 4), match=round(match, 3), bpm=round(float(60 / np.median(np.diff(ref))), 1))
        for k, e in est.items():
            e = np.asarray(e, float)
            row[k + " beats"] = e.round(3).tolist()
            f1 = mir_eval.beat.f_measure(mir_eval.beat.trim_beats(ref, 0), mir_eval.beat.trim_beats(e, 0)) if len(e) > 1 else 0.0
            ratio = float(np.median(np.diff(ref)) / np.median(np.diff(e)))           # 2 = tracker at double tempo if len(e) > 2 else 0.0
            # for QUANTIZING, a grid twice as fine is fine: does the tracker's grid contain the true beats, and do
            # the true beats + half beats contain the tracker's?  (+-70 ms)
            hit = lambda a_, b_: float(np.mean([np.min(np.abs(b_ - x)) < .07 for x in a_])) if len(a_) and len(b_) else 0.0
            half = np.sort(np.r_[ref, (ref[1:] + ref[:-1]) / 2])
            row[k] = dict(f1=round(float(f1), 3), tempo_ratio=round(ratio, 3), true_beats_on_grid=round(hit(ref, e), 3),
                          grid_on_true_halves=round(hit(e, half), 3))
        rows.append(row)
        print(json.dumps(row), flush=True)
    out = {"clips": len(rows)}
    for k in ("madmom", "madmom downbeat model"):
        f1 = np.array([r[k]["f1"] for r in rows]); ra = np.array([r[k]["tempo_ratio"] for r in rows])
        near = lambda x: np.abs(ra / x - 1) < .04
        tb = np.array([r[k]["true_beats_on_grid"] for r in rows]); gh = np.array([r[k]["grid_on_true_halves"] for r in rows])
        out[k] = dict(true_beats_on_grid=round(float(tb.mean()), 3), grid_on_true_halves=round(float(gh.mean()), 3),
                      usable_grid=round(float(((tb > .9) & (gh > .9)).mean()), 3), f1_mean=round(float(f1.mean()), 3), f1_median=round(float(np.median(f1)), 3),
                      f1_above_0_9=round(float((f1 > .9).mean()), 3), right_tempo=round(float(near(1).mean()), 3),
                      double_tempo=round(float(near(2).mean()), 3), half_tempo=round(float(near(.5).mean()), 3),
                      other_tempo=round(float(1 - near(1).mean() - near(.5).mean() - near(2).mean()), 3))
    print(json.dumps(out, indent=1))
    pathlib.Path(a.out).write_text(json.dumps(dict(summary=out, rows=rows), indent=1))


if __name__ == "__main__":
    main()
