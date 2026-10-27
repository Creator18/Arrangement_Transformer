"""How cleanly do the arrangements' written notes sit on their beat grid?  (Before a learned quantizer: 2026-10-27.)

POP909: the dataset's own beats (beat_midi.txt).  Music2MIDI: no beat labels, so the beats the renderer uses
(render_arrangements.estimate_beats: librosa's tracker on the arrangement's own onsets).  For every note start
inside the beat grid: its position in the beat (0-1), the share within 2% / 0.5% of a beat of a sixteenth /
sixteenth+triplet / thirty-second grid, and the tempo the beats imply.  Per song too, so a song whose grid fails
(tempo octave, drift) shows up as a low on-grid share.

    python eval/grid_check.py --pop909 <dir> --m2m /scratch/basu.ang/m2m --n 200 --out <json>
"""
import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import render_arrangements as RA  # noqa: E402

GRIDS = {"16ths": np.arange(0, 1.01, .25),
         "16ths+triplets": np.unique(np.r_[np.arange(0, 1.01, .25), np.arange(0, 1.01, 1 / 3)]),
         "32nds": np.arange(0, 1.01, .125)}


def positions(on, beats):
    on = on[(on >= beats[0]) & (on < beats[-1])]
    k = np.searchsorted(beats, on, side="right") - 1
    return (on - beats[k]) / (beats[k + 1] - beats[k])


def summary(fr):
    out = {"notes": int(len(fr))}
    for g, grid in GRIDS.items():
        d = np.min(np.abs(fr[:, None] - grid[None]), 1)
        out[g] = {"within 2%": round(float((d < .02).mean()), 3), "exact": round(float((d < .005).mean()), 3)}
    h, _ = np.histogram(fr, bins=24, range=(0, 1))
    out["hist24"] = (h / max(1, len(fr))).round(3).tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop909", required=True)
    ap.add_argument("--m2m", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {}
    pops = sorted(p for p in pathlib.Path(a.pop909).iterdir() if p.is_dir() and (p / "beat_midi.txt").exists())[:a.n]
    m2m = pathlib.Path(a.m2m)
    ids = json.loads((m2m / "m2m_units.json").read_text())["splits"]["train"][:a.n]
    for name, items, load in (("pop909", pops, lambda d: RA.load_pop909(d)),
                              ("m2m", ids, lambda i: RA.load_m2m(m2m / "midi_numpy" / f"{i}.npy"))):
        allf, per = [], []
        for it in items:
            try:
                notes, _, beats, _ = load(it)[:4]
            except Exception as e:                                       # noqa: BLE001
                print(f"{name} {it}: {type(e).__name__} {e}")
                continue
            if len(beats) < 8:
                continue
            fr = positions(notes[:, 0], np.asarray(beats, float))
            d = np.min(np.abs(fr[:, None] - GRIDS["16ths+triplets"][None]), 1)
            per.append({"song": str(getattr(it, "name", it)), "bpm": round(float(60 / np.median(np.diff(beats))), 1),
                        "on_grid": round(float((d < .02).mean()), 3)})
            allf.append(fr)
        allf = np.concatenate(allf)
        og = np.array([p["on_grid"] for p in per])
        res[name] = dict(songs=len(per), **summary(allf),
                         song_on_grid_pct=dict(zip(("10%", "25%", "50%", "75%", "90%"), np.percentile(og, [10, 25, 50, 75, 90]).round(3).tolist())),
                         songs_below_0_8=int((og < .8).sum()), bpm_pct=np.percentile([p["bpm"] for p in per], [10, 50, 90]).round(1).tolist(),
                         worst=sorted(per, key=lambda p: p["on_grid"])[:8])
        print(f"\n== {name}: {len(per)} songs, {len(allf)} notes")
        print(json.dumps({k: v for k, v in res[name].items() if k != "hist24"}, indent=1))
        print("position in beat (24 bins): " + " ".join(f"{v:.2f}" for v in res[name]["hist24"]))
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
