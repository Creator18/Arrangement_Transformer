"""Does the vocal-stem / mix SIMILARITY (eval/m2m_stem_similarity.py) find where the base loses the voice?

On the 106 held-out Music2MIDI test songs, for every sung note (the human arrangement's top line where the vocal stem
actually sounds, as in eval/solo_threshold.py): the similarity at that moment (median over +-100 ms of the note
start), the base's recall there (pitch class within 150 ms), and how many of Demucs' other stems (drums, bass,
other) play then (a stem plays when it is within 20 dB of its own loud level, 98th percentile).  Per similarity band:
share of sung notes, base recall, cumulative recall from the top band down (what a "sim >= x" bar would cover), and
the mean number of other stems playing.  Also the share of voiced time per band.

    python eval/solo_sim_stats.py --ckpt <base best.pt> --out <dir> [--shard k --nshards n]  |  --out <dir> --merge
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from eval.rhythm_groups import top_voice  # noqa: E402

SR = 24000
EDGES = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.01]
NAMES = [f"< {EDGES[0]:.2f}"] + [f"{a:.2f}-{min(b, 1):.2f}" for a, b in zip(EDGES[:-1], EDGES[1:])]
KEYS = ("sim", "sim_log")


def bin_of(v):
    return int(np.searchsorted(EDGES, v, side="right"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2m", default="/scratch/basu.ang/m2m")
    ap.add_argument("--audio", default="/scratch/basu.ang/solo_det/m2m")
    ap.add_argument("--sim", default="/scratch/basu.ang/solo_det/m2m_sim")
    ap.add_argument("--ckpt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not a.merge:
        from app.services import spectral_transcription as ST
        m2m = pathlib.Path(a.m2m)
        U = json.loads((m2m / "m2m_units.json").read_text())
        old = set(json.loads((m2m / "m2m_splits.json").read_text())["train"])
        test = [i for i in U["splits"]["test"] if i not in old][a.shard::a.nshards]
        (out / "notes").mkdir(exist_ok=True)
        rows = []
        for sid in test:
            zf = pathlib.Path(a.sim) / f"{sid}.npz"
            if not zf.exists():
                continue
            z = np.load(zf)
            hop = float(z["hop"])
            vdb = z["vocals_db"].astype(float)
            vmax = np.percentile(vdb, 99.5)
            voiced = (vdb > vmax - 35) & (vdb > -60)
            loud = {k: np.percentile(z[f"{k}_db"], 98) for k in ("drums", "bass", "other")}
            plays = sum((z[f"{k}_db"] > loud[k] - 20).astype(int) for k in loud)
            nf = out / "notes" / f"{sid}.json"                       # the base's notes, kept for later runs
            if nf.exists():
                notes = json.loads(nf.read_text())
            else:
                y = np.load(pathlib.Path(a.audio) / f"{sid}.npy").astype(np.float32)
                wav = out / f"{sid}.wav"
                sf.write(wav, y, SR)
                notes = ST.transcribe(wav, onset_thr=0.9, ckpt=pathlib.Path(a.ckpt), frame_thr=0.5)
                wav.unlink()
                nf.write_text(json.dumps([[round(g["onset"], 3), round(g["offset"], 3), int(g["pitch"])] for g in notes]))
                notes = json.loads(nf.read_text())
            hum = np.load(m2m / "midi_numpy" / f"{sid}.npy")
            hum = hum[np.lexsort((-hum[:, 2], hum[:, 0]))]
            mel = hum[top_voice(hum)]
            on = np.array([g[0] for g in notes])
            pc = np.array([g[2] % 12 for g in notes])
            res = []
            for t, _, p in mel[:, :3]:
                i0, i1 = max(0, int((t - 0.1) / hop)), int((t + 0.1) / hop) + 1
                if i0 >= len(vdb) or not voiced[i0:i1].any():
                    continue
                hit = bool(np.any((np.abs(on - t) <= 0.15) & (pc == int(p) % 12))) if len(on) else False
                res.append([bin_of(np.median(z["sim"][i0:i1])), bin_of(np.median(z["sim_log"][i0:i1])), int(hit),
                            float(np.median(plays[i0:i1]))])
            time_bins = {k: np.bincount([bin_of(v) for v in z[k][voiced]], minlength=len(NAMES)).tolist() for k in KEYS}
            rows.append(dict(song=sid, notes=res, time=time_bins))
            print(f"{sid}: {len(res)} sung notes, recall {np.mean([r[2] for r in res]) if res else 0:.3f}", flush=True)
        (out / f"rows_{a.shard}.json").write_text(json.dumps(rows))
        if a.nshards > 1:
            return
    rows = [r for f in sorted(out.glob("rows_*.json")) for r in json.loads(f.read_text())]
    allb = np.array([n for r in rows for n in r["notes"]])
    summ = {}
    print(f"\n{len(rows)} songs, {len(allb)} sung notes where the vocal stem sounds; base recall overall "
          f"{allb[:, 2].mean():.3f}")
    for j, k in enumerate(KEYS):
        tb = np.sum([r["time"][k] for r in rows], 0)
        print(f"\n{k}: similarity of vocal stem and mix")
        print(f"{'band':<12}{'notes':>7}{'share':>8}{'base recall':>13}{'recall at >= band':>19}{'other stems':>13}"
              f"{'voiced time':>13}{'time >= band':>14}")
        summ[k] = []
        for b in range(len(NAMES)):
            sel = allb[allb[:, j] == b]
            up = allb[allb[:, j] >= b]
            rec = sel[:, 2].mean() if len(sel) else float("nan")
            st = sel[:, 3].mean() if len(sel) else float("nan")
            summ[k].append(dict(band=NAMES[b], notes=len(sel), recall=rec, recall_up=up[:, 2].mean() if len(up) else None,
                                stems=st, time=float(tb[b] / tb.sum()), time_up=float(tb[b:].sum() / tb.sum())))
            print(f"{NAMES[b]:<12}{len(sel):>7d}{len(sel) / len(allb):>8.1%}{rec:>13.3f}"
                  f"{(up[:, 2].mean() if len(up) else float('nan')):>19.3f}{st:>13.2f}{tb[b] / tb.sum():>13.1%}"
                  f"{tb[b:].sum() / tb.sum():>14.1%}")
    print("\nother stems = median number of drums / bass / other playing (within 20 dB of their own loud level)")
    (out / "solo_sim_stats.json").write_text(json.dumps(summ, indent=1, default=float))


if __name__ == "__main__":
    main()
