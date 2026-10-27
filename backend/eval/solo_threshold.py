"""Where should the song-relative "light band" threshold sit?  (the user, 2026-10-26: "test and see what will be a good
threshold" for the bottom-X% rule of eval/solo_rel_stats.py / solo_train.py --label rel)

The fill only helps where the base loses the voice.  So: run the base over the 106 held-out Music2MIDI test songs and,
for every sung note (the human arrangement's top line, >= C4, as in eval/solo_fill_real.py), place the instrumental's
level at that moment within the song's own range (Demucs curves: 2nd-98th percentile while it sounds; median over
+-100 ms of the note start).  Then the base's recall per band -- pitch class within 150 ms, the measure used for solo
parts -- shows where it starts losing the voice.  A good threshold covers the bands where recall is low and stops
where it reaches the full-band level.

    python eval/solo_threshold.py --ckpt <base best.pt> --out <dir> [--shard k --nshards n]  |  --out <dir> --merge
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from app.services import spectral_transcription as ST  # noqa: E402
from eval.rhythm_groups import top_voice  # noqa: E402

SR = 24000
EDGES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
NAMES = ["silent"] + [f"{int(a * 100)}-{int(b * 100)}%" for a, b in zip([0] + EDGES[:-1], EDGES)] + [">100%"]


def song_range(rdb):
    top = np.percentile(rdb, 99.5)
    sounding = rdb > max(-70.0, top - 50)
    if sounding.sum() > 50:
        return float(np.percentile(rdb[sounding], 2)), float(np.percentile(rdb[sounding], 98))
    return float(top - 50), float(top)


def band(pos, silent):
    if silent:
        return 0
    for i, e in enumerate(EDGES):
        if pos < e:
            return i + 1
    return len(EDGES) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2m", default="/scratch/basu.ang/m2m")
    ap.add_argument("--audio", default="/scratch/basu.ang/solo_det/m2m")
    ap.add_argument("--curves", default="/scratch/basu.ang/solo_det/m2m_curves")
    ap.add_argument("--ckpt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not a.merge:
        m2m = pathlib.Path(a.m2m)
        U = json.loads((m2m / "m2m_units.json").read_text())
        old = set(json.loads((m2m / "m2m_splits.json").read_text())["train"])
        test = [i for i in U["splits"]["test"] if i not in old][a.shard::a.nshards]
        rows = []
        for sid in test:
            cf = pathlib.Path(a.curves) / f"{sid}.npz"
            if not cf.exists():
                continue
            z = np.load(cf)
            hop = float(z["hop"])
            vdb, rdb = z["vocal_db"].astype(float), z["rest_db"].astype(float)
            soft, loud = song_range(rdb)
            vmax = np.percentile(vdb, 99.5)
            y = np.load(pathlib.Path(a.audio) / f"{sid}.npy").astype(np.float32)
            wav = out / f"{sid}.wav"
            sf.write(wav, y, SR)
            notes = ST.transcribe(wav, onset_thr=0.9, ckpt=pathlib.Path(a.ckpt), frame_thr=0.5)
            wav.unlink()
            hum = np.load(m2m / "midi_numpy" / f"{sid}.npy")
            hum = hum[np.lexsort((-hum[:, 2], hum[:, 0]))]
            mel = hum[top_voice(hum)]
            on = np.array([g["onset"] for g in notes])
            pc = np.array([g["pitch"] % 12 for g in notes])
            res = []
            for t, _, p in mel[:, :3]:
                i0, i1 = max(0, int((t - 0.1) / hop)), int((t + 0.1) / hop) + 1
                if i0 >= len(rdb):
                    continue
                v, r = np.median(vdb[i0:i1]), np.median(rdb[i0:i1])
                if not (v > vmax - 35 and v > -60):                 # the voice is not actually sounding here
                    continue
                pos = (r - soft) / max(loud - soft, 1e-6)
                hit = bool(np.any((np.abs(on - t) <= 0.15) & (pc == int(p) % 12))) if len(on) else False
                res.append([band(pos, r < soft), int(hit)])
            rows.append(dict(song=sid, notes=res))
            print(f"{sid}: {len(res)} sung notes, recall {np.mean([h for _, h in res]) if res else 0:.3f}", flush=True)
        (out / f"rows_{a.shard}.json").write_text(json.dumps(rows))
        if a.nshards > 1:
            return
    rows = [r for f in sorted(out.glob("rows_*.json")) for r in json.loads(f.read_text())]
    allb = np.array([n for r in rows for n in r["notes"]])
    print(f"\n{len(rows)} songs, {len(allb)} sung notes (human top line where the voice sounds)")
    print(f"{'instrumental level (song-relative)':<36}{'notes':>8}{'share':>8}{'base recall':>13}{'cumulative recall':>19}")
    cum_n = cum_h = 0
    for b, name in enumerate(NAMES):
        sel = allb[allb[:, 0] == b]
        cum_n += len(sel)
        cum_h += sel[:, 1].sum() if len(sel) else 0
        rec = sel[:, 1].mean() if len(sel) else float("nan")
        print(f"{name:<36}{len(sel):>8d}{len(sel) / len(allb):>8.1%}{rec:>13.3f}{cum_h / max(cum_n, 1):>19.3f}")
    print(f"{'all':<36}{len(allb):>8d}{'':>8}{allb[:, 1].mean():>13.3f}")
    print("cumulative = recall over every band up to and including this one (what a threshold there would cover)")


if __name__ == "__main__":
    main()
