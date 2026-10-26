"""How much solo voice is in MUSDB18-HQ?  From the true stems: a 100 ms frame is "solo voice" when the vocal stem
is active (within 35 dB of the song's loudest vocal frame) and the rest (bass + drums + other) sits at least
`margin` dB below the vocal.  Solo stretches shorter than a minimum length are ignored.

    python eval/musdb_solo_stats.py /scratch/basu.ang/datasets/musdb18hq [--out stats.json]
"""
import argparse
import json
import pathlib

import numpy as np
import soundfile as sf

HOP = 0.1
FINE = 0.02     # 20 ms frames for the moment-level count


def frame_db(x, sr, hop=None):
    h = int((hop or HOP) * sr)
    n = len(x) // h
    e = (x[: n * h].reshape(n, h) ** 2).mean(1)
    return 10 * np.log10(e + 1e-12)


def runs(mask):
    """(start, length) of True runs, in frames."""
    d = np.diff(np.r_[0, mask.astype(int), 0])
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return list(zip(s, e - s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    margins, mins = (10, 15, 20), (1.0, 2.0, 4.0)
    rows = []
    for split in ("train", "test"):
        for d in sorted((pathlib.Path(a.root) / split).iterdir()):
            stems = {}
            for k in ("vocals", "bass", "drums", "other"):
                x, sr = sf.read(d / f"{k}.wav", dtype="float32")
                stems[k] = x.mean(1) if x.ndim > 1 else x
            v = frame_db(stems["vocals"], sr)
            r = frame_db(stems["bass"] + stems["drums"] + stems["other"], sr)
            active = v > v.max() - 35
            row = {"split": split, "song": d.name, "dur": len(v) * HOP, "vocal_s": float(active.sum() * HOP)}
            for m in margins:
                solo = active & (r < v - m)
                for mn in mins:
                    row[f"solo_{m}dB_{mn:g}s"] = float(sum(L for _, L in runs(solo) if L * HOP >= mn) * HOP)
            # moment level: vocal note starts (rises of 6 dB+ in 20 ms frames), and how quiet the band is in the
            # 250 ms around each one -- pickups and entries before the band count here even if they are short
            vf = frame_db(stems["vocals"], sr, FINE)
            rf = frame_db(stems["bass"] + stems["drums"] + stems["other"], sr, FINE)
            act = vf > vf.max() - 35
            rise = np.r_[False, (vf[1:] - vf[:-1] > 6) & act[1:]]
            on = np.flatnonzero(rise)
            on = on[np.r_[True, np.diff(on) > 5]]                     # one start per 100 ms
            w = int(0.125 / FINE)
            gap = np.array([np.median(vf[max(0, i - 1):i + w]) - np.median(rf[max(0, i - w):i + w]) for i in on])
            row["starts"] = int(len(on))
            for m in margins:
                row[f"clear_starts_{m}dB"] = int((gap >= m).sum())
            rows.append(row)
            print(f"{split:5s} {d.name[:40]:40s} {row['dur']:6.0f}s vocal {row['vocal_s']:5.0f}s  "
                  f"solo(20dB,2s) {row['solo_20dB_2s']:5.1f}s  solo(10dB,1s) {row['solo_10dB_1s']:5.1f}s", flush=True)
    print()
    for split in ("train", "test", "all"):
        rs = [r for r in rows if split == "all" or r["split"] == split]
        voc = sum(r["vocal_s"] for r in rs)
        print(f"== {split}: {len(rs)} songs, {sum(r['dur'] for r in rs) / 3600:.1f} h, vocals {voc / 3600:.1f} h")
        for m in margins:
            for mn in mins:
                k = f"solo_{m}dB_{mn:g}s"
                tot = sum(r[k] for r in rs)
                has = sum(r[k] > 0 for r in rs)
                print(f"   rest >= {m:2d} dB below voice, runs >= {mn:g} s: {tot / 60:6.1f} min "
                      f"({tot / max(voc, 1):5.1%} of vocal time), songs with any: {has}/{len(rs)}")
        st = sum(r["starts"] for r in rows if split == "all" or r["split"] == split)
        for m in margins:
            c = sum(r[f"clear_starts_{m}dB"] for r in rows if split == "all" or r["split"] == split)
            print(f"   vocal starts with the band >= {m:2d} dB below in the 250 ms around: {c} / {st} ({c / max(st, 1):.1%})")
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
