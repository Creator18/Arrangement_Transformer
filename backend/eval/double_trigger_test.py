"""Can a decoder rule remove double triggers without losing real fast repeats?

error_breakdown.py: 310 of 1,285 false notes are "double" — a same-key
estimate within 150 ms of a matched one, one strike heard twice.  But real
fast repeats exist too (repeat recall 0.902), so the rule must tell them apart.

For consecutive estimates on the same key with onset gap < G, the second is
merged into the first (first note's offset extended) when:

  gap       always (gap alone)
  weaker    its onset peak is below the first's by more than d
  softer    its predicted velocity is below the first's by more than dv
  no-rise   the onset probability never fell below r between the two peaks
            (one continuous activation, not two separate attacks)

Exploratory, tuned on the same 12 recordings it is scored on; a winner is
re-checked on held-out clips before it ships.

    python eval/double_trigger_test.py --dir E:/arrangement_transformer/maestro_real
"""
from __future__ import annotations

import argparse
import collections
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
from real_audio_eval import midi_truth, score       # noqa: E402

THR = 0.85


def merge(notes, onk, fps, gap, test):
    out, last = [], {}
    for n in sorted(notes, key=lambda n: (n["onset"], n["pitch"])):
        k = n["pitch"]
        f = int(round(n["onset"] * fps))
        n = dict(n, _f=f, _peak=float(onk[k - 21, max(0, f - 1):f + 2].max()))
        p = last.get(k)
        if p is not None and n["onset"] - p["onset"] < gap and test(p, n, onk[k - 21]):
            p["offset"] = max(p["offset"], n["offset"])
            continue
        out.append(n); last[k] = n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    from app.services import spectral_transcription as ST
    from app.ml.spectral import frontend as S
    from app.ml.transcriber.decode import decode_notes

    def dip(r):
        return lambda p, n, row: row[p["_f"]:n["_f"] + 1].min() >= r
    rules = [("baseline", None, None)]
    for g in (0.06, 0.1, 0.15):
        rules += [(f"gap<{g * 1000:.0f}", g, lambda p, n, row: True)]
        rules += [(f"gap<{g * 1000:.0f} weaker d{d}", g,
                   (lambda d: lambda p, n, row: n["_peak"] < p["_peak"] - d)(d)) for d in (0.0, 0.05)]
        rules += [(f"gap<{g * 1000:.0f} softer dv{dv}", g,
                   (lambda dv: lambda p, n, row: n["velocity"] < p["velocity"] - dv)(dv)) for dv in (10, 20)]
        rules += [(f"gap<{g * 1000:.0f} no-rise r{r}", g, dip(r)) for r in (0.3, 0.5)]

    rows = collections.defaultdict(list)
    rep_hit, rep_tot = collections.Counter(), 0
    for wav in sorted(pathlib.Path(args.dir).glob("*.wav")):
        ref = midi_truth(wav.with_suffix(".midi"))
        rep_tot += sum(r["gap"] < 0.15 for r in ref)
        on, fr, vl = ST.rolls(wav)
        onk = on if on.shape[0] == 88 else on.T
        base = decode_notes(on, fr, vl, THR, 0.5, min_frames=1, min_note_s=0.0, keep_restruck=True)
        for name, g, test in rules:
            est = base if g is None else merge(base, onk, S.FPS, g, test)
            p, r, f, fo, hit, _ = score(ref, est)
            rows[name].append((f, p, r, fo, len(est)))
            rep_hit[name] += sum(1 for i in hit if ref[i]["gap"] < 0.15)
        print(f"  {wav.stem[:50]}", flush=True)

    b = np.array(rows["baseline"])[:, 0].mean()
    print(f"\n{'rule':<26}{'F1':>7}{'P':>7}{'R':>7}{'F1+off':>8}{'notes':>8}{'repeats R':>11}")
    for name, *_ in rules:
        a = np.array(rows[name]); m = a[:, :4].mean(0)
        print(f"{name:<26}{m[0]:>7.4f}{m[1]:>7.3f}{m[2]:>7.3f}{m[3]:>8.3f}{int(a[:, 4].sum()):>8}"
              f"{rep_hit[name] / rep_tot:>11.3f}   dF1 {m[0] - b:+.4f}")
    print(f"\n({rep_tot} true notes re-struck within 150 ms)")


if __name__ == "__main__":
    main()
