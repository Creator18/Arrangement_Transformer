"""Does a soft-aware onset threshold recover missed soft notes without flooding false ones?

soft_miss_check.py found 36% of missed soft notes have onset peaks of 0.5-0.85,
heard but rejected at the 0.85 threshold.  Two rules, decoded at a low
threshold t_lo and then filtered:

  velocity  keep a note if its peak >= 0.85, OR its predicted velocity < v_cut
            (the model itself thinks it is soft)
  masked    keep a note if its peak >= 0.85, OR a louder note (>= v_cut) was
            decoded within 60 ms (a soft note under a loud chord)

Exploratory: tuned on the same 12 recordings it is scored on.  A rule that
wins here gets re-tuned on validation clips before it ships.

    python eval/soft_threshold_test.py --dir E:/arrangement_transformer/maestro_real
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

BASE = 0.85


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    from app.services import spectral_transcription as ST
    from app.ml.spectral import frontend as S
    from app.ml.transcriber.decode import decode_notes

    rules = [("baseline 0.85", None, None, None)]
    rules += [(f"velocity lo{t} v<{v}", "velocity", t, v) for t in (0.3, 0.5, 0.7) for v in (35, 45, 55)]
    rules += [(f"masked   lo{t} v>={v}", "masked", t, v) for t in (0.3, 0.5, 0.7) for v in (60, 70)]
    rows = collections.defaultdict(list)
    soft_hit, soft_tot = collections.Counter(), 0
    for wav in sorted(pathlib.Path(args.dir).glob("*.wav")):
        ref = midi_truth(wav.with_suffix(".midi"))
        soft_tot += sum(r["velocity"] < 50 for r in ref)
        on, fr, vl = ST.rolls(wav)
        onk = on if on.shape[0] == 88 else on.T
        cache = {}
        for name, kind, t, v in rules:
            lo = BASE if kind is None else t
            if lo not in cache:
                ns = decode_notes(on, fr, vl, lo, 0.5, min_frames=1, min_note_s=0.0, keep_restruck=True)
                for n in ns:
                    f = int(round(n["onset"] * S.FPS)); k = n["pitch"] - 21
                    n["_peak"] = float(onk[k, max(0, f - 1):f + 2].max())
                cache[lo] = ns
            ns = cache[lo]
            if kind is None:
                est = ns
            elif kind == "velocity":
                est = [n for n in ns if n["_peak"] >= BASE or n["velocity"] < v]
            else:
                loud = np.array(sorted(n["onset"] for n in ns if n["_peak"] >= BASE and n["velocity"] >= v))
                def near(t0):
                    i = np.searchsorted(loud, t0)
                    return any(abs(loud[j] - t0) <= 0.06 for j in (i - 1, i) if 0 <= j < len(loud))
                est = [n for n in ns if n["_peak"] >= BASE or near(n["onset"])]
            p, r, f1, fo, hit, _ = score(ref, est)
            rows[name].append((f1, p, r, fo, len(est)))
            soft_hit[name] += sum(1 for i in hit if ref[i]["velocity"] < 50)
        print(f"  {wav.stem[:50]}", flush=True)

    print(f"\n{'rule':<24}{'F1':>7}{'P':>7}{'R':>7}{'F1+off':>8}{'notes':>8}{'soft R':>8}")
    b = np.array(rows["baseline 0.85"])
    for name, *_ in rules:
        a = np.array(rows[name]); m = a[:, :4].mean(0)
        print(f"{name:<24}{m[0]:>7.3f}{m[1]:>7.3f}{m[2]:>7.3f}{m[3]:>8.3f}{int(a[:, 4].sum()):>8}"
              f"{soft_hit[name] / soft_tot:>8.3f}   dF1 {m[0] - b[:, 0].mean():+.4f}")


if __name__ == "__main__":
    main()
