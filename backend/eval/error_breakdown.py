"""Where do the spectral model's remaining errors come from?

Asked to decide whether "look between frames where surprise is high" has
headroom.  Every MISSED true note is put in exactly one bucket, first match wins:

  timing    an unmatched estimate of the same key lies 50-150 ms away —
            found, but outside the onset tolerance.  Finer timing fixes it.
  merged    another true onset of the same key lies within 48 ms (3 frames)
            and was matched instead — two strikes heard as one.
  pitch     an unmatched estimate within 50 ms is an octave or semitone off.
  soft      none of the above, velocity < 50.
  missed    none of the above.

Every FALSE note likewise: timing (paired above), pitch (paired above),
double (same key matched within 150 ms — one strike heard twice), other.

Plus the onset error of matched notes: how much a between-frame estimate
could tighten timing at all.

12 real MAESTRO test recordings, unfiltered MIDI truth, thr 0.85, fixed decoder.

    python eval/error_breakdown.py --dir E:/arrangement_transformer/maestro_real
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--thr", type=float, default=0.85)
    args = ap.parse_args()
    from app.services import spectral_transcription as ST
    from app.ml.transcriber.decode import decode_notes

    miss, false = collections.Counter(), collections.Counter()
    errs, n_ref, n_est = [], 0, 0
    for wav in sorted(pathlib.Path(args.dir).glob("*.wav")):
        ref = midi_truth(wav.with_suffix(".midi"))
        on, fr, vl = ST.rolls(wav)
        est = decode_notes(on, fr, vl, args.thr, 0.5, min_frames=1, min_note_s=0.0,
                           keep_restruck=True)
        *_, pairs = score(ref, est)
        n_ref += len(ref); n_est += len(est)
        mr = {i for i, _ in pairs}; me = {j for _, j in pairs}
        errs += [est[j]["onset"] - ref[i]["onset"] for i, j in pairs]
        by_key = collections.defaultdict(list)
        for j, e in enumerate(est):
            by_key[e["pitch"]].append(j)
        ref_key = collections.defaultdict(list)
        for i, r in enumerate(ref):
            ref_key[r["pitch"]].append(i)
        used = {}
        for i, r in enumerate(ref):
            if i in mr:
                continue
            t, k = r["onset"], r["pitch"]
            near = lambda kk, lo, hi: [j for j in by_key.get(kk, []) if j not in me and j not in used
                                       and lo <= abs(est[j]["onset"] - t) <= hi]
            if (c := near(k, 0.05, 0.15)):
                miss["timing"] += 1; used[c[0]] = "timing"
            elif any(i2 != i and i2 in mr and abs(ref[i2]["onset"] - t) <= 0.048 for i2 in ref_key[k]):
                miss["merged"] += 1
            elif (c := [j for d in (-12, 12, -1, 1) for j in near(k + d, 0, 0.05)]):
                miss["pitch"] += 1; used[c[0]] = "pitch"
            elif r["velocity"] < 50:
                miss["soft"] += 1
            else:
                miss["missed"] += 1
        matched_key_times = collections.defaultdict(list)
        for i, j in pairs:
            matched_key_times[est[j]["pitch"]].append(est[j]["onset"])
        for j, e in enumerate(est):
            if j in me:
                continue
            if j in used:
                false[used[j]] += 1
            elif any(abs(t - e["onset"]) <= 0.15 for t in matched_key_times[e["pitch"]]):
                false["double"] += 1
            else:
                false["other"] += 1
        print(f"  {wav.stem[:50]}", flush=True)

    e = np.abs(np.array(errs)) * 1000
    tot_m, tot_f = sum(miss.values()), sum(false.values())
    print(f"\n{n_ref} true notes, {n_est} estimated, {len(errs)} matched")
    print(f"\nMISSED true notes: {tot_m} ({tot_m / n_ref:.1%} of truth)")
    for k in ("timing", "merged", "pitch", "soft", "missed"):
        print(f"   {k:<8}{miss[k]:>6}  {miss[k] / max(1, tot_m):>6.1%}")
    print(f"\nFALSE notes: {tot_f} ({tot_f / n_est:.1%} of output)")
    for k in ("timing", "pitch", "double", "other"):
        print(f"   {k:<8}{false[k]:>6}  {false[k] / max(1, tot_f):>6.1%}")
    print(f"\nonset error of matched notes (ms): median {np.median(e):.1f}  mean {e.mean():.1f}  "
          f"p90 {np.percentile(e, 90):.1f}   >8 ms {np.mean(e > 8):.1%}  >16 ms {np.mean(e > 16):.1%}"
          f"  >30 ms {np.mean(e > 30):.1%}")
    s = np.array(errs) * 1000
    print(f"signed: median {np.median(s):+.1f} ms (positive = we are late)")


if __name__ == "__main__":
    main()
