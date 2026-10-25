"""Context-aware onset threshold (evaluation only; the app still uses one threshold).

Quick notes lose confidence to their neighbours, so in busy spots a lower bar finds them: a strict
first pass (base threshold) gives the onsets; right AFTER each one, for `win` frames, the bar drops
to `low` -- for the same key only ("same": fast repeats) or for every key ("any": runs too).
Everywhere else the strict bar stays, so wrong notes in quiet spots are not let in.

    notes = context_decode(on, fr, vl, base=0.9, low=0.6, win=9, mode="any")
"""
import numpy as np

from app.ml.transcriber.decode import decode_notes


def threshold_map(on, base, low, win, mode):
    first = decode_notes(on, on * 0, None, base, 0.5, min_frames=1, min_note_s=0.0, keep_restruck=True)
    thr = np.full(on.shape, float(base), np.float32)
    T = on.shape[1]
    for n in first:
        k = int(n["pitch"]) - 21; f = int(round(n["onset"] * 62.5))
        a, b = f + 2, min(T, f + 1 + win)                 # just after the strike (its own +-1 run excluded)
        if a >= b:
            continue
        if mode in ("same", "both"):
            thr[k, a:b] = np.minimum(thr[k, a:b], low)
        if mode in ("any", "both"):
            thr[:, f + 1:b] = np.minimum(thr[:, f + 1:b], low)
    return thr


def context_decode(on, fr, vl, base=0.9, low=0.6, win=9, mode="any", frame_thr=0.5):
    return decode_notes(on, fr, vl, threshold_map(on, base, low, win, mode), frame_thr,
                        min_frames=1, min_note_s=0.0, keep_restruck=True)
