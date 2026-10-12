"""Does the spectral model use long context?  Score vs inference window length.

If note F1 flattens by ~10 s, transcription does not need more context, and
frame selection / segment recurrence belong in the arrangement stage instead.
If it keeps climbing, a longer-context transcriber is worth building.

Caveat: the model was trained on 8 s crops, so windows beyond 8 s test whether
it can USE context it never trained on (rotary positions allow it), not what a
model trained on longer crops could do.

12 real MAESTRO test recordings, unfiltered MIDI truth, fixed decoder.

    python eval/context_length_check.py --dir E:/arrangement_transformer/maestro_real
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
from real_audio_eval import midi_truth, score       # noqa: E402

WINDOWS = [2.5, 5.0, 10.0, 20.0, 40.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--thr", type=float, default=0.85)
    args = ap.parse_args()
    from app.services import spectral_transcription as ST
    from app.ml.transcriber.decode import decode_notes

    wavs = sorted(pathlib.Path(args.dir).glob("*.wav"))
    res = {w: [] for w in WINDOWS}
    for wav in wavs:
        ref = midi_truth(wav.with_suffix(".midi"))
        for w in WINDOWS:
            ST.WINDOW_S, ST.EDGE_S = w, min(2.0, w / 4)
            on, fr, vl = ST.rolls(wav)
            est = decode_notes(on, fr, vl, args.thr, 0.5, min_frames=1,
                               min_note_s=0.0, keep_restruck=True)
            p, r, f, fo, _, _ = score(ref, est)
            res[w].append((f, p, r, fo))
        print(f"  {wav.stem[:50]}: " + "  ".join(f"{w:g}s {res[w][-1][0]:.3f}" for w in WINDOWS),
              flush=True)
    print(f"\n{len(wavs)} recordings, thr {args.thr}\n{'window':>8}{'edge':>6}{'F1':>8}{'P':>7}{'R':>7}{'F1+off':>8}")
    for w in WINDOWS:
        a = np.mean(res[w], 0)
        print(f"{w:>7g}s{min(2.0, w / 4):>5g}s{a[0]:>8.3f}{a[1]:>7.3f}{a[2]:>7.3f}{a[3]:>8.3f}")


if __name__ == "__main__":
    main()
