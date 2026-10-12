"""Does the model fall apart on very fast playing (Flight of the Bumblebee)?

The user hears Flight of the Bumblebee "butchered" and suspects the 16 ms
frame rate is too coarse.  Two measurements:

  A  CONTROLLED: a Bumblebee-style passage (chromatic 16th-note runs in the
     right hand, optionally doubled an octave down, staccato left-hand chords
     on the beat) rendered through each sampled piano at rising tempos, so
     the truth is exact and only the speed changes.  Note F1, recall, and
     onset error per tempo.  At 16 ms frames a 16th note at 220 BPM (68 ms)
     still spans ~4 frames, so if the frame rate were the limit it would
     show as recall collapsing with tempo.

  B  REAL: on the 12 real MAESTRO recordings, recall binned by local note
     rate (true onsets per second within +-0.5 s).

    python eval/speed_test.py --dir E:/arrangement_transformer/maestro_real
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
from real_audio_eval import midi_truth, score       # noqa: E402

TEMPOS = [100, 130, 160, 190, 220, 260]
# the opening figure: chromatic descent and turn, in 16ths (A5 = 81)
FIGURE = [81, 80, 79, 78, 77, 76, 77, 76, 75, 74, 73, 72, 73, 72, 71, 70,
          69, 70, 71, 72, 73, 74, 75, 76, 77, 76, 75, 74, 75, 74, 73, 72]


def passage(bpm: float, octaves: bool, bars: int = 8):
    six = 60.0 / bpm / 4
    rng = np.random.default_rng(int(bpm) + 7 * octaves)
    notes, t = [], 0.5
    for b in range(bars):
        shift = [0, 0, -2, 3][b % 4]
        for i, p in enumerate(FIGURE[:16]):
            v = int(rng.integers(68, 92))
            notes.append({"pitch": p + shift, "onset": t, "offset": t + six * 0.95, "velocity": v})
            if octaves:
                notes.append({"pitch": p + shift - 12, "onset": t, "offset": t + six * 0.95, "velocity": v - 6})
            if i % 4 == 0 and not octaves:              # left hand: A minor-ish stabs on the beat
                for q in (45 + shift, 52 + shift, 57 + shift):
                    notes.append({"pitch": q, "onset": t, "offset": t + six * 1.5, "velocity": 60})
            t += six
    return sorted(notes, key=lambda n: (n["onset"], n["pitch"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    import soundfile as sf
    from app.ml.spectral import render as R, frontend as S
    from app.services import spectral_transcription as ST

    lat = R.latencies()
    tmp = pathlib.Path(tempfile.mkdtemp())
    print("A  CONTROLLED Bumblebee-style passage, 8 bars, all pianos")
    print(f"{'bpm':>5}{'16th ms':>9}  {'texture':<14}{'F1':>7}{'P':>7}{'R':>7}{'onset err':>11}")
    for bpm in TEMPOS:
        for octaves in (False, True):
            ref = passage(bpm, octaves)
            rows, errs = [], []
            for name in R.pianos():
                if name == "intimate":
                    continue
                y = R.render_with(name, ref)
                wav = tmp / f"{name}_{bpm}_{octaves}.wav"
                sf.write(wav, y, S.SR)
                est = ST.transcribe(wav)
                for n in est:                    # undo the label convention's attack latency
                    n["onset"] -= lat.get(name, 0.0); n["offset"] -= lat.get(name, 0.0)
                p, r, f, fo, hit, pairs = score(ref, est)
                rows.append((f, p, r))
                errs += [abs(est[j]["onset"] - ref[i]["onset"]) * 1000 for i, j in pairs]
            m = np.mean(rows, 0)
            tex = "octaves" if octaves else "run + LH chords"
            print(f"{bpm:>5}{60000 / bpm / 4:>8.0f}  {tex:<14}{m[0]:>7.3f}{m[1]:>7.3f}{m[2]:>7.3f}"
                  f"{np.median(errs) if errs else float('nan'):>9.1f}ms", flush=True)

    print("\nB  REAL recordings: recall by local note rate (true onsets/s within +-0.5 s)")
    from app.ml.transcriber.decode import decode_notes
    bins = [(0, 8), (8, 12), (12, 16), (16, 20), (20, 25), (25, 1000)]
    hit_b, tot_b = np.zeros(len(bins)), np.zeros(len(bins))
    for wav in sorted(pathlib.Path(args.dir).glob("*.wav")):
        ref = midi_truth(wav.with_suffix(".midi"))
        on, fr, vl = ST.rolls(wav)
        est = decode_notes(on, fr, vl, ST.ONSET_THR, 0.5, min_frames=1, min_note_s=0.0,
                           keep_restruck=True)
        *_, hit, _ = score(ref, est)
        t = np.array([r["onset"] for r in ref])
        rate = np.searchsorted(t, t + 0.5) - np.searchsorted(t, t - 0.5)
        for i, rt in enumerate(rate):
            for b, (lo, hi) in enumerate(bins):
                if lo <= rt < hi:
                    tot_b[b] += 1; hit_b[b] += i in hit
    for b, (lo, hi) in enumerate(bins):
        lab = f"{lo}-{hi}" if hi < 1000 else f"{lo}+"
        print(f"   {lab:>6} notes/s  {int(tot_b[b]):>6} notes   recall {hit_b[b] / max(1, tot_b[b]):.3f}")


if __name__ == "__main__":
    main()
