"""The song as a "hum": every voice in one plain sine tone, continuous pitch kept (CPU/GPU, local).

The user's idea: arrange from a hum, not from the song.  Basic Pitch's contour
output is a pitch-salience map (3 bins per semitone, A0 up, 86 fps), with
instrument timbre folded away.  Each bin is resynthesised as a sine at its own
frequency, so slides and vibrato survive as the bins they move through.

Loudness: salience says WHETHER a pitch sounds, not how loud.  Each sine's
amplitude is salience x the song's own energy at that pitch (CQT, same bins),
so swells, tremolo and re-articulations of a held note — where a pianist
would re-strike — stay audible in the hum.

Writes <out>/<name>_hum.wav (the hum), <name>_hum_mix.wav (hum left, song
right, to check it against the original) and <name>_contour.npy.

    python eval/hum.py <audio> --out <dir>
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio"); ap.add_argument("--out", required=True)
    ap.add_argument("--floor", type=float, default=0.15, help="salience below this is silent")
    a = ap.parse_args()
    import librosa
    import soundfile as sf
    from basic_pitch import inference, constants as C
    from scipy.ndimage import uniform_filter1d

    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    name = pathlib.Path(a.audio).parent.name[:8] if pathlib.Path(a.audio).stem == "audio" else pathlib.Path(a.audio).stem
    sr, hop = C.AUDIO_SAMPLE_RATE, C.FFT_HOP
    mo, _, notes = inference.predict(a.audio)
    sal = mo["contour"].astype(np.float32)                         # (T, 264)
    T, B = sal.shape
    y, _ = librosa.load(a.audio, sr=sr, mono=True)
    E = np.abs(librosa.cqt(y, sr=sr, hop_length=hop, fmin=C.ANNOTATIONS_BASE_FREQUENCY,
                           n_bins=B, bins_per_octave=12 * C.CONTOURS_BINS_PER_SEMITONE)).T
    n = min(T, len(E)); sal, E = sal[:n], E[:n]
    E = E / (np.percentile(E[E > 0], 99) + 1e-9)
    amp = np.where(sal >= a.floor, sal, 0) * np.sqrt(np.clip(E, 0, 1))
    amp = uniform_filter1d(amp, 2, axis=0)                          # ~23 ms: no clicks, keeps pulses
    f = C.ANNOTATIONS_BASE_FREQUENCY * 2 ** (np.arange(B) / (12 * C.CONTOURS_BINS_PER_SEMITONE))
    N = n * hop; hum = np.zeros(N, np.float64)
    t_frames = (np.arange(n) + 0.5) * hop
    live = np.flatnonzero(amp.max(0) > 0)
    for b in live:                                                  # one phase-continuous sine per bin
        env = np.interp(np.arange(N), t_frames, amp[:, b])
        if env.max() <= 0:
            continue
        hum += env * np.sin(2 * np.pi * f[b] * np.arange(N) / sr)
    hum /= np.abs(hum).max() + 1e-9; hum *= 0.9
    sf.write(out / f"{name}_hum.wav", hum.astype(np.float32), sr)
    yy = y[:N] / (np.abs(y).max() + 1e-9) * 0.9
    sf.write(out / f"{name}_hum_mix.wav", np.stack([hum[:len(yy)], yy], 1).astype(np.float32), sr)
    np.save(out / f"{name}_contour.npy", mo["contour"])
    print(f"{n} frames ({n * hop / sr:.0f} s), {len(live)} of {B} pitch bins used, "
          f"Basic Pitch's own notes: {len(notes)}")
    print(f"wrote {out / (name + '_hum.wav')} and {name}_hum_mix.wav")


if __name__ == "__main__":
    main()
