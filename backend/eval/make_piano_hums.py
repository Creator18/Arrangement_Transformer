"""Piano hums for stage 1: rendered piano clip -> Basic Pitch pitch map -> sine-tone hum, labels kept (CPU).

Stage 1 of the hum plan teaches the frozen solo-piano transcriber (LoRA) to
read piano notes back from a HUM of piano.  Input stays audio, so the frozen
front end and model see the format they know: each clip of the rendered
multi-piano MAESTRO set (labels exact by construction) is turned into the
same kind of hum as eval/hum.py — every Basic Pitch contour bin (3 per
semitone) resynthesised as a sine, its amplitude = salience x the clip's own
energy at that pitch, so swells and re-strikes stay audible.

Writes <out>/<split>/<clip>.npz with y = hum (int16, 24 kHz, same length as
the clip), on, fr (the clip's own labels, unchanged), piano.

    python eval/make_piano_hums.py --src /scratch/.../maestro_render_multi --out <dir> \\
        --shard i/n [--max-train 10000]
"""
from __future__ import annotations

import argparse
import os
import pathlib
import random
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FLOOR = 0.15
# Basic Pitch's contour rises late: measured on a rendered clip, the hum's
# onset envelope peaks 3 frames (48 ms) after the piano's own.  Shifted back,
# so the hum lines up with the clip's labels.
SHIFT_S = 0.048


def hum_audio(y22: np.ndarray, sal: np.ndarray, out_sr: int, n_out: int) -> np.ndarray:
    """Sine resynthesis of a Basic Pitch contour map (T, 264) at out_sr, n_out samples."""
    import librosa
    from basic_pitch import constants as C
    from scipy.ndimage import uniform_filter1d
    sr, hop = C.AUDIO_SAMPLE_RATE, C.FFT_HOP
    B = sal.shape[1]
    E = np.abs(librosa.cqt(y22, sr=sr, hop_length=hop, fmin=C.ANNOTATIONS_BASE_FREQUENCY,
                           n_bins=B, bins_per_octave=12 * C.CONTOURS_BINS_PER_SEMITONE)).T
    n = min(len(sal), len(E)); sal, E = sal[:n], E[:n]
    pos = E[E > 0]
    E = E / (np.percentile(pos, 99) + 1e-9) if pos.size else E
    amp = uniform_filter1d(np.where(sal >= FLOOR, sal, 0) * np.sqrt(np.clip(E, 0, 1)), 2, axis=0)
    f = C.ANNOTATIONS_BASE_FREQUENCY * 2 ** (np.arange(B) / (12 * C.CONTOURS_BINS_PER_SEMITONE))
    t = np.arange(n_out) / out_sr
    tf = (np.arange(n) + 0.5) * hop / sr - SHIFT_S
    hum = np.zeros(n_out, np.float64)
    for b in np.flatnonzero(amp.max(0) > 0):
        hum += np.interp(t, tf, amp[:, b], right=0.0) * np.sin(2 * np.pi * f[b] * t)
    pk = np.abs(hum).max()
    return (hum / pk * 0.9 if pk > 0 else hum).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="train,validation,test")
    ap.add_argument("--max-train", type=int, default=0, help="a fixed random subset of train clips")
    ap.add_argument("--shard", default="0/1"); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    import soundfile as sf
    from scipy.signal import resample_poly
    from basic_pitch import inference, ICASSP_2022_MODEL_PATH
    from app.ml.spectral import frontend as S
    model = inference.Model(ICASSP_2022_MODEL_PATH)
    si, sn = (int(v) for v in a.shard.split("/"))
    tmp = pathlib.Path(os.environ.get("TMPDIR", tempfile.gettempdir())) / f"hum_{os.getpid()}.wav"
    done = 0
    for split in a.splits.split(","):
        paths = sorted(p for p in (pathlib.Path(a.src) / split).glob("*.npz") if not p.name.endswith(".tmp.npz"))
        if split == "train" and a.max_train:
            paths = sorted(random.Random(0).sample(paths, min(a.max_train, len(paths))))
        out = pathlib.Path(a.out) / split; out.mkdir(parents=True, exist_ok=True)
        for p in paths[si::sn][: a.limit or None]:
            dst = out / p.name
            if dst.exists():
                continue
            z = np.load(p)
            y = z["y"].astype(np.float32) / 32767
            y22 = resample_poly(y, 147, 160).astype(np.float32)             # 24 kHz -> 22.05 kHz
            sf.write(tmp, y22, 22050)
            sal = inference.run_inference(tmp, model)["contour"]
            hum = hum_audio(y22, sal, S.SR, len(y))
            t = dst.with_name(dst.stem + ".tmp.npz")
            np.savez_compressed(t, y=(np.clip(hum, -1, 1) * 32767).astype(np.int16), on=z["on"], fr=z["fr"],
                                piano=z["piano"] if "piano" in z.files else "salamander")
            t.replace(dst); done += 1
            if done % 50 == 0:
                print(f"  {done} hums", flush=True)
    tmp.unlink(missing_ok=True)
    print(f"{done} hums written", flush=True)


if __name__ == "__main__":
    main()
