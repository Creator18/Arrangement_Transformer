"""
A/B piano libraries objectively, and render the same passage from each to hear.

sfz_sampler already resolves ARRANGEMENT_SFZ from the environment, so trying a
library needs no code change — only a path.  What was missing is a way to say
whether a candidate is actually better rather than merely different, and a
way to hear them back to back on identical material.

Measures chosen for BODY, which is what Salamander lacks — not brightness,
which was the wrong target earlier:

  centroid_hz   spectral centroid of the attack.  Where the energy sits.
  bw_60db       highest frequency within 60 dB of peak — the practical
                bandwidth, measured on the ATTACK window.  Measuring whole
                files gives nonsense here: piano HF lives in the first ~200 ms
                and vanishes into a ten-second average.
  t60_ms        time for the note to fall 60 dB.  Sustain length — a big part
                of what "body" means.
  low_mid_pct   energy in 150-800 Hz, where a piano's weight and warmth live.
                This is the number to watch for body specifically.
  release_pct   energy after key release relative to total — resonance and
                room tail rather than the struck note.

Usage:
    python backend/eval/compare_pianos.py                       # current only
    python backend/eval/compare_pianos.py --sfz "A=/path/a.sfz" "B=/path/b.sfz"

Writes a WAV per library to backend/eval/data/piano_ab/ so they can be
compared by ear, which remains the deciding test.
"""
from __future__ import annotations

import argparse, os, pathlib, sys
import numpy as np
import soundfile as sf

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

SR = 44100
OUT = ROOT / "backend/eval/data/piano_ab"

# A short passage that exercises what matters: a low anchor for weight, a
# mid chord for warmth, a melodic line for attack, and a held pedal so the
# resonance and release tail are audible.
PASSAGE = (
    [{"pitch": 36, "onset": 0.00, "offset": 2.60, "velocity": 88}] +
    [{"pitch": p,  "onset": 0.05, "offset": 2.40, "velocity": 70}
     for p in (52, 55, 60)] +
    [{"pitch": p,  "onset": 0.9 + i * 0.28, "offset": 1.2 + i * 0.28, "velocity": 96}
     for i, p in enumerate((67, 69, 71, 72, 76))]
)
PEDAL = [{"onset": 0.0, "offset": 3.4}]


def measure(y: np.ndarray, sr: int = SR) -> dict:
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = np.asarray(y, dtype=float)
    peak = np.abs(y).max() or 1.0

    i0 = int(np.argmax(np.abs(y) > 0.02 * peak))
    atk = y[i0:i0 + int(sr * 0.15)]
    S = np.abs(np.fft.rfft(atk * np.hanning(len(atk)))) + 1e-12
    f = np.fft.rfftfreq(len(atk), 1 / sr)
    db = 20 * np.log10(S / S.max())
    within = np.where(db > -60)[0]

    tot = (S ** 2).sum()
    low_mid = (S[(f >= 150) & (f < 800)] ** 2).sum() / tot

    # T60 from the smoothed envelope after the last onset.
    env = np.abs(y)
    win = int(sr * 0.02)
    env = np.convolve(env, np.ones(win) / win, mode="same")
    pk = env.argmax()
    tail = env[pk:]
    thr = env[pk] * (10 ** (-60 / 20))
    below = np.where(tail < thr)[0]
    t60 = (below[0] / sr * 1000) if len(below) else len(tail) / sr * 1000

    rel_i = int(3.4 * sr)
    release = (y[rel_i:] ** 2).sum() / max((y ** 2).sum(), 1e-12) if len(y) > rel_i else 0.0

    return {
        "centroid_hz": float((f * S).sum() / S.sum()),
        "bw_60db": float(f[within[-1]]) if len(within) else 0.0,
        "t60_ms": float(t60),
        "low_mid_pct": float(100 * low_mid),
        "release_pct": float(100 * release),
        "peak": float(peak),
    }


def render_with(sfz_path: str | None) -> np.ndarray:
    if sfz_path:
        os.environ["ARRANGEMENT_SFZ"] = sfz_path
    for m in [m for m in list(sys.modules) if "sfz_sampler" in m or "sampler" in m]:
        sys.modules.pop(m, None)
    from app.services.sfz_sampler import render_notes_sfz
    return np.asarray(render_notes_sfz(PASSAGE, sr=SR, pedal=PEDAL))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sfz", nargs="*", default=[],
                    help='NAME=/path/to/lib.sfz, repeatable')
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    libs = [("current", None)]
    for spec in args.sfz:
        name, _, path = spec.partition("=")
        libs.append((name or pathlib.Path(path).stem, path))

    print(f"\n{'library':<16}{'centroid':>10}{'bw60':>9}{'T60 ms':>9}"
          f"{'low-mid %':>11}{'release %':>11}")
    print("-" * 66)
    for name, path in libs:
        try:
            y = render_with(path)
        except Exception as e:
            print(f"{name:<16} FAILED  {type(e).__name__}: {str(e)[:38]}")
            continue
        m = measure(y)
        wav = OUT / f"{name}.wav"
        sf.write(str(wav), y.astype(np.float32), SR)
        print(f"{name:<16}{m['centroid_hz']:>10.0f}{m['bw_60db']:>9.0f}"
              f"{m['t60_ms']:>9.0f}{m['low_mid_pct']:>11.2f}{m['release_pct']:>11.3f}")
    print(f"\nWAVs in {OUT} — the ear decides, these numbers only narrow it down.")


if __name__ == "__main__":
    main()


# ── Baseline, 2026-08-10 — Salamander Grand Piano V3 ─────────────────────────
#
#   centroid  661 Hz     bw60 4460 Hz     T60 4138 ms
#   low-mid   60.37%     release 0.071%
#
# The surprise is that Salamander is NOT thin.  60% of its energy sits in
# 150-800 Hz, which is warmth, and it decays for over four seconds.  The one
# figure that is nearly zero is the release tail: 0.071% of total energy after
# the damper drops.
#
# That reframes "needs more body".  The instrument has weight; what it has no
# trace of is ROOM.  Salamander is close-miked with almost no ambience, so
# every note stops dead at the boundary of the sample instead of decaying into
# a space.  That is what makes it sound like samples rather than a piano
# somewhere.
#
# Which means there are two routes, and the cheaper one is not a new library:
#   1. a library recorded with room mics — raises release_pct directly
#   2. convolution reverb over our dry render — adds the space without
#      changing the instrument, and costs one impulse response
#
# Route 2 should be tried first: it is reversible, needs no download, and if
# room is genuinely what is missing it will show up immediately in
# release_pct and in the ear.
