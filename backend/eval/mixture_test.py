"""
Does the piano projection survive a MIXTURE, and is Demucs worth its CPU?

Every measurement so far had one instrument sounding at a time.  Real songs
never do.  Two things happen in a mixture that cannot happen in a solo render:

  * Drums are broadband transients with no sparse piano explanation, so they
    smear across the whole dictionary.  Octave ghosting already reached 21.8%
    on sustained material; percussion is the most likely thing to swamp it.
  * Bass and guitar share registers, so the decomposition has to divide energy
    between parts that genuinely overlap in pitch.

It also answers an architecture question with a cost attached.  Demucs runs at
2.40x realtime on 8 threads and is the single largest CPU item in the
pipeline, which matters for a free CPU-only host.  Running the decomposition
on the raw mixture AND on the separated stems states its value as a number
rather than an assumption.

Ground truth, and why it is constructed
---------------------------------------
There is no multitrack pop corpus here, so the mixture is BUILT: a MAESTRO
piece split by register, the low part rendered as bass guitar, the upper part
as steel guitar, optionally doubled by a string pad, plus a drum kit on the
beat grid.  Every pitched note is known because we placed it, and the union of
the parts is exactly MAESTRO's own roll, so the existing ground truth applies
unchanged.

Constructed is a real limitation and worth naming: the parts are all playing
one composer's piano writing rather than idiomatic band parts.  What it buys
is that percussion becomes an isolated variable, which no real recording would
give us.  Drums are deliberately NOT in the ground truth — they have no pitch,
so every unit of energy they attract is by definition error, which is the
cleanest possible way to measure what they cost.

RESULT (4 MAESTRO pieces, 30 s each, 2026-08-11)
------------------------------------------------
    condition                    correct  decay  octave  fifth  other  accounted
    bass+guitar (no drums)         37.2%  26.2%    8.9%   4.4%  23.4%     63.3%
    bass+guitar+drums              25.0%  17.8%    8.6%   4.1%  44.5%     42.8%
    + string pad + drums           28.2%  16.3%    9.7%   5.0%  40.8%     44.6%
    bass+guitar+drums, DEMUCS      37.1%  25.9%    9.1%   4.5%  23.4%     63.0%
    + pad + drums, DEMUCS          38.3%  22.5%   10.2%   5.4%  23.6%     60.8%

Three things, and the third is the surprise.

1. A pitched MIXTURE costs almost nothing beyond timbre.  63.3% sits above
   solo nylon guitar's 56.5%, so bass and guitar sharing registers did not
   confuse the decomposition the way expected.  (Different piece counts, 4 vs
   5, so treat the direction as real and the margin as soft.)

2. Drums cost 20.5 points, 63.3% -> 42.8%, and DEMUCS RECOVERS ESSENTIALLY
   ALL OF IT — 63.0%, within 0.3 points of never having had drums.  That is a
   direct answer to whether separation earns its 2.40x-realtime CPU: for the
   song pipeline, yes, decisively, and it is the difference between a usable
   decomposition and a contaminated one.

3. Percussion does NOT corrupt pitch.  Octave ghosting is flat across the
   drum conditions (8.9% -> 8.6%), and every point drums cost lands in
   'other'.  So broadband transients raise a noise floor rather than pulling
   the decomposition onto wrong notes.  That is much better news than a rise
   in ghosting would have been: a floor can be thresholded away downstream,
   whereas structurally wrong pitches cannot.
"""
from __future__ import annotations

import argparse, json, pathlib, sys, warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "backend/eval"))
warnings.filterwarnings("ignore")
import scipy.signal, scipy.signal.windows                       # noqa: E402
if not hasattr(scipy.signal, "hann"):
    scipy.signal.hann = scipy.signal.windows.hann

from octave_ghost_test import (                                  # noqa: E402
    SR, stft_mag, nmf_h, categorise, ground_truth, load_dictionary)
from cross_timbre_test import GM                                 # noqa: E402

RENDER_SR = 44100        # Demucs wants 44.1k stereo; analysis downsamples
SPLIT = 55               # below this goes to the bass part

# (label, program, bank, gain)
BASS   = (33, 0, 0.85)   # Electric Bass (finger)
GUITAR = (25, 0, 0.75)   # Steel Guitar
STRING = (48, 0, 0.45)   # String Ensemble, doubling the guitar part
DRUMS  = (0, 128, 0.60)  # Standard Kit


def render_part(notes, program, bank):
    import app.services.sf2_sampler as s
    if not notes:
        return None
    s._smpl_pcm, s._key_map = s._parse_sf2(GM.read_bytes(),
                                           program=program, bank=bank)
    s._loaded = True
    s._pitch_cache.clear()
    y = np.asarray(s.render_notes_sf2(notes, sr=RENDER_SR), dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    p = float(np.abs(y).max())
    return y / p if p > 1e-9 else y


def build_parts(midi_path, seconds):
    """Split a MAESTRO piece into band parts + a drum track on its beat grid."""
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(str(midi_path))

    low, high = [], []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            if n.start >= seconds:
                continue
            d = {"pitch": int(n.pitch), "onset": float(n.start),
                 "offset": float(min(n.end, seconds)),
                 "velocity": int(n.velocity)}
            (low if n.pitch < SPLIT else high).append(d)
    low.sort(key=lambda x: x["onset"]); high.sort(key=lambda x: x["onset"])

    # Drums on the piece's own beat grid: kick and snare alternating, hats on
    # every beat.  Crude, but the point is broadband transients in the right
    # places, not a convincing groove.
    beats = [b for b in pm.get_beats() if b < seconds]
    drums = []
    for i, t in enumerate(beats):
        drums.append({"pitch": 36 if i % 2 == 0 else 38, "onset": float(t),
                      "offset": float(t) + 0.25, "velocity": 100})
        drums.append({"pitch": 42, "onset": float(t),
                      "offset": float(t) + 0.12, "velocity": 70})
    return low, high, drums


def mix(*parts):
    parts = [(p, g) for p, g in parts if p is not None and p.size]
    if not parts:
        return np.zeros(RENDER_SR, dtype=np.float32)
    n = max(p.size for p, _ in parts)
    out = np.zeros(n, dtype=np.float32)
    for p, g in parts:
        out[:p.size] += p * g
    peak = float(np.abs(out).max())
    return out / peak * 0.95 if peak > 1e-9 else out


def demucs_strip_drums(y44):
    """Demucs the mixture and sum everything that is not the drum stem."""
    import torch
    from app.services.separation import _get_separator, _demucs_separate
    model = _get_separator()
    # _demucs_separate expects a (2, samples) torch tensor at 44.1k, not numpy.
    stereo = torch.from_numpy(np.stack([y44, y44], axis=0).astype(np.float32))
    stems = _demucs_separate(stereo, model)
    keep = [v for k, v in stems.items() if k != "drums"]
    return np.sum(keep, axis=0).mean(axis=0).astype(np.float32)


def analyse(y44, W, roll_path):
    import librosa
    y = librosa.resample(y44, orig_sr=RENDER_SR, target_sr=SR)
    V = stft_mag(y)
    H = nmf_h(V, W)
    return categorise(H, ground_truth(roll_path, V.shape[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pieces", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--no-demucs", action="store_true")
    args = ap.parse_args()

    man = json.loads((ROOT / "backend/eval/data/manifest.json").read_text())
    W = load_dictionary()
    print(f"dictionary: {W.shape}  (piano, unchanged)\n")

    pieces = []
    for p in man["pieces"]:
        midi = ROOT / p["midi_path"]
        roll = ROOT / p["midi_path"].replace(".midi", ".roll.100.npy")
        if midi.exists() and roll.exists():
            pieces.append((p, midi, roll))
        if len(pieces) >= args.pieces:
            break

    conds = {}
    for p, midi, roll in pieces:
        low, high, drums = build_parts(midi, args.seconds)
        b = render_part(low, *BASS[:2])
        g = render_part(high, *GUITAR[:2])
        st = render_part(high, *STRING[:2])
        dr = render_part(drums, DRUMS[0], DRUMS[1])

        pitched = mix((b, BASS[2]), (g, GUITAR[2]))
        with_dr = mix((b, BASS[2]), (g, GUITAR[2]), (dr, DRUMS[2]))
        with_pad = mix((b, BASS[2]), (g, GUITAR[2]), (st, STRING[2]),
                       (dr, DRUMS[2]))

        runs = [("bass+guitar (no drums)", pitched),
                ("bass+guitar+drums", with_dr),
                ("+ string pad + drums", with_pad)]
        if not args.no_demucs:
            runs += [("bass+guitar+drums, DEMUCS", demucs_strip_drums(with_dr)),
                     ("+ pad + drums, DEMUCS", demucs_strip_drums(with_pad))]

        for label, y in runs:
            conds.setdefault(label, []).append(analyse(y, W, roll))
        print(f"  done {p['composer'][:24]:<26}"
              f"({len(low)} bass / {len(high)} upper / {len(drums)} drum hits)",
              flush=True)

    print(f"\n{'condition':<30}{'correct':>9}{'decay':>8}{'octave':>8}"
          f"{'fifth':>7}{'other':>8}{'accounted':>11}")
    print("-" * 81)
    for label, rows in conds.items():
        m = np.array(rows).mean(axis=0) * 100
        print(f"{label:<30}{m[0]:8.1f}%{m[1]:7.1f}%{m[2]:7.1f}%{m[3]:6.1f}%"
              f"{m[4]:7.1f}%{m[0]+m[1]:10.1f}%")

    print("\nReference points from the earlier tests, same metric:")
    print("  matched piano 87.7%   cross-sampler piano 70.7%   nylon guitar 56.5%")
    print("Drums are absent from the ground truth, so every unit of energy they")
    print("attract lands in 'other'.  The DEMUCS rows minus their raw counterparts")
    print("is what separation is worth here, in points.")


if __name__ == "__main__":
    main()
