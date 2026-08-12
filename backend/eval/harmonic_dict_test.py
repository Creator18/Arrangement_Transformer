"""
Test A — does a timbre-neutral dictionary beat the piano one, and what does it
cost in octave errors?

The piano dictionary matches on partial AMPLITUDES, which is the part of a note
that changes with the instrument.  Partial POSITIONS do not change: a C4 puts
energy at f0, 2f0, 3f0 whether it is sung, plucked or struck.  So matching on
positions alone should generalise across timbre, which is the whole problem
with pointing a piano dictionary at a voice.

The reason this might backfire
------------------------------
octave_ghost_test measured only 2.0% octave ghosting, and the mechanism mattered:
piano templates carry distinctive relative partial strengths, so a stack of
higher notes cannot cheaply imitate a lower one.  A generic comb has no such
signature — a comb at f0/2 contains every harmonic of f0 plus extra teeth in
between, so a sub-octave can always explain a note at least as well.  This is
not hypothetical: a harmonic-comb scorer written earlier today to check SF2
tuning reported instruments as two octaves flat until it was forced to require
real energy at the fundamental.

So this may simply trade a timbre problem for an octave problem, and the
ghosting column decides it, not the correct column.

Predictions, recorded before the run
------------------------------------
  * harmonic templates IMPROVE correct-energy on strings, organ and choir
  * harmonic templates LOSE on piano, and probably on guitar
  * octave ghosting RISES for harmonic templates on every voice

Limitation, stated up front: there is no real vocal with ground truth here.
GM Choir Aahs is sustained and formant-bearing but it is not a singer — no
vibrato, no portamento, and those are the two things that were argued to break
the piano dictionary in the first place.  This is a lower bound on difficulty,
the same way the matched-dictionary test was.

RESULT (2 tuning + 2 held-out pieces, 20 s each, rolloff 1.5, 2026-08-11)
------------------------------------------------------------------------
    voice              dictionary     correct  decay  octave  other  accounted
    grand piano (GM)   piano            31.0%  33.8%   5.4%   27.1%     64.8%
                       harmonic         32.6%  37.0%   5.1%   22.2%     69.6%
    nylon guitar       piano            29.8%  31.9%   7.0%   25.9%     61.7%
                       harmonic         32.0%  32.8%   5.8%   25.8%     64.7%
    string ensemble    piano            33.7%  14.2%  10.6%   36.4%     47.9%
                       harmonic         50.6%  14.3%   3.3%   29.0%     64.9%
    choir aahs         piano            14.3%  20.1%  12.9%   45.7%     34.4%
                       harmonic         23.8%  21.3%   7.3%   42.7%     45.1%
    church organ       piano            14.3%  27.1%  15.9%   40.6%     41.5%
                       harmonic         22.5%  34.7%  10.8%   28.6%     57.2%

A clean sweep: the harmonic dictionary wins on EVERY voice and EVERY column.
Sustained sources gain most, as predicted — strings +17.0 points, organ +15.7,
choir +10.7 — but piano and guitar improve too, which was predicted to go the
other way.

And octave ghosting FALLS everywhere, most where it was worst: strings
10.6% -> 3.3%, organ 15.9% -> 10.8%, choir 12.9% -> 7.3%.  That was predicted
to rise, and the prediction was wrong for a specific reason worth recording.

Why the octave argument was wrong
---------------------------------
The claim was that a comb at f0/2 contains every harmonic of f0, so a
sub-octave explains a note for free.  It contains them, but it also has teeth
at 1.5*f0, 2.5*f0 and so on, where the real signal has NOTHING.  KL divergence
penalises predicting energy that is not there, so those extra teeth are a
cost, not a freebie — and a steeper rolloff makes them more expensive, which is
why 1.5 beat 1.0 and 0.5 in tuning.

The mistake was over-generalising from a bug earlier the same day: the
hand-rolled comb scorer in the SF2 tuning check DID fail this way, because it
only summed energy at template positions and never penalised absent partials.
That is a property of that scorer, not of comb templates inside NMF.  A
generative reconstruction is scored both ways; a matching score is not.

Caveat that Test B should keep in mind: both dictionaries and all test audio
sit on the same equal-tempered grid, and these comb teeth are narrow (1.2%).
Real recordings drift, use A=442, and bend.  The piano dictionary shares the
assumption, so the comparison is fair, but the absolute numbers may not survive
detuned material.
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
from cross_timbre_test import GM, UPRIGHT, render_midi           # noqa: E402
from app.services.piano_projection import build_harmonic_dictionary  # noqa: E402

# (label, sf2, program, bank, sounds_st)
VOICES = [
    ("grand piano (GM)", GM, 0, 0, 0),
    ("nylon guitar", GM, 24, 0, 0),
    ("string ensemble", GM, 48, 0, 0),
    ("choir aahs", GM, 52, 0, 0),
    ("church organ", GM, 19, 0, -12),
]
ROLLOFFS = [0.5, 1.0, 1.5]


def shift_gt(GT, sounds_st):
    if not sounds_st:
        return GT
    GT = np.roll(GT, sounds_st, axis=0)
    if sounds_st < 0:
        GT[sounds_st:, :] = False
    else:
        GT[:sounds_st, :] = False
    return GT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", type=int, default=2)
    ap.add_argument("--test", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=20.0)
    args = ap.parse_args()

    man = json.loads((ROOT / "backend/eval/data/manifest.json").read_text())
    pieces = []
    for p in man["pieces"]:
        midi = ROOT / p["midi_path"]
        roll = ROOT / p["midi_path"].replace(".midi", ".roll.100.npy")
        if midi.exists() and roll.exists():
            pieces.append((p, midi, roll))
        if len(pieces) >= args.tune + args.test:
            break
    tune_set, test_set = pieces[: args.tune], pieces[args.tune:]

    dicts = {"piano": load_dictionary()}
    for a in ROLLOFFS:
        dicts[f"harmonic a={a}"] = build_harmonic_dictionary(a)
    print(f"dictionaries: {', '.join(dicts)}")
    print(f"tuning the rolloff on {len(tune_set)} pieces, reporting on "
          f"{len(test_set)} held-out, {args.seconds:.0f}s each\n")

    # Cache the decompositions once per (voice, piece, dictionary).
    results = {}
    for label, sf2, prog, bank, sounds in VOICES:
        for p, midi, roll in tune_set + test_set:
            y, _ = render_midi(midi, args.seconds, sf2, prog, bank)
            if y.ndim > 1:
                y = y.mean(axis=1)
            V = stft_mag(y)
            GT = shift_gt(ground_truth(roll, V.shape[1]), sounds)
            for dname, W in dicts.items():
                results[(label, p["id"], dname)] = categorise(nmf_h(V, W), GT)
        print(f"  decomposed {label}", flush=True)

    # Pick one rolloff for all voices, on the tuning pieces only, by accounted
    # energy minus octave ghosting -- ghosting is the failure this is testing
    # for, so it should not be free to grow.
    best_a, best_s = ROLLOFFS[0], -9e9
    for a in ROLLOFFS:
        dn = f"harmonic a={a}"
        vals = [results[(v[0], p["id"], dn)] for v in VOICES for p, _, _ in tune_set]
        s = float(np.mean([r[0] + r[1] - r[2] for r in vals]))
        print(f"  rolloff {a}: tuning score {s:.4f}")
        if s > best_s:
            best_a, best_s = a, s
    chosen = f"harmonic a={best_a}"
    print(f"\nchosen rolloff: {best_a}\n")

    print(f"{'voice':<20}{'dictionary':<16}{'correct':>9}{'decay':>8}"
          f"{'octave':>8}{'other':>8}{'accounted':>11}")
    print("-" * 80)
    for label, *_ in VOICES:
        for dname in ("piano", chosen):
            rows = [results[(label, p["id"], dname)] for p, _, _ in test_set]
            m = np.array(rows).mean(axis=0) * 100
            print(f"{label:<20}{dname:<16}{m[0]:8.1f}%{m[1]:7.1f}%{m[2]:7.1f}%"
                  f"{m[4]:7.1f}%{m[0]+m[1]:10.1f}%")
        print()

    print("The gate is the octave column, not the correct column.  A harmonic")
    print("dictionary that reads sustained sources better but hallucinates")
    print("octaves has traded one failure for a worse one, because wrong pitches")
    print("cannot be repaired downstream and a noise floor can.")


if __name__ == "__main__":
    main()
