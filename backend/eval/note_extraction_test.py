"""
From 88 activation curves to actual notes.

The three previous tests all measured where ENERGY goes.  That was the right
question while the premise was in doubt, but energy is not a deliverable — the
pipeline ships note events, with onsets, offsets and pitches.  This is the step
that turns a decomposition into an arrangement, and the first one whose output
could be played.

Method
------
Per pitch, the activation row is smoothed, thresholded into runs, and each run
is split at internal onset peaks so that repeated notes on one key do not merge
into a single long one.  Runs shorter than MIN_DUR are dropped.  Scoring is
mir_eval COnP (onset within +-50 ms and correct pitch) and COnPOff (which also
requires the offset), the same pair the transcription evaluation already uses,
so the numbers sit on the same scale as everything else in eval/.

Thresholds are TUNED AND TESTED ON DIFFERENT PIECES
---------------------------------------------------
Peak-picking has free parameters, and fitting them on the same audio you then
report is how a method looks better than it is.  The first `--tune` pieces
choose the threshold; the remaining ones are scored with it and never
influence it.  Both numbers are printed, because the gap between them is
itself informative.

What a fair comparison looks like
---------------------------------
Onsets & Frames scores note F1 0.9677 on real piano and this will not come
close.  That is not the comparison that matters: O&F cannot be pointed at a
guitar or a mixture at all, and those are the rows this exists to fill.  The
matched-piano row is here as a CEILING — the best this machinery can do when
the input is exactly what the dictionary was built from — so the cost of every
step away from that is legible.

RESULT (2 tuning + 4 held-out pieces, 30 s each, 2026-08-11)
------------------------------------------------------------
    condition             q*   COnP   prec    rec   COnPOff  notes/ref
    matched sfizz piano  0.95  0.575  0.918  0.421    0.100     0.46x
    grand piano (GM)     0.95  0.481  0.774  0.356    0.100     0.48x
    nylon guitar         0.95  0.384  0.594  0.287    0.056     0.49x
    mixture + demucs     0.90  0.375  0.439  0.329    0.096     0.75x

This is now the weakest link in the chain, and the shape of the failure is
specific: PRECISION IS HIGH AND RECALL IS NOT.  On matched piano the extractor
is right about 92% of the notes it emits and finds only 42% of the notes that
are there.

Widening the threshold sweep did not fix it.  The first run used 0.80-0.99 and
every condition sat at the bottom of the range, which normally means the range
is truncated; extended to 0.20-0.99 the optimum stayed at 0.95.  So recall is
not being held back by selectivity — dropping the threshold adds false
positives faster than it recovers real notes.  The missing notes are ones
whose activation never rises far above the floor at all: soft inner voices
inside dense chords, absorbed by the louder notes sharing their partials.

Offsets are bad everywhere (COnPOff ~0.10), which is consistent with the rest
of the pipeline — offsets already run 2.10x long through Onsets & Frames.

The honest reading: the PROJECTION is not the bottleneck, the naive extractor
is.  Energy allocation reached 63% on a full mixture, and this converts that
into 0.375.  Threshold-and-run peak-picking is the crudest thing that could
work, and everything it ignores is available — the beat grid madmom already
produces, spectral flux, the harmonic relationship between simultaneously
active rows.  That is where the headroom is, not in a better dictionary.
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
    SR, FPS, LOW, stft_mag, nmf_h, load_dictionary)

# Imported from the service, not reimplemented here — this file's numbers are
# only meaningful if they describe the extractor that actually runs.
from app.services.piano_projection import (                      # noqa: E402
    extract_notes, MIN_DUR, SMOOTH)


def ref_notes(midi_path, seconds):
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    out = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            if n.start < seconds:
                out.append((float(n.start), float(min(n.end, seconds)),
                            int(n.pitch)))
    return sorted(out)


def score(ref, est):
    import mir_eval
    if not est or not ref:
        return 0.0, 0.0, 0.0, 0.0
    def split(ns):
        iv = np.array([[a, max(b, a + 1e-3)] for a, b, _ in ns])
        hz = np.array([440.0 * 2 ** ((p - 69) / 12) for _, _, p in ns])
        return iv, hz
    ri, rh = split(ref)
    ei, eh = split(est)
    p, r, conp, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ri, rh, ei, eh, onset_tolerance=0.05, pitch_tolerance=50.0,
        offset_ratio=None)
    _, _, conpoff, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ri, rh, ei, eh, onset_tolerance=0.05, pitch_tolerance=50.0,
        offset_ratio=0.2)
    return conp, conpoff, p, r


def activations(y, W):
    if y.ndim > 1:
        y = y.mean(axis=1)
    return nmf_h(stft_mag(y), W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", type=int, default=2)
    ap.add_argument("--test", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--no-demucs", action="store_true")
    args = ap.parse_args()

    import librosa
    from cross_timbre_test import GM, render_midi
    from mixture_test import (build_parts, render_part, mix, demucs_strip_drums,
                              BASS, GUITAR, DRUMS, RENDER_SR)

    man = json.loads((ROOT / "backend/eval/data/manifest.json").read_text())
    W = load_dictionary()

    pieces = []
    for p in man["pieces"]:
        midi = ROOT / p["midi_path"]
        wav = ROOT / p["audio_synth_path"]
        if midi.exists() and wav.exists():
            pieces.append((p, midi, wav))
        if len(pieces) >= args.tune + args.test:
            break
    tune_set, test_set = pieces[: args.tune], pieces[args.tune:]
    print(f"tuning on {len(tune_set)} pieces, reporting on {len(test_set)} "
          f"held-out pieces, {args.seconds:.0f}s each\n")

    def make(cond, midi, wav):
        if cond == "matched sfizz piano":
            y, _ = librosa.load(str(wav), sr=SR, mono=True, duration=args.seconds)
            return y
        if cond == "grand piano (GM)":
            y, _ = render_midi(midi, args.seconds, GM, 0, 0)
            return y
        if cond == "nylon guitar":
            y, _ = render_midi(midi, args.seconds, GM, 24, 0)
            return y
        # mixture + demucs
        low, high, drums = build_parts(midi, args.seconds)
        m = mix((render_part(low, *BASS[:2]), BASS[2]),
                (render_part(high, *GUITAR[:2]), GUITAR[2]),
                (render_part(drums, DRUMS[0], DRUMS[1]), DRUMS[2]))
        m = demucs_strip_drums(m)
        return librosa.resample(m, orig_sr=RENDER_SR, target_sr=SR)

    conds = ["matched sfizz piano", "grand piano (GM)", "nylon guitar"]
    if not args.no_demucs:
        conds.append("mixture + demucs")

    # The first sweep ran 0.80-0.99 and every condition landed at or near the
    # BOTTOM of it, emitting 0.35-0.75x as many notes as the reference.  That
    # is a truncated range, not an optimum: the extractor was recall-bound and
    # never allowed to be less selective.
    QS = [0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95, 0.99]
    print(f"{'condition':<24}{'q*':>6}{'tune F1':>10}{'COnP':>9}{'prec':>8}"
          f"{'rec':>8}{'COnPOff':>10}{'notes/ref':>12}")
    print("-" * 82)

    for cond in conds:
        cache = {}
        for p, midi, wav in tune_set + test_set:
            cache[p["id"]] = (activations(make(cond, midi, wav), W),
                              ref_notes(midi, args.seconds))

        best_q, best_f1 = QS[0], -1.0
        for q in QS:
            f1s = [score(cache[p["id"]][1],
                         extract_notes(cache[p["id"]][0], q))[0]
                   for p, _, _ in tune_set]
            if np.mean(f1s) > best_f1:
                best_q, best_f1 = q, float(np.mean(f1s))

        conps, offs, ratios, precs, recs = [], [], [], [], []
        for p, _, _ in test_set:
            H, ref = cache[p["id"]]
            est = extract_notes(H, best_q)
            a, b, pr, rc = score(ref, est)
            conps.append(a); offs.append(b); precs.append(pr); recs.append(rc)
            ratios.append(len(est) / max(len(ref), 1))
        print(f"{cond:<24}{best_q:>6.2f}{best_f1:>10.3f}{np.mean(conps):>9.3f}"
              f"{np.mean(precs):>8.3f}{np.mean(recs):>8.3f}"
              f"{np.mean(offs):>10.3f}{np.mean(ratios):>11.2f}x", flush=True)

    print("\nCOnP = onset within +-50 ms and correct pitch; COnPOff also requires")
    print("the offset.  Onsets & Frames scores 0.9677 COnP on real piano, but it")
    print("cannot be pointed at the guitar or mixture rows at all -- those are the")
    print("rows this exists to fill.  'notes/ref' above 1 means over-segmentation.")


if __name__ == "__main__":
    main()
