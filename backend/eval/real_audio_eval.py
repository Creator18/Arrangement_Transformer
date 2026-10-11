"""The real-audio test: does a model trained only on rendered pianos hear a real one?

Every score so far was on audio rendered through sample libraries.  These are
REAL MAESTRO recordings — a Yamaha Disklavier in a hall, through microphones,
with real pedalling — from the test split (never trained on), with MIDI
captured from the instrument itself and aligned to the audio within ~3 ms.

Scored against the FULL ground truth: every MIDI note, no length filter.
(Earlier references were filtered by the decoder and missed 42% of notes.)

Compares, on identical recordings:
  spectral  old decoder   80 ms minimum + re-strike deletion (as first shipped)
  spectral  fixed decoder short notes and fast repeats kept, several thresholds
  ByteDance               the piano path's current transcriber
and, for durations, the corrector's effect on the fixed spectral output.

    python eval/real_audio_eval.py --dir E:/arrangement_transformer/maestro_real
"""
from __future__ import annotations

import argparse
import collections
import pathlib
import sys
import warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
import scipy.signal, scipy.signal.windows                       # noqa: E402
if not hasattr(scipy.signal, "hann"):
    scipy.signal.hann = scipy.signal.windows.hann

THRS = [0.5, 0.7, 0.85, 0.9, 0.93, 0.95]
BANDS = [("soft (<50)", 0, 50), ("medium (50-79)", 50, 80), ("loud (80+)", 80, 128)]


def midi_truth(path):
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(str(path))
    notes = [{"pitch": n.pitch, "onset": n.start, "offset": n.end, "velocity": n.velocity}
             for i in pm.instruments for n in i.notes if 21 <= n.pitch <= 108]
    notes.sort(key=lambda n: (n["onset"], n["pitch"]))
    last = {}
    for n in notes:                      # time since this key was last struck
        n["gap"] = n["onset"] - last.get(n["pitch"], -1e9)
        last[n["pitch"]] = n["onset"]
    return notes


def arrays(ns):
    iv = np.array([[n["onset"], max(n["offset"], n["onset"] + 1e-3)] for n in ns]) if ns else np.zeros((0, 2))
    hz = 440 * 2 ** ((np.array([n["pitch"] for n in ns], float) - 69) / 12) if ns else np.zeros(0)
    return iv, hz


def score(ref, est):
    """onset-only P/R/F1, offset-aware F1, and the set of matched ref indices."""
    from mir_eval.transcription import precision_recall_f1_overlap as prf, match_notes
    ri, rp = arrays(ref)
    ei, ep = arrays(est)
    if len(ei) == 0:
        return 0.0, 0.0, 0.0, 0.0, set(), []
    p, r, f, _ = prf(ri, rp, ei, ep, onset_tolerance=0.05, offset_ratio=None)
    fo = prf(ri, rp, ei, ep, onset_tolerance=0.05, offset_ratio=0.2)[2]
    pairs = match_notes(ri, rp, ei, ep, onset_tolerance=0.05, offset_ratio=None)
    return p, r, f, fo, {i for i, _ in pairs}, pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--no-bytedance", action="store_true")
    ap.add_argument("--ckpt", action="append", default=[],
                    help="label=path; repeat to compare models (default: the app's checkpoint)")
    args = ap.parse_args()
    import librosa
    from app.services import spectral_transcription as ST
    from app.ml.transcriber.decode import decode_notes
    from app.ml.corrector import correct_notes

    wavs = sorted(pathlib.Path(args.dir).glob("*.wav"))
    ckpts = [tuple(c.split("=", 1)) for c in args.ckpt] or [("spectral", str(ST.CKPT))]
    rows = collections.defaultdict(list)
    band_hit = collections.defaultdict(lambda: np.zeros(len(BANDS)))
    rep_hit = collections.Counter()
    band_tot, rep_tot, n_true = np.zeros(len(BANDS)), 0, 0
    dur = collections.defaultdict(list)

    for w in wavs:
        ref = midi_truth(w.with_suffix(".midi"))
        n_true += len(ref)
        band_tot += [sum(lo <= n["velocity"] < hi for n in ref) for _, lo, hi in BANDS]
        rep_tot += sum(n["gap"] < 0.15 for n in ref)
        systems = {}
        for label, path in ckpts:
            ST.CKPT = pathlib.Path(path); ST._cache.clear()
            on, fr, vl = ST.rolls(w)
            if label == ckpts[0][0]:
                systems[(f"{label} old-dec", 0.93)] = decode_notes(on, fr, vl, 0.93, 0.5)
            for t in THRS:
                systems[(label, t)] = decode_notes(on, fr, vl, t, 0.5, min_frames=1,
                                                   min_note_s=0.0, keep_restruck=True)
        if not args.no_bytedance:
            from app.services import transcription as TR
            systems[("ByteDance", None)] = TR._onsets_frames_notes(w)
        for key, est in systems.items():
            p, r, f, fo, hit, pairs = score(ref, est)
            rows[key].append((p, r, f, fo, len(est)))
            band_hit[key] += [sum(1 for i in hit if lo <= ref[i]["velocity"] < hi) for _, lo, hi in BANDS]
            rep_hit[key] += sum(1 for i in hit if ref[i]["gap"] < 0.15)
        # durations: fixed spectral at 0.93, before vs after the corrector
        est = systems[(ckpts[-1][0], 0.85)]
        y, sr = librosa.load(str(w), sr=22050, mono=True)
        beats = librosa.frames_to_time(librosa.beat.beat_track(y=y, sr=sr)[1], sr=sr)
        for name, ns in (("as decoded", est), ("after corrector", correct_notes(est, beats))):
            p, r, f, fo, hit, pairs = score(ref, ns)
            dur[name].append(fo)
            ri, _ = arrays(ref); ei, _ = arrays(ns)
            dur[name + " ratio"] += [np.log((ei[j, 1] - ei[j, 0]) / (ri[i, 1] - ri[i, 0])) for i, j in pairs]
        print(f"  {w.stem[:60]}: {len(ref)} notes", flush=True)

    print(f"\n{len(wavs)} real recordings, {n_true} MIDI notes (unfiltered)\n")
    print(f"{'system':<16}{'thr':>6}{'F1':>8}{'P':>7}{'R':>7}{'F1+off':>8}{'repeats R':>11}")
    for key in rows:
        a = np.array(rows[key])[:, :4].mean(0)
        thr = "" if key[1] is None else f"{key[1]:.2f}"
        print(f"{key[0]:<16}{thr:>6}{a[2]:>8.3f}{a[0]:>7.3f}{a[1]:>7.3f}{a[3]:>8.3f}"
              f"{rep_hit[key] / max(1, rep_tot):>11.3f}")
    print(f"\n(repeats = {rep_tot} notes re-struck within 150 ms of the same key)")
    print(f"\nrecall by dynamics:\n{'system':<16}{'thr':>6}" + "".join(f"{b[0]:>16}" for b in BANDS))
    for key in rows:
        if key[1] is not None and key[1] not in (0.5, 0.85, 0.93):
            continue
        thr = "" if key[1] is None else f"{key[1]:.2f}"
        print(f"{key[0]:<16}{thr:>6}" + "".join(f"{band_hit[key][i] / max(1, band_tot[i]):>16.3f}"
                                               for i in range(len(BANDS))))
    print(f"   band sizes: " + ", ".join(f"{b[0]} {int(band_tot[i])}" for i, b in enumerate(BANDS)))
    print("\ncorrector on real audio (newest model, thr 0.85):")
    for name in ("as decoded", "after corrector"):
        r = np.array(dur[name + " ratio"])
        print(f"   {name:<16} F1+offsets {np.mean(dur[name]):.3f}   median length vs true "
              f"{np.exp(np.median(r)):.2f}x   within 25% {np.mean(np.abs(r) < np.log(1.25)) * 100:.1f}%")


if __name__ == "__main__":
    main()
