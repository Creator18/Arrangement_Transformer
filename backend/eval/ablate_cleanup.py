"""
Which post-processing stage loses the notes on fast material?

Raw Onsets & Frames scores COnP 0.9486 / recall 0.9630 on the fastest pieces
in the corpus — as good as on the slowest.  The shipped pipeline scores
0.60-0.77 with recall 0.45-0.65 on those same pieces.  The model is not the
problem, so the loss is somewhere in our own tail.

This applies that tail one stage at a time to raw model output and scores
after each, so the cost of every stage is visible rather than inferred.
"""
import sys, os, json, warnings
import numpy as np

# scipy 1.8+ dropped scipy.signal.hann and librosa still calls it.  app/main.py
# installs this shim at import time; a standalone script bypasses main, so it
# has to be done here too or librosa.beat.beat_track raises.
import scipy.signal, scipy.signal.windows
if not hasattr(scipy.signal, "hann"):
    scipy.signal.hann = scipy.signal.windows.hann

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

import librosa, pretty_midi, mir_eval                        # noqa: E402
from app.services.transcription import (                     # noqa: E402
    _get_of_transcriptor, refine_notes, dedupe_octave_partials,
    _beat_adaptive_quantize,
)

SR = 16000
N_PIECES = 4


def truth_of(midi_path):
    pm = pretty_midi.PrettyMIDI(midi_path)
    ns = sorted((n for i in pm.instruments for n in i.notes), key=lambda n: n.start)
    return (np.array([[n.start, n.end] for n in ns]),
            np.array([librosa.midi_to_hz(n.pitch) for n in ns]),
            len(ns))


def conp(est, ref_iv, ref_hz):
    if not est:
        return 0.0, 0.0, 0.0
    iv = np.array([[n["onset"], n["offset"]] for n in est])
    hz = np.array([librosa.midi_to_hz(n["pitch"]) for n in est])
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_iv, ref_hz, iv, hz,
        onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None)
    return p, r, f


def main():
    man = json.load(open("backend/eval/data/manifest.json"))
    pieces = man["pieces"] if isinstance(man, dict) and "pieces" in man else man

    ranked = []
    for p in pieces:
        try:
            pm = pretty_midi.PrettyMIDI(p["midi_path"])
        except Exception:
            continue
        ns = [n for i in pm.instruments for n in i.notes]
        if len(ns) < 50 or not os.path.exists(p["audio_synth_path"]):
            continue
        on = np.array([n.start for n in ns])
        ranked.append((len(ns) / max(1e-6, on.max() - on.min()), p))
    ranked.sort(key=lambda x: -x[0])
    sel = ranked[:N_PIECES]

    inf = _get_of_transcriptor()
    acc = {}

    for dens, p in sel:
        audio, _ = librosa.load(p["audio_synth_path"], sr=SR, mono=True)
        out = inf.transcribe(audio, None)
        raw = [{"pitch": int(e["midi_note"]),
                "pitch_name": "", "onset": float(e["onset_time"]),
                "offset": float(e["offset_time"]),
                "velocity": max(1, min(127, int(e["velocity"])))}
               for e in out.get("est_note_events", [])
               if 21 <= int(e["midi_note"]) <= 108
               and float(e["offset_time"]) > float(e["onset_time"])]
        raw.sort(key=lambda n: (n["onset"], n["pitch"]))

        # Realistic beat grid for the quantiser (the real pipeline gets this
        # from madmom via separation; librosa is close enough to measure the
        # stage's COST, which is what this file is for).
        tempo, beat_f = librosa.beat.beat_track(y=audio, sr=SR, units="time")
        beats = np.asarray(beat_f, dtype=float)
        bpm = float(tempo) if np.ndim(tempo) == 0 else float(np.atleast_1d(tempo)[0])

        stages = [("raw O&F", list(raw))]
        s = refine_notes(list(raw), bpm)
        stages.append(("+ refine_notes", list(s)))
        s = dedupe_octave_partials(s)
        stages.append(("+ dedupe_octave", list(s)))
        if len(beats) >= 2:
            s = _beat_adaptive_quantize(s, beats, subdivisions=4)
        stages.append(("+ beat_quantize", list(s)))

        ref_iv, ref_hz, n_truth = truth_of(p["midi_path"])
        for name, notes in stages:
            pr, rc, f1 = conp(notes, ref_iv, ref_hz)
            acc.setdefault(name, []).append((f1, pr, rc, len(notes) / max(1, n_truth)))

    print(f"\n{N_PIECES} fastest pieces ({sel[-1][0]:.1f}-{sel[0][0]:.1f} notes/s)\n")
    print(f"{'stage':<18}{'COnP F1':>10}{'precision':>11}{'recall':>9}{'pred/truth':>12}{'dF1':>9}")
    print("-" * 69)
    prev = None
    for name in ["raw O&F", "+ refine_notes", "+ dedupe_octave", "+ beat_quantize"]:
        v = acc.get(name)
        if not v:
            continue
        f1, pr, rc, ratio = (float(np.mean([x[i] for x in v])) for i in range(4))
        d = "" if prev is None else f"{f1 - prev:+.4f}"
        print(f"{name:<18}{f1:>10.4f}{pr:>11.4f}{rc:>9.4f}{ratio:>12.2f}{d:>9}")
        prev = f1


if __name__ == "__main__":
    main()


# ── Results, 2026-08-09 ───────────────────────────────────────────────────────
#
# 4 fastest pieces (19.5-20.6 notes/s), stage by stage:
#
#   stage              COnP F1   precision   recall   pred/truth      dF1
#   raw O&F             0.9486      0.9348   0.9630         1.03
#   + refine_notes      0.8190      0.9359   0.7296         0.78   -0.1296
#   + dedupe_octave     0.8190      0.9359   0.7296         0.78   -0.0000
#   + beat_quantize     0.7973      0.9111   0.7103         0.78   -0.0217
#
# refine_notes is the whole loss, and inside it, one line: a hard 60 ms
# minimum duration.  Precision is untouched (0.9348 -> 0.9359) while recall
# collapses, so it was deleting true positives, not filtering noise.
#
# Threshold sweep that followed, over ALL 50 pieces:
#
#   floor    COnP F1   precision   recall
#    60 ms    0.8632      0.9281   0.8135
#    25 ms    0.9529      0.9335   0.9736
#
# +0.0896 F1 mean, 50/50 improved, none worse.  Shipped as
# MIN_NOTE_DURATION_S = 0.025 in transcription.py.
