"""Same song, same arrangement rules, two different note sources — by ear.

The head-to-head on Slakh says the transcriber roughly doubles the NMF on
ground truth (note F1 0.468 against 0.247, chance 0.078).  That is a
measurement about NOTE DETECTION on synthetic full-band renders at ~27 notes
per second.  It does not say an arrangement of a real pop song will sound
better, and the only instrument that can answer that is an ear.

So both sources feed the SAME downstream stages and get rendered side by side.
Anything that differs is the note source, because nothing else does.

One asymmetry worth stating, because it is not a handicap but a real
difference in what each needs:

    NMF          gets harmonic.wav — Demucs output with drums removed.  It has
                 to, because a fixed dictionary must explain every scrap of
                 energy it is handed, and drum transients become notes.
    transcriber  gets audio.wav, the original mix, drums and all.  Slakh's
                 mixes contain drums while its targets exclude them, so
                 ignoring percussion is exactly what it was trained to do.
                 It needs no separation at all.

Also renders a stricter threshold.  The model emits 1.8x too many notes
(precision 0.366 against recall 0.671) because it was trained with a heavy
positive weight on onsets, and onset_thr is the same dial without retraining.
"""
from __future__ import annotations

import argparse, json, pathlib, shutil, sys, warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "backend/eval"))
warnings.filterwarnings("ignore")
import scipy.signal, scipy.signal.windows                       # noqa: E402
if not hasattr(scipy.signal, "hann"):
    scipy.signal.hann = scipy.signal.windows.hann

from eartest import PAGE, render, copy_audio, OUT, NAMES          # noqa: E402


def arrange(notes, beats, key):
    """The shipping reduction, applied identically to every source."""
    from app.services.transformation import (
        snap_to_beat_grid, enforce_melody_on_top, cap_polyphony,
        drop_non_scale, MAX_SOUNDING)
    from app.services.music_theory import _scale_pitch_classes
    steps = [("raw", len(notes))]
    notes = snap_to_beat_grid(notes, beats);      steps.append(("snap", len(notes)))
    notes = enforce_melody_on_top(notes);         steps.append(("melody-top", len(notes)))
    notes = cap_polyphony(notes, MAX_SOUNDING);   steps.append((f"cap{MAX_SOUNDING}", len(notes)))
    notes = drop_non_scale(notes, _scale_pitch_classes(*key))
    steps.append(("scale", len(notes)))
    return notes, steps


def key_of(notes):
    from app.services.nmf_voice_leading import key_from_pitch_classes
    w = np.zeros(12)
    for n in notes:
        w[int(n["pitch"]) % 12] += max(
            float(n["offset"]) - float(n["onset"]), 0.05) * float(n.get("velocity", 80))
    return key_from_pitch_classes(w) or (0, "major")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--name", default="ab_source")
    ap.add_argument("--strict-thr", type=float, default=0.7)
    ap.add_argument("--best-thr", type=float, default=0.90)
    args = ap.parse_args()

    from app.services import piano_projection as PP
    from app.services import ml_transcription as ML

    sess = ROOT / "backend/temp" / args.session
    meta = json.loads((sess / "components.json").read_text())
    hop, msr = float(meta.get("hop_length", 512)), float(meta.get("sr", 22050))
    beats = np.asarray(meta.get("beat_frames", []), float) * hop / msr
    OUT.mkdir(exist_ok=True)

    sources, dur = {}, 0.0

    # A — what ships today: two-voice decode off the drumless mix
    nmf = PP.project_from_stems(sess / "components")
    nk = (int(nmf[0]["key_root"]), str(nmf[0]["key_scale"])) if nmf and nmf[0].get(
        "key_root") is not None else key_of(nmf)
    todo = [("nmf", f"NMF two-voice (drumless)", nmf, nk)]

    # B — the transcriber, on the ORIGINAL mix, no separation
    if ML.available():
        for thr, tag in ((args.best_thr, "ml"), (args.strict_thr, "ml_loose")):
            ns = ML.transcribe(sess / "audio.wav", onset_thr=thr)
            todo.append((tag, f"transcriber (raw mix, thr {thr})", ns, key_of(ns)))
    else:
        print("no transcriber checkpoint — NMF only")

    for key_id, label, notes, key in todo:
        arranged, steps = arrange(notes, beats, key)
        print(f"{label:<38} " + " -> ".join(f"{n}={v}" for n, v in steps)
              + f"  | key {NAMES[key[0]]} {key[1]}")
        wav = OUT / f"{args.name}_{key_id}.wav"
        dur = max(dur, render(arranged, wav))
        sources[key_id] = {
            "label": label, "key": f"{NAMES[key[0]]} {key[1]}",
            "piano": wav.name, "audio": None,
            "notes": [[int(n["pitch"]), round(float(n["onset"]), 3),
                       round(float(n["offset"]), 3), int(n.get("velocity", 80))]
                      for n in arranged]}

    orig = OUT / f"{args.name}_original.mp3"
    if not orig.exists():
        shutil.copy(sess / "original.mp3", orig)
    sources["mix"] = {"label": "original mix", "key": "-", "piano": None,
                      "audio": orig.name, "notes": []}

    page = OUT / f"{args.name}.html"
    page.write_text(PAGE % {"title": f"{args.name}: note source A/B",
                            "data": json.dumps(sources), "dur": dur, "pps": 60},
                    encoding="utf-8")
    print(f"\nwrote {page}")


if __name__ == "__main__":
    main()
