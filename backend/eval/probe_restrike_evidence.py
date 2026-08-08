"""
probe_restrike_evidence.py — does the model SEE re-strikes and we discard
them, or does it genuinely not see them?

We emit only 46% of the re-strikes present in real performances.  A
hand-built spectral detector failed badly (eval run 8: 45/50 pieces
worse), because raw STFT bins are contested — harmonic collision,
broadband hammer noise, sympathetic resonance.

But the transcriber already produces, every 10 ms and for each of the 88
pitches, its OWN disentangled estimates:
    reg_onset_output   did this pitch begin now
    frame_output       is this pitch sounding now
These are the model's opinion after it has done the source separation
that defeated my spectral approach.  We currently threshold them at
onset >= 0.3 and throw the rest away.

This probe takes ground-truth MIDI, finds the re-strikes we MISSED, and
reads what the model's raw outputs were doing at exactly those moments.

Three possible answers, each implying different work:
    high onset prob     -> the evidence is there; this is a threshold /
                           post-processing fix, cheap
    middling            -> evidence is weak but present; a small trained
                           head on these curves could recover it
    at the noise floor  -> the model genuinely cannot see it; nothing
                           downstream will help and the direction is dead

Control: the same statistic at re-strikes we DID capture, and at random
sustained (non-restrike) moments, so "high" and "low" mean something.

    python backend/eval/probe_restrike_evidence.py --pieces 4
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "backend"))

import scipy.signal, scipy.signal.windows
if not hasattr(scipy.signal, "hann"):
    scipy.signal.hann = scipy.signal.windows.hann

FPS = 100          # model frame rate
BEGIN_NOTE = 21    # MIDI pitch of output column 0


def _truth_restrikes(midi_path: Path, max_gap: float = 0.35
                     ) -> List[Tuple[float, int]]:
    """(time, pitch) for every note that re-strikes a still-recent pitch."""
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    by_pitch: Dict[int, List[Tuple[float, float]]] = {}
    for inst in pm.instruments:
        for n in inst.notes:
            by_pitch.setdefault(int(n.pitch), []).append((float(n.start), float(n.end)))
    out = []
    for p, v in by_pitch.items():
        v.sort()
        for a, b in zip(v, v[1:]):
            if b[0] - a[1] <= max_gap:
                out.append((b[0], p))
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pieces", type=int, default=4)
    args = ap.parse_args()

    from app.services.transcription import _get_of_transcriptor
    import librosa
    from piano_transcription_inference import sample_rate as OF_SR

    man = json.load(open(REPO / "backend/eval/data/manifest.json", encoding="utf-8"))
    transcriptor = _get_of_transcriptor()

    hit_on, miss_on, ctrl_on = [], [], []
    hit_fr, miss_fr, ctrl_fr = [], [], []

    for piece in man["pieces"][: args.pieces]:
        wav = REPO / piece["audio_synth_path"]
        run = REPO / "backend/eval/runs" / piece["id"] / "transcription.json"
        if not wav.exists() or not run.exists():
            continue

        audio, _ = librosa.load(str(wav), sr=OF_SR, mono=True)
        res = transcriptor.transcribe(audio.astype(np.float32), midi_path=None)
        od = res["output_dict"]
        onset = np.asarray(od["reg_onset_output"])   # (frames, 88)
        frame = np.asarray(od["frame_output"])

        ours = json.load(open(run, encoding="utf-8"))["notes"]
        truth = _truth_restrikes(REPO / piece["midi_path"])

        # A re-strike is "captured" if we emitted a note of that pitch within 60 ms
        by_pitch: Dict[int, List[float]] = {}
        for n in ours:
            by_pitch.setdefault(int(n["pitch"]), []).append(float(n["onset"]))

        for t, p in truth:
            col = p - BEGIN_NOTE
            fi = int(round(t * FPS))
            if col < 0 or col >= onset.shape[1] or fi < 2 or fi >= onset.shape[0] - 2:
                continue
            # peak in a +-30 ms window, since our timing is not exact
            o = float(onset[fi - 3:fi + 4, col].max())
            f = float(frame[fi - 3:fi + 4, col].max())
            captured = any(abs(x - t) <= 0.06 for x in by_pitch.get(p, []))
            (hit_on if captured else miss_on).append(o)
            (hit_fr if captured else miss_fr).append(f)

        # Control: sustained moments with no re-strike anywhere near
        rng = np.random.RandomState(0)
        restrike_frames = {int(round(t * FPS)) for t, _ in truth}
        for _ in range(len(truth)):
            fi = int(rng.randint(5, max(6, onset.shape[0] - 5)))
            col = int(rng.randint(0, onset.shape[1]))
            if any(abs(fi - r) < 20 for r in restrike_frames):
                continue
            if frame[fi, col] < 0.5:        # only count while the pitch sounds
                continue
            ctrl_on.append(float(onset[fi - 3:fi + 4, col].max()))
            ctrl_fr.append(float(frame[fi - 3:fi + 4, col].max()))

    def rep(name, v):
        if not v:
            print(f"  {name:<34} (no samples)")
            return
        a = np.asarray(v)
        print(f"  {name:<34} n={len(a):>6}  median {np.median(a):.3f}  "
              f"mean {a.mean():.3f}  >=0.3: {100*np.mean(a>=0.3):5.1f}%  "
              f">=0.15: {100*np.mean(a>=0.15):5.1f}%")

    print("\nreg_onset_output at ground-truth re-strike moments")
    rep("re-strikes we CAPTURED", hit_on)
    rep("re-strikes we MISSED", miss_on)
    rep("control: sustained, no re-strike", ctrl_on)
    print("\nframe_output at the same moments (is the pitch sounding)")
    rep("re-strikes we CAPTURED", hit_fr)
    rep("re-strikes we MISSED", miss_fr)
    rep("control: sustained, no re-strike", ctrl_fr)

    if miss_on and ctrl_on:
        m, c = np.asarray(miss_on), np.asarray(ctrl_on)
        print(f"\nVERDICT")
        print(f"  missed-restrike onset prob vs control: "
              f"{np.median(m):.3f} vs {np.median(c):.3f}")
        if np.median(m) > 2 * max(np.median(c), 1e-6):
            print("  -> evidence IS present and being discarded by the 0.3 "
                  "threshold; a cheap post-processing fix is plausible")
        elif np.median(m) > np.median(c) * 1.2:
            print("  -> evidence is weak but real; a small trained head on "
                  "these curves could recover it")
        else:
            print("  -> the model does NOT see these re-strikes; nothing "
                  "downstream of it will help")


if __name__ == "__main__":
    main()
