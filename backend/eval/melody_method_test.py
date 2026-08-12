"""
Test B — for the MELODY specifically, which method wins?

Two hypotheses have been tangled together and need separating:

  (1) timbre-neutral harmonic templates beat piano templates   [Test A: yes]
  (2) a monophonic continuous-F0 tracker beats frame-independent NMF entirely

If (2) holds, the dictionary improvement is the wrong fight for the melody and
belongs only in the accompaniment path.  So pYIN goes in the ring against both
dictionaries, on the same audio, scored the same way.

The stem
--------
The top voice of each MAESTRO piece — the highest sounding note at each instant
— which is monophonic by construction and is what a listener follows.  Rendered
with GM Choir Aahs, the closest thing available to a sung line: sustained,
formant-bearing, and the exact case the piano dictionary handles worst.

Vibrato is simulated, not assumed away
--------------------------------------
The argument for pYIN was that a voice moves CONTINUOUSLY — vibrato and
portamento — and that frame-independent NMF reads that as a burst of separate
notes while a pitch tracker reads it as one.  A choir sample with no vibrato
would quietly skip the part of the claim that matters, so the stem is also
tested with vibrato applied.

Because the stem is monophonic, modulating the pitch of the WHOLE signal is
exactly vibrato on the single sounding note — no per-note synthesis needed.
It is done by resampling along a warped time axis at +-50 cents, 5.5 Hz, which
is a realistic singer's depth and rate.

RESULT (4 pieces, 20 s each, 2026-08-11) — hypothesis (2) is rejected
---------------------------------------------------------------------
    condition  method          frame acc  pitch-class  octave err
    plain      piano dict          33.1%        63.0%       29.9%
    plain      harmonic dict       60.2%        68.6%        8.4%
    plain      pYIN                48.7%        58.7%       10.0%
    vibrato    piano dict          25.5%        59.7%       34.2%
    vibrato    harmonic dict       54.5%        64.5%       10.0%
    vibrato    pYIN                46.7%        53.7%        7.0%

The harmonic dictionary beats pYIN outright, plain or with vibrato, and beats
it under vibrato by more than pYIN manages without any.  So the melody does
NOT obviously need a pitch tracker, and the argument made earlier — that a
monophonic continuous-F0 model must win on a voice — is not supported.

The piano dictionary's melody failure is almost entirely OCTAVE ERROR: its
pitch-class accuracy is 63.0%, nearly as good as the harmonic dictionary's
68.6%, but its exact-pitch accuracy is half.  It was hearing roughly the right
note in the wrong register, which is a far more specific diagnosis than
"sustained sources project badly".

Vibrato behaves as argued but the effect is small: it costs pYIN 2.0 points,
the harmonic dictionary 5.7 and the piano dictionary 7.6.  Directionally the
pitch tracker IS the most robust to continuous pitch movement.  It is simply
not enough to overcome starting 11 points behind.

Why the pYIN number is probably pessimistic
-------------------------------------------
The reference here is the SKYLINE of dense classical piano writing — 159 notes
in 20 s for the Debussy — which is not a singable line.  pYIN is built for
sustained monophonic material and needs stable pitch across time to commit;
a top voice that changes eight times a second is close to its worst case and a
real vocal is close to its best.  Treat this as evidence that the harmonic
dictionary is good, not as evidence that pYIN is bad.  Settling that properly
needs a real vocal line with ground truth, which this repo does not have.

Two earlier versions of this test were invalid and both were caught by a
number being implausible rather than merely bad:
  * top_voice compared each note only against the LAST one appended, so a long
    high note suppressed everything under it — 15 "melody" notes in 20 s.
  * pYIN was scored through _pyin_to_notes, which merges frames within TWO
    semitones and so welds stepwise melodies into single notes.  Its raw F0 is
    within half a semitone on 63% (choir) and 89% (guitar) of frames; the 0.045
    it first scored was the segmenter, not the tracker.
Note-level COnP was measuring segmentation rather than method, which is why
this settled on frame accuracy — no method needs a segmenter to be scored.
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

from octave_ghost_test import SR, stft_mag, nmf_h, load_dictionary  # noqa: E402
from note_extraction_test import score                             # noqa: E402
from cross_timbre_test import GM                                   # noqa: E402
from mixture_test import render_part, RENDER_SR                    # noqa: E402
from app.services.piano_projection import (                        # noqa: E402
    build_harmonic_dictionary, extract_notes, THRESH_Q)

CHOIR = (52, 0)
VIB_CENTS = 50.0
VIB_HZ = 5.5


GRID = 200.0          # frames per second for the skyline


def top_voice(midi_path, seconds, min_dur=0.05):
    """The highest sounding note at each instant — a true skyline.

    The first version of this compared each note only against the LAST one
    appended, which is not the same as against everything currently sounding:
    a single long high note suppressed every note that started under it, and
    the "melody" came out as 15 notes in 20 seconds for one piece and 234 for
    another.  Every method then scored against a broken reference, pYIN
    included, which is what made the numbers implausible rather than merely
    bad.

    Sampling the timeline and taking the maximum sounding pitch per frame has
    no such ordering assumption.
    """
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    n_fr = int(seconds * GRID)
    top = np.zeros(n_fr, dtype=np.int16)
    vel = np.zeros(n_fr, dtype=np.int16)
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            if n.start >= seconds:
                continue
            a = int(n.start * GRID)
            b = min(int(min(n.end, seconds) * GRID), n_fr)
            if b <= a:
                continue
            m = top[a:b] < n.pitch
            seg_v = vel[a:b]; seg_v[m] = n.velocity
            seg_t = top[a:b]; seg_t[m] = n.pitch

    out, i = [], 0
    while i < n_fr:
        if top[i] == 0:
            i += 1
            continue
        j = i
        while j < n_fr and top[j] == top[i]:
            j += 1
        if (j - i) / GRID >= min_dur:
            out.append([i / GRID, j / GRID, int(top[i]), int(vel[i]) or 80])
        i = j
    return out


def vibrato(y, sr, cents=VIB_CENTS, rate=VIB_HZ):
    """Pitch-modulate a monophonic signal by resampling along a warped clock."""
    n = len(y)
    t = np.arange(n) / sr
    depth = 2.0 ** (cents / 1200.0) - 1.0
    # instantaneous rate -> cumulative read position
    inst = 1.0 + depth * np.sin(2 * np.pi * rate * t)
    pos = np.cumsum(inst)
    pos = pos / pos[-1] * (n - 1)
    return np.interp(np.arange(n), np.arange(n), np.interp(pos, np.arange(n), y)
                     ).astype(np.float32)


def pyin_notes(y, sr, hop=512, min_dur=0.05, bridge=0.08):
    """pYIN F0 -> notes, segmented for MELODY rather than bass.

    The shipped `_pyin_to_notes` merges consecutive frames within TWO SEMITONES
    into one note.  That is right for a bass line, which moves in leaps and
    holds; on a melody moving by step it welds adjacent scale degrees into one
    long note, which is why pYIN first scored 0.044 here — an implausible
    result for the one method that should own a clean monophonic line.

    Segmenting on the semitone the F0 actually rounds to gives runs directly
    comparable to the skyline reference, which is built the same way.
    """
    import librosa
    f0, voiced, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
        sr=sr, hop_length=hop, frame_length=4096, fill_na=None)
    fps = sr / hop
    n = len(f0)
    pitch = np.zeros(n, dtype=np.int16)
    for i in range(n):
        if voiced[i] and f0[i] and np.isfinite(f0[i]):
            pitch[i] = int(round(69 + 12 * np.log2(f0[i] / 440.0)))

    # Bridge short unvoiced gaps so a consonant or a breath does not split one
    # sung note into two.
    gap = int(bridge * fps)
    for i in range(1, n - 1):
        if pitch[i] == 0:
            j = i
            while j < n and pitch[j] == 0:
                j += 1
            if j < n and j - i <= gap and pitch[i - 1] == pitch[j]:
                pitch[i:j] = pitch[i - 1]

    out, i = [], 0
    while i < n:
        if pitch[i] == 0:
            i += 1
            continue
        j = i
        while j < n and pitch[j] == pitch[i]:
            j += 1
        if (j - i) / fps >= min_dur:
            out.append((i / fps, j / fps, int(pitch[i])))
        i = j
    return out


def nmf_notes(y, W):
    return extract_notes(nmf_h(stft_mag(y), W), THRESH_Q)


# ── frame-level pitch accuracy ───────────────────────────────────────────────
# Note-level COnP turned out to measure the SEGMENTER more than the method: the
# first pYIN score here was 0.045, yet its raw F0 is within half a semitone on
# 63-89% of frames.  Segmenting a pitch track into note events is a separate
# problem with its own free parameters, and tuning three of them fairly is a
# bigger job than the question deserves.
#
# Frame accuracy asks only "at this instant, does the method know the pitch",
# which is exactly where a dictionary and a pitch tracker genuinely differ, and
# it needs no segmentation from anybody.

def ref_frames(line, n_fr, fps):
    ref = np.zeros(n_fr, dtype=np.int16)
    for a, b, q, _v in line:
        ref[int(a * fps):min(int(b * fps), n_fr)] = q
    return ref


def frames_nmf(y, W, fps_out, n_fr):
    H = nmf_h(stft_mag(y), W)
    from app.services.piano_projection import LOW, FPS
    top = np.argmax(H, axis=0) + LOW
    top[H.max(axis=0) <= 0] = 0
    idx = np.clip((np.arange(n_fr) / fps_out * FPS).astype(int), 0, len(top) - 1)
    return top[idx]


def frames_pyin(y, sr, fps_out, n_fr, hop=512):
    import librosa
    f0, voiced, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
        sr=sr, hop_length=hop, frame_length=4096, fill_na=None)
    fps = sr / hop
    out = np.zeros(len(f0), dtype=np.int16)
    ok = voiced & np.isfinite(f0) & (f0 > 0)
    out[ok] = np.round(69 + 12 * np.log2(f0[ok] / 440.0)).astype(np.int16)
    idx = np.clip((np.arange(n_fr) / fps_out * fps).astype(int), 0, len(out) - 1)
    return out[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pieces", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=20.0)
    args = ap.parse_args()

    import librosa
    man = json.loads((ROOT / "backend/eval/data/manifest.json").read_text())
    W_piano = load_dictionary()
    W_harm = build_harmonic_dictionary(1.5)

    pieces = []
    for p in man["pieces"]:
        midi = ROOT / p["midi_path"]
        if midi.exists():
            pieces.append((p, midi))
        if len(pieces) >= args.pieces:
            break

    rows = {}
    for p, midi in pieces:
        line = top_voice(midi, args.seconds)
        ref = [(a, b, q) for a, b, q, _ in line]
        y44 = render_part([{"pitch": q, "onset": a, "offset": b, "velocity": v}
                           for a, b, q, v in line], *CHOIR)
        if y44 is None:
            continue
        y = librosa.resample(y44, orig_sr=RENDER_SR, target_sr=SR)

        n_fr = int(args.seconds * GRID)
        R = ref_frames(line, n_fr, GRID)
        sounding = R > 0
        for cond, sig in (("plain", y), ("vibrato", vibrato(y, SR))):
            ests = {
                "piano dict": frames_nmf(sig, W_piano, GRID, n_fr),
                "harmonic dict": frames_nmf(sig, W_harm, GRID, n_fr),
                "pYIN": frames_pyin(sig, SR, GRID, n_fr),
            }
            for name, E in ests.items():
                exact = float(np.mean(E[sounding] == R[sounding]))
                # Octave-forgiving: right pitch class, any octave.  The gap
                # between this and `exact` is octave error and nothing else.
                pc = float(np.mean((E[sounding] % 12) == (R[sounding] % 12)))
                rows.setdefault((cond, name), []).append((exact, pc))
        print(f"  {p['composer'][:22]:<24} melody {len(ref)} notes", flush=True)

    print(f"\n{'condition':<10}{'method':<16}{'frame acc':>11}"
          f"{'pitch-class':>13}{'octave err':>12}")
    print("-" * 62)
    for cond in ("plain", "vibrato"):
        for name in ("piano dict", "harmonic dict", "pYIN"):
            m = np.array(rows[(cond, name)]).mean(axis=0) * 100
            print(f"{cond:<10}{name:<16}{m[0]:>10.1f}%{m[1]:>12.1f}%"
                  f"{m[1]-m[0]:>11.1f}%")
        print()

    print("frame acc = correct MIDI pitch at that instant; pitch-class ignores")
    print("octave, so the gap between them is octave error alone.  The vibrato")
    print("block is what decides whether the melody needs a pitch tracker: if")
    print("NMF collapses there and pYIN holds, the dictionary is the wrong lever.")


if __name__ == "__main__":
    main()
