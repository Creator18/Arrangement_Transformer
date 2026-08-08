"""
restrike.py — recover repeated-note articulation.

The problem, reported by ear and then measured: when a pitch is struck
again while the previous instance is still ringing, the transcriber
usually misses the second attack and emits one long held note instead of
two.  Across 25 eval pieces the ground truth contains 26,560 such
re-strikes and we produce 12,275 — we lose more than half of them.

Why a general transcriber struggles here: at a re-attack the harmonics of
that pitch are ALREADY sounding, so the new onset is a small transient
against a large background.  In a full-mix spectrogram that evidence is
easily swamped.

Why a targeted detector can do better: we are not asking the general
question.  For every note we already detected we know the exact pitch, so
we know exactly which frequency bins to watch and exactly when.  Energy
being re-injected into one pitch's own harmonic series, part-way through
that note, is a re-strike.  That is a far narrower question than
polyphonic transcription, and it is answerable with plain spectral flux.

Method, per detected note:
    1. take the STFT frames spanning the note
    2. sum magnitude across that pitch's first few harmonic bins only
    3. look for sharp positive jumps in that curve (rising edges), which
       is energy appearing where it was already decaying
    4. each accepted jump splits the note in two at that point

Deliberately conservative: piano tone decays monotonically after the
attack, so a genuine rise is unambiguous.  Requiring a real rise rather
than merely "not decaying" keeps vibrato, beating between detuned
strings, and pedal resonance from producing phantom re-strikes.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


# STFT settings.  1024 at 22.05 kHz gives ~46 ms windows / ~11.6 ms hops —
# short enough to localise an attack, long enough to resolve low harmonics.
N_FFT      = 1024
HOP        = 256

# A re-strike must raise the pitch's own energy by at least this ratio over
# the local minimum preceding it.  Piano decay means any real rise is a new
# excitation; the margin is for noise, not for ambiguity.
RISE_RATIO = 1.6

# Minimum musical spacing.  Below ~60 ms a "repeat" is more likely one
# attack smeared across frames than two distinct strikes, and the ground
# truth shows only 1.1% of real re-strikes closer than 50 ms.
MIN_GAP_S  = 0.06

# Harmonics to watch.  The fundamental alone is unreliable in the bass (weak
# or missing); the first few partials together are a robust signature.
N_HARMONICS = 4

# Ignore this much time after every attack — the original onset and each
# accepted re-strike.  A piano attack is itself a steep rise, so without
# this the detector fires on the strike it just accounted for and splits a
# note a few frames after its own beginning.  (Observed directly: a clean
# four-strike test produced spurious splits at 0.07 s and at 1.29 s, both
# 70 ms after a real attack.)
ATTACK_SKIP_S = 0.09

# How many frames after a candidate rise must still be elevated for it to
# count as a strike rather than a passing transient.  ~6 frames at hop 256 /
# 22.05 kHz is roughly 70 ms — long enough that a neighbouring key's hammer
# noise has decayed, short enough to sit inside the shortest real repeat.
SUSTAIN_FRAMES = 6

# Silence inserted before a recovered re-strike, so the two notes do not read
# as one continuous tone.  A damper needs roughly this long to fall.
RELEASE_S = 0.025


def _harmonic_bins(pitch: int, sr: int, n_fft: int,
                   n_harm: int = N_HARMONICS) -> List[int]:
    f0 = 440.0 * (2.0 ** ((pitch - 69) / 12.0))
    bins: List[int] = []
    for h in range(1, n_harm + 1):
        b = int(round(f0 * h * n_fft / sr))
        if 0 < b < n_fft // 2:
            bins.append(b)
    return bins


def detect_restrikes(
    audio:       np.ndarray,
    sr:          int,
    notes:       Sequence[Dict],
    rise_ratio:  float = RISE_RATIO,
    min_gap_s:   float = MIN_GAP_S,
    max_splits:  int = 6,
) -> List[Dict]:
    """
    Split notes that contain a re-attack.  Returns a new note list; every
    input note appears at least once, so this can only ADD notes — which is
    the right direction for a pipeline that already under-emits.
    """
    if audio is None or len(audio) == 0 or not notes:
        return list(notes)

    import librosa

    S = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP)).astype(np.float32)
    n_frames = S.shape[1]
    frame_t = HOP / sr

    out: List[Dict] = []
    n_added = 0

    for n in notes:
        onset = float(n["onset"])
        offset = float(n.get("offset", onset + 0.2))
        dur = offset - onset
        # Too short to contain a second attack
        if dur < 2 * min_gap_s:
            out.append(n)
            continue

        f0_i = int(onset / frame_t)
        f1_i = min(n_frames, int(offset / frame_t))
        if f1_i - f0_i < 4:
            out.append(n)
            continue

        bins = _harmonic_bins(int(n["pitch"]), sr, N_FFT)
        if not bins:
            out.append(n)
            continue

        # Per-harmonic envelopes, NOT their sum.
        #
        # Summing was the first attempt and it produced false splits from
        # harmonic collision: G4's fundamental (392 Hz) is within a bin of
        # C4's third harmonic (785 Hz), so striking G4 while C4 is held
        # dumps energy into a bin being watched for C4, and the summed curve
        # jumps.  Verified directly — a held C4 under E4/G4 attacks was split
        # in two.
        #
        # The physics separates them: re-striking a key excites the whole
        # series at once, so ALL of that pitch's harmonics rise together,
        # whereas a colliding neighbour lifts only the one or two bins it
        # happens to share.  Requiring agreement across harmonics keeps the
        # sensitivity and drops the collisions.
        harm = S[bins, f0_i:f1_i]
        env = harm.sum(axis=0)
        if env.size < 4 or float(env.max()) <= 0:
            out.append(n)
            continue

        # Walk the envelope tracking the running minimum since the last
        # accepted attack.  A rise above rise_ratio * that minimum is energy
        # being re-injected into this pitch — i.e. the key was struck again.
        # Track a running minimum per harmonic, and require a majority of
        # them to rise at the same frame.
        need = max(2, int(np.ceil(len(bins) * 0.75)))
        # A harmonic near the noise floor produces enormous RATIOS from tiny
        # absolute wobbles — 0.8 -> 2.0 reads as a 2.5x rise but is nothing.
        # Diagnosed on the held-C4 case: the two weakest harmonics were
        # flickering around 1-3 while the strong ones decayed cleanly, and
        # their noise was enough to reach the agreement threshold.  A
        # harmonic only votes when it carries real energy for this note.
        peak = harm.max(axis=1).astype(np.float64)
        floor = 0.15 * peak
        splits: List[float] = []
        run_min = harm[:, 0].astype(np.float64).copy()
        last_attack = onset
        for i in range(1, env.size):
            t = onset + i * frame_t
            col = harm[:, i].astype(np.float64)
            if t - last_attack < ATTACK_SKIP_S:
                run_min = col.copy()
                continue
            rising = (run_min > 1e-9) & (col > rise_ratio * run_min) & (col > floor)
            # A rise must PERSIST to count.  Piano attacks are broadband —
            # hammer noise from striking any key briefly lifts every bin, so
            # harmonic agreement alone still passed a held C4 under E4/G4
            # attacks.  The physical difference is duration, not spectrum:
            # re-striking this key leaves its own harmonics sitting at a new
            # level, while a neighbour's transient decays straight back.
            sustained = False
            if int(rising.sum()) >= need:
                j0, j1 = i + 1, min(env.size, i + 1 + SUSTAIN_FRAMES)
                if j1 > j0:
                    later = harm[:, j0:j1].mean(axis=1)
                    sustained = int(((run_min > 1e-9) &
                                     (later > rise_ratio * 0.7 * run_min) &
                                     (later > floor)).sum()) >= need
            if (sustained
                    and t - last_attack >= min_gap_s
                    and offset - t >= min_gap_s):
                splits.append(t)
                last_attack = t
                run_min = col.copy()
                if len(splits) >= max_splits:
                    break
            else:
                run_min = np.minimum(run_min, col)

        if not splits:
            out.append(n)
            continue

        # Emit the original note plus one per detected re-attack, carving the
        # span at the split points.  Velocity is inherited; the detector
        # measures that a strike happened, not how hard.
        edges = [onset] + splits + [offset]
        for a, b in zip(edges, edges[1:]):
            if b - a < min_gap_s * 0.5:
                continue
            m = dict(n)
            m["onset"] = round(a, 4)
            # Leave a short release before the next strike.  Physically the
            # damper must fall before the key can sound again, and it keeps
            # the two notes from reading as one continuous tone downstream.
            m["offset"] = round(max(a + min_gap_s * 0.5, b - RELEASE_S), 4)
            m["restrike"] = a > onset + 1e-9
            out.append(m)
        n_added += len(edges) - 2

    if n_added:
        print(f"[restrike] recovered {n_added} repeated-note attacks "
              f"({len(notes)} -> {len(out)} notes)")
    out.sort(key=lambda x: (float(x["onset"]), int(x["pitch"])))
    return out
