"""
Is this a solo piano recording, or a song with other instruments?

Used to decide whether separation is worth running.  Measured earlier: on
solo piano, beat-tracking the RAW mix reproduces the Demucs-derived grid
exactly (F = 1.000) at zero cost, because the "drum-less harmonic mix" of a
piano recording is the recording.  With drums present Demucs earns its ~10s.
So the right answer is conditional, and this is the condition.

The bias is deliberate and asymmetric
--------------------------------------
Calling a song "piano" skips Demucs, feeds the beat tracker a drum-heavy mix
and degrades the arrangement SILENTLY — the user never learns why it got
worse.  Calling piano "song" costs ten seconds.  Those are not equal, so this
returns piano_only=True only on strong evidence and lets everything else fall
through to the slow, safe path.

Signal: spectral flatness (cheap — no HPSS, which measured 13s, slower than
the Demucs it would be saving)
-----------------------------------------------------------------------------
Flatness is the ratio of geometric to arithmetic mean of the spectrum: high
for noise-like content (cymbals, snare, distortion), low for the harmonic
stacks a piano makes.  On 20 MAESTRO solo-piano recordings it measured a
median of 0.00020 and a maximum of 0.00152 — an extremely tight, extremely
low distribution, which is what makes it usable as a gate.

A percussive-onset ratio was tried first and DISCARDED: defined as the share
of onset energy in the top decile of frames, it sits near 0.5 for any peaky
signal by construction, and solo piano scored 0.517 median.  It measured
nothing.  Recorded here because it looks like an obvious feature to try.

Sampled from several short excerpts rather than the whole file: an intro can
be solo piano while the rest is a full band, and the decision must reflect
the piece.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


# 0.005 — a bit over 3x the highest value seen across 20 solo-piano
# recordings (0.00152), so genuine piano has headroom, while remaining far
# below anything with a drum kit in it.  Set from the measured distribution
# rather than guessed; the first guess of 0.030 was 20x too loose.
PIANO_MAX_FLATNESS   = 0.005
EXCERPT_S            = 4.0
N_EXCERPTS           = 5


def analyse(audio: np.ndarray, sr: int) -> Dict:
    """Return the raw measurements plus a conservative piano_only verdict."""
    import librosa

    if audio.size < sr:
        return {"piano_only": False, "reason": "too short to judge",
                "flatness": None}

    dur = audio.size / sr
    step = max(EXCERPT_S, (dur - EXCERPT_S) / max(1, N_EXCERPTS - 1))
    flat = []
    for i in range(N_EXCERPTS):
        t0 = min(i * step, max(0.0, dur - EXCERPT_S))
        seg = audio[int(t0 * sr): int((t0 + EXCERPT_S) * sr)]
        if seg.size < sr // 2:
            continue
        f = librosa.feature.spectral_flatness(y=seg)
        flat.append(float(np.mean(f)))

    if not flat:
        return {"piano_only": False, "reason": "no usable excerpts",
                "flatness": None}

    fl = float(np.median(flat))
    piano = fl <= PIANO_MAX_FLATNESS
    return {
        "piano_only": bool(piano),
        "flatness": round(fl, 6),
        "reason": "solo piano" if piano else "other instruments likely",
    }


# ── Validation, 2026-08-09 ────────────────────────────────────────────────────
#
#   20 MAESTRO solo-piano recordings -> piano_only 20/20
#     flatness median 0.000195   max 0.001519   (threshold 0.005)
#
#   same piano with broadband noise added, standing in for a kit:
#     +1%  -> piano_only False, flatness 0.128
#     +3%  -> piano_only False, flatness 0.279
#     +8%  -> piano_only False, flatness 0.450
#
# The gap is roughly 650x between clean piano and piano with 1% noise, which
# is what makes a single threshold safe here rather than fragile.
#
# HONEST LIMIT: the negative side is only validated against synthetic noise,
# and broadband noise is not a drum kit — real percussion is transient rather
# than continuous, and a sparse kit under a loud piano will score far lower
# than these figures suggest.  The piano side (20/20, the side that must not
# produce false positives) is validated on real recordings; the song side is
# not, and needs actual song material before this gates anything.
