"""
Tone shaping — a little more weight, a little less edge.

Two shelving filters over the finished render: a low shelf lifting the bass
and a high shelf easing the treble.  Shelves rather than peaking bands because
the request is about the overall balance of the instrument, not about fixing a
resonance at one frequency.

Where the corners sit, and why not higher
-----------------------------------------
Measured, the render's spectral centroid is ~661 Hz and its practical
bandwidth is ~4.5 kHz — there is almost nothing above 6.8 kHz to cut.  So a
high shelf at 6 or 8 kHz, which is where "reduce treble" would normally land,
would do essentially nothing here.  The perceived brightness in this material
lives around 2-4 kHz, so the shelf sits at 3 kHz where there is actually
energy to shape.

The low shelf sits at 220 Hz: high enough to lift the fundamental of the bass
register (A0 is 27.5 Hz but its weight is carried by partials well above
that), low enough not to thicken the middle where 60% of the energy already
sits and where more would read as muddy rather than full.

Gains are modest on purpose — "slightly" was the request, and this stacks on
top of a room that already adds low-mid energy.
"""
from __future__ import annotations

import numpy as np

LOW_SHELF_HZ = 220.0
LOW_SHELF_DB = 2.5        # bass lift

HIGH_SHELF_HZ = 3000.0
HIGH_SHELF_DB = -2.0      # treble ease

APPLY_TONE = True


def _shelf(sr: int, f0: float, gain_db: float, high: bool) -> tuple:
    """RBJ cookbook shelving biquad. Returns (b, a)."""
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / sr
    cos_w0, sin_w0 = np.cos(w0), np.sin(w0)
    # S = 1 → the gentlest shelf slope, which is what "slightly" wants; a
    # steeper one would announce itself as processing.
    alpha = sin_w0 / 2.0 * np.sqrt(2.0)
    two_sqrtA_alpha = 2.0 * np.sqrt(A) * alpha

    if high:
        b = [A * ((A + 1) + (A - 1) * cos_w0 + two_sqrtA_alpha),
             -2 * A * ((A - 1) + (A + 1) * cos_w0),
             A * ((A + 1) + (A - 1) * cos_w0 - two_sqrtA_alpha)]
        a = [(A + 1) - (A - 1) * cos_w0 + two_sqrtA_alpha,
             2 * ((A - 1) - (A + 1) * cos_w0),
             (A + 1) - (A - 1) * cos_w0 - two_sqrtA_alpha]
    else:
        b = [A * ((A + 1) - (A - 1) * cos_w0 + two_sqrtA_alpha),
             2 * A * ((A - 1) - (A + 1) * cos_w0),
             A * ((A + 1) - (A - 1) * cos_w0 - two_sqrtA_alpha)]
        a = [(A + 1) + (A - 1) * cos_w0 + two_sqrtA_alpha,
             -2 * ((A - 1) + (A + 1) * cos_w0),
             (A + 1) + (A - 1) * cos_w0 - two_sqrtA_alpha]

    a0 = a[0]
    return np.array(b) / a0, np.array(a) / a0


def apply_tone(audio: np.ndarray, sr: int) -> np.ndarray:
    """Low shelf up, high shelf down, level preserved."""
    if not APPLY_TONE or audio is None or np.size(audio) == 0:
        return audio
    from scipy.signal import lfilter

    y = np.asarray(audio, dtype=np.float32)
    mono = y.ndim == 1
    chans = [y] if mono else [y[:, c] for c in range(y.shape[1])]

    bl, al = _shelf(sr, LOW_SHELF_HZ, LOW_SHELF_DB, high=False)
    bh, ah = _shelf(sr, HIGH_SHELF_HZ, HIGH_SHELF_DB, high=True)

    out = []
    for ch in chans:
        z = lfilter(bl, al, ch)
        z = lfilter(bh, ah, z)
        out.append(np.asarray(z, dtype=np.float32))

    res = out[0] if mono else np.stack(out, axis=1)

    # Preserve level, for the same reason the reverb does: a bass lift raises
    # peak, and an A/B at different loudness stops being a test of tone.
    src_peak = float(np.abs(y).max())
    cur_peak = float(np.abs(res).max())
    if cur_peak > 1e-9 and src_peak > 1e-9:
        res = res * (src_peak / cur_peak)
    peak = float(np.abs(res).max())
    if peak > 0.99:
        res = res * (0.99 / peak)
    return res
