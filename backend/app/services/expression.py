"""Expression for the song arranger's notes: dynamics (velocity) and sustain (pedal), both read from the song itself.

The user (2026-10-27): every note is numerically right but it is not the song they remember.  Two measured causes:

  dynamics   the base's velocities sit in a narrow band (one real song: p10 78, median 83, p90 93), so soft notes
             play as loud as strong ones.  Its velocity head is a squared-error regression (hedges to the mean), 40%
             of Music2MIDI targets are a flat default 80, and the renders' stem gains / soundfonts / mastering break
             the link between a velocity and how loud it sounds.
  sustain    the song path sent no pedal at all (transcription._last_pedal = []), so every note's sound stopped at its
             key release: the cut-off the user hears.  The sampler already plays a pedal track (CC64, Salamander's
             sustain layer and resonance); only its source was missing.

  ARRANGEMENT_DYNAMICS=1   the base's velocities stretched per song to a pianist's spread: its ORDERING is already right
                           (Pearson 0.81 with pianists' velocities on pedalled MAESTRO renders, eval/expression_bench.py)
                           but it is squeezed (spread 0.73, soft notes +17 too loud), and squeezed harder on pop songs
                           (one song: p10-p90 78-93).  Per song: 64 + 14 x (v - median) / std: spread 1.03, soft bias
                           +1.2, MAE 11.9 -> 10.3, ordering unchanged (a fixed x1.36 fits MAESTRO but not pop).  The key's CQT level and the song's loudness
                           curve were fitted too and added nothing (the base already hears loudness; it only squeezes it).
  ARRANGEMENT_SUSTAIN=1    pedal down where the song's sound outlasts its keys (a released note's pitch keeps ringing),
                           lifted and re-pressed when the bass moves, so chords do not blur.  On the renders no rule
                           beats "always down" on frame F1 (pianists hold it 61% of the time); this one scores 0.62
                           (down 79%) and catches 45% of the pianist's lifts.  The ear decides.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

DYNAMICS = os.environ.get("ARRANGEMENT_DYNAMICS", "0") == "1"
SUSTAIN = os.environ.get("ARRANGEMENT_SUSTAIN", "0") == "1"

SR = 24000
HOP = 384                                  # 16 ms, the front end's frame
FPS = SR / HOP

# velocity = C0 + C_BASE * (base - song median) + C_KEY * key level + C_SONG * song loudness (dB, see features());
# the stretch (eval/expression_bench.py: MAESTRO renders, pianists' median 64; x1.36 brings the spread to 0.98)
COEF = {"C0": 64.0, "C_BASE": 1.36, "C_KEY": 0.0, "C_SONG": 0.0}
DYN_CENTER, DYN_STD = 64.0, 14.0           # what apply_dynamics uses: the per-song stretch
VEL_LO, VEL_HI = 12, 120

# sustain: a released note "rings" while its key's level stays within RING_DB of its peak; pedal regions are where
# at least MIN_RING released notes ring (smoothed over SMOOTH_S); inside them the pedal changes at each new chord
# whose pitch classes overlap the held ones by less than CHANGE_JACCARD (0 = off), or whose bass moves.  Chosen on
# the bench's validation renders: RING_DB 40 (any audible ring), bass moves only.
RING_DB = 40.0
MIN_RING = 1
SMOOTH_S = 0.25
CHANGE_JACCARD = 0.0
LIFT_S = 0.05                              # the damper lifts at the new chord and comes back down this much later
MIN_PEDAL_S = 0.15


def cqt_db(y: np.ndarray) -> np.ndarray:
    """(88, T) dB per piano key, 16 ms frames."""
    import librosa
    C = np.abs(librosa.cqt(y.astype(np.float32), sr=SR, hop_length=HOP, fmin=librosa.midi_to_hz(21),
                           n_bins=88, bins_per_octave=12))
    return librosa.amplitude_to_db(C, ref=np.max(C) if C.size else 1.0)


def loudness_curve(y: np.ndarray, n: int) -> np.ndarray:
    """The song's loudness per 16 ms frame (RMS dB), smoothed over ~1.5 s."""
    from scipy.ndimage import uniform_filter1d
    m = len(y) // HOP
    e = (y[: m * HOP].astype(np.float64).reshape(m, HOP) ** 2).mean(1) if m else np.zeros(1)
    db = uniform_filter1d(10 * np.log10(e + 1e-10), size=max(1, int(1.5 * FPS)))
    return np.pad(db, (0, max(0, n - len(db))), mode="edge")[:n]


def features(notes: List[Dict], y: np.ndarray, C: Optional[np.ndarray] = None) -> np.ndarray:
    """(N, 3) per note: the base's velocity minus the song's median; the key's peak level in its first 80 ms
    relative to the song's loudness there (both in dB below the song's own maximum); the song's loudness at the
    start minus its median (quiet verse < 0 < loud chorus)."""
    C = cqt_db(y) if C is None else C
    T = C.shape[1]
    L = loudness_curve(y, T)
    Lmax, Lmed = L.max(), np.median(L)
    v = np.array([float(g["velocity"]) for g in notes])
    med = np.median(v) if len(v) else 0.0
    out = np.zeros((len(notes), 3))
    for i, g in enumerate(notes):
        k = int(g["pitch"]) - 21
        a = min(T - 1, max(0, int(g["onset"] * FPS)))
        b = min(T, a + max(1, int(0.08 * FPS)))
        key = C[k, a:b].max() if 0 <= k < 88 else C[:, a].max()
        out[i] = (v[i] - med, key - (L[a] - Lmax), L[a] - Lmed)
    return out


def apply_dynamics(notes: List[Dict], center: float = DYN_CENTER, std: float = DYN_STD) -> List[Dict]:
    """The base's velocities stretched about the song's median to a pianist's spread (order kept)."""
    if len(notes) < 2:
        return notes
    v = np.array([float(g["velocity"]) for g in notes])
    z = (v - np.median(v)) / max(1e-6, v.std())
    return [dict(g, velocity=int(np.clip(round(center + std * x), VEL_LO, VEL_HI))) for g, x in zip(notes, z)]


def sustain_pedal(notes: List[Dict], y: np.ndarray, C: Optional[np.ndarray] = None, ring_db: float = RING_DB,
                  min_ring: int = MIN_RING, change_jaccard: float = CHANGE_JACCARD) -> List[Dict]:
    """Damper-down intervals [{onset, offset}] for the notes, from how the song's sound rings past its keys."""
    from scipy.ndimage import uniform_filter1d
    if not notes:
        return []
    C = cqt_db(y) if C is None else C
    T = C.shape[1]
    ring = np.zeros(T)
    by_key: Dict[int, List[Dict]] = {}
    for g in notes:
        by_key.setdefault(int(g["pitch"]), []).append(g)
    for p, gs in by_key.items():
        k = p - 21
        if not 0 <= k < 88:
            continue
        gs = sorted(gs, key=lambda g: g["onset"])
        for j, g in enumerate(gs):
            a = min(T - 1, int(g["onset"] * FPS))
            off = min(T, max(a + 1, int(g["offset"] * FPS)))
            nxt = min(T, int(gs[j + 1]["onset"] * FPS)) if j + 1 < len(gs) else min(T, off + int(3 * FPS))
            if nxt <= off:
                continue
            peak = C[k, a:min(T, a + max(1, int(0.1 * FPS)))].max()
            seg = C[k, off:nxt] >= peak - ring_db
            run = np.argmin(seg) if not seg.all() else len(seg)         # ringing until it first drops below
            ring[off:off + run] += 1
    ring = uniform_filter1d(ring, size=max(1, int(SMOOTH_S * FPS)))
    down = ring >= min_ring * 0.5
    # chord changes: onset clusters (40 ms); a change when the bass moves or the pitch classes overlap the held
    # chord's by less than change_jaccard
    ons = sorted(notes, key=lambda g: g["onset"])
    clusters, cur = [], [ons[0]]
    for g in ons[1:]:
        if g["onset"] - cur[0]["onset"] <= 0.04:
            cur.append(g)
        else:
            clusters.append(cur)
            cur = [g]
    clusters.append(cur)
    changes, held, bass = [], set(), None
    for cl in clusters:
        pcs = {int(g["pitch"]) % 12 for g in cl}
        b = min(int(g["pitch"]) for g in cl) % 12
        jac = len(pcs & held) / max(1, len(pcs | held))
        if not held or jac < change_jaccard or (len(cl) >= 2 and b != bass):
            changes.append(cl[0]["onset"])
            held, bass = set(pcs), b
        else:
            held |= pcs
    # pedal intervals: within each down region, from just after a change to the next change
    out = []
    edges = np.flatnonzero(np.diff(np.r_[0, down.astype(np.int8), 0]))
    for r0, r1 in zip(edges[::2], edges[1::2]):
        s0, s1 = r0 / FPS, r1 / FPS
        cuts = [c for c in changes if s0 < c < s1]
        bounds = [s0] + cuts + [s1]
        for a, b in zip(bounds[:-1], bounds[1:]):
            a2 = a + (LIFT_S if a in cuts else 0.0)
            b2 = b - 0.01 if b in cuts else b
            if b2 - a2 >= MIN_PEDAL_S:
                out.append({"onset": round(a2, 4), "offset": round(b2, 4)})
    return out


def load(audio_path) -> np.ndarray:
    import librosa
    y, _ = librosa.load(str(audio_path), sr=SR, mono=True)
    return y
