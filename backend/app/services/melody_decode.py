"""
Decode the melody as a line, instead of filtering it out of a cloud.

melody_cleanup.py takes detected notes and removes the ones that look wrong,
using four constants tuned by hand.  It works, and it is the wrong shape for the
problem.  Note-level precision on this path is 0.31-0.47, so half the input is
noise, and no threshold over noise recovers a line.

But the FRAME-LEVEL pitch information is good: Test B measured 60.2% frame
accuracy in the melody register, better than pYIN's 48.7%.  So the question is
not "which of these notes are wrong" but "what is the most likely sung line
through this activation matrix" — a decode, with a musical prior, over evidence
we already have.

This is the same move that fixed the accompaniment.  Individual detections are
unreliable; aggregates are not.  There, the aggregate was the bar's harmony and
rhythm.  Here, it is the shape of melodies in general, measured from 909
human-notated vocal lines (eval/pop909_build_priors.py).

What the decode replaces
------------------------
Four filters become one model, and none of them survives as a constant:

    monophony       the path IS single-valued, by construction
    gap bridging    a re-struck note is one run of frames at one pitch, so it
                    is one note without anything having to merge it
    minimum length  the dwell prior makes short notes expensive rather than
                    illegal — a genuinely short sung note can still win if the
                    evidence is strong
    grid snapping   the metrical prior PULLS onsets toward strong positions
                    while decoding, instead of moving them afterwards
    scale tiebreak  the interval prior does this better: it knows a tone is
                    four times more likely than a semitone (35.4% vs 8.2%),
                    which is what actually kills vibrato smear

States are the pitches in the melody register plus one silence state, because
without an explicit silence the decoder sings continuously — there is always
some activation somewhere.
"""
from __future__ import annotations

import pathlib
from typing import Any, Dict, List, Optional

import numpy as np

DECODE_MELODY = True

MEL_LO, MEL_HI = 56, 96       # G#3-C7: the register the pipeline calls melody
FPS = 100.0                   # must match the priors' fps
OBS_GAMMA = 0.75              # how sharply activation is trusted, see below
SIL_FLOOR = 0.55              # activation level silence has to beat
MIN_NOTE_S = 0.06             # only to discard single-frame decode flickers
# Activation rise, as a share of the run's peak, that counts as striking the
# same key again.  Swept against the POP909 distributions on two real uploads
# and the optimum is sharp, not a plateau:
#
#   rise   notes/s   dur p50   unison%   on-16th   score
#   0.35      2.26     0.210      45.7      64.8   2.016
#   0.50      1.60     0.263      23.2      73.9   0.317   <- human 23.6 / 76.4
#   0.65      1.28     0.390       3.9      74.7   1.259
#   0.80+     1.23     0.408       0.0      76.0   1.505
#
# Above 0.65 no re-attack ever qualifies and unison collapses to zero; at 0.35
# every ripple splits a note and density doubles.  0.50 puts repeated notes
# within half a point of human.
REARTIC_RISE = 0.50

_PRIORS: Optional[Dict[str, Any]] = None
_PRIOR_PATH = pathlib.Path(__file__).with_name("_melody_priors.npz")


def priors() -> Optional[Dict[str, Any]]:
    """POP909-derived priors, or None if the artefact is missing."""
    global _PRIORS
    if _PRIORS is not None:
        return _PRIORS
    if not _PRIOR_PATH.exists():
        return None
    z = np.load(_PRIOR_PATH)
    _PRIORS = {
        "interval": z["interval"], "metre": z["metre"],
        "p_stay": float(z["p_stay"]), "p_sing": float(z["p_sing"]),
        "max_iv": int(z["max_iv"]), "slots": int(z["slots"]),
    }
    return _PRIORS


def _slot_of_frame(n_fr: int, beat_times: Optional[np.ndarray],
                   downbeats: Optional[np.ndarray], slots: int,
                   beats_per_bar: int = 4) -> np.ndarray:
    """Metrical slot (0..slots-1) for every frame."""
    out = np.zeros(n_fr, dtype=np.int16)
    if beat_times is None or len(beat_times) < 2:
        return out
    edges = (np.asarray(downbeats, float) if downbeats is not None
             and len(downbeats) >= 3 else np.asarray(beat_times, float)[::beats_per_bar])
    if len(edges) < 2:
        return out
    t = np.arange(n_fr) / FPS
    idx = np.clip(np.searchsorted(edges, t) - 1, 0, len(edges) - 2)
    b0, b1 = edges[idx], edges[idx + 1]
    span = np.maximum(b1 - b0, 1e-6)
    out = np.clip(((t - b0) / span * slots).astype(np.int16), 0, slots - 1)
    return out


def decode(H: np.ndarray, low: int, beat_times: Optional[np.ndarray] = None,
           downbeats: Optional[np.ndarray] = None,
           beats_per_bar: int = 4) -> List[Dict[str, Any]]:
    """Viterbi a monophonic line through the activation matrix.

    `H` is (88, T) activations from the projection, `low` its lowest MIDI pitch.
    Returns note dicts covering only the melody register.
    """
    P = priors()
    if not DECODE_MELODY or P is None or H.size == 0:
        return []

    n_pitch = MEL_HI - MEL_LO + 1
    a, b = MEL_LO - low, MEL_HI - low + 1
    if a < 0 or b > H.shape[0]:
        return []

    # ── resample activations onto the priors' frame rate ─────────────────────
    from app.services.piano_projection import FPS as PROJ_FPS
    T = H.shape[1]
    n_fr = max(2, int(T / PROJ_FPS * FPS))
    src = np.clip((np.arange(n_fr) / FPS * PROJ_FPS).astype(int), 0, T - 1)
    A = H[a:b][:, src]                                  # (n_pitch, n_fr)

    # ── observation log-likelihoods ─────────────────────────────────────────
    # gamma < 1 deliberately.  Raw activations span orders of magnitude, and at
    # gamma 1 the loudest frame overwhelms every prior, which turns the decode
    # back into per-frame argmax — exactly the thing being replaced.
    ref = float(np.percentile(A[A > 0], 60)) if np.any(A > 0) else 1.0
    obs = OBS_GAMMA * np.log(A + 1e-9)
    sil = OBS_GAMMA * float(np.log(ref * SIL_FLOOR + 1e-9))

    # ── transitions ─────────────────────────────────────────────────────────
    iv = P["interval"]
    max_iv = P["max_iv"]
    log_iv = np.log(iv[1:-1] + 1e-9)                    # -max_iv..+max_iv
    log_stay = float(np.log(P["p_stay"]))
    log_change_base = float(np.log(1.0 - P["p_stay"]))

    metre = P["metre"]
    metre_w = np.log(metre / metre.mean() + 1e-9)       # 0-mean-ish multiplier
    slot = _slot_of_frame(n_fr, beat_times, downbeats, P["slots"], beats_per_bar)

    # Silence dwell from the voiced fraction: rests last as long as the voiced
    # proportion implies, so the decoder does not chatter in and out of song.
    p_sing = P["p_sing"]
    mean_note = 1.0 / (1.0 - P["p_stay"])
    mean_rest = max(2.0, mean_note * (1.0 - p_sing) / max(p_sing, 1e-3))
    log_sil_stay = float(np.log(1.0 - 1.0 / mean_rest))
    log_sil_exit = float(np.log(1.0 / mean_rest))
    log_to_sil = log_change_base + float(np.log(0.35))  # rest is one of the moves

    NEG = -1e18
    delta = np.full(n_pitch, NEG, dtype=np.float64)
    delta_sil = 0.0
    bp = np.zeros((n_fr, n_pitch + 1), dtype=np.int16)   # last col = silence

    for t in range(1, n_fr):
        w = metre_w[slot[t]]
        # staying on the current pitch
        best = delta + log_stay
        arg = np.arange(n_pitch, dtype=np.int16)
        # moving by d semitones — 25 shifted vectors beats a 41x41 matrix
        for d in range(-max_iv, max_iv + 1):
            if d == 0:
                continue
            cand = np.full(n_pitch, NEG)
            if d > 0:
                cand[d:] = delta[:-d]
            else:
                cand[:d] = delta[-d:]
            cand = cand + log_change_base + w + log_iv[d + max_iv]
            better = cand > best
            best[better] = cand[better]
            arg[better] = (np.arange(n_pitch, dtype=np.int16) - d)[better]
        # entering from silence
        from_sil = delta_sil + log_sil_exit + w
        better = from_sil > best
        best[better] = from_sil
        arg[better] = n_pitch                            # came from silence

        new_sil = max(delta_sil + log_sil_stay, float(delta.max()) + log_to_sil)
        bp[t, n_pitch] = (n_pitch if delta_sil + log_sil_stay >=
                          float(delta.max()) + log_to_sil else int(delta.argmax()))
        bp[t, :n_pitch] = arg

        delta = best + obs[:, t]
        delta_sil = new_sil + sil

    # ── backtrace ───────────────────────────────────────────────────────────
    state = int(delta.argmax()) if float(delta.max()) > delta_sil else n_pitch
    path = np.empty(n_fr, dtype=np.int16)
    for t in range(n_fr - 1, -1, -1):
        path[t] = state
        state = int(bp[t, state])

    # ── runs become notes, re-articulated where the singer struck again ─────
    #
    # A run of frames at one pitch is ONE note to the state machine, so the
    # decoder cannot express a repeated note — measured, that produced 0.0%
    # unisons against a human 23.6%, and unison is the most common interval in
    # pop melody.  It also inflated median note length to 0.407 s against 0.291.
    #
    # The pitch path cannot fix this; the evidence can.  Within a run, a dip in
    # activation followed by a clear rise is a fresh attack on the same key, so
    # runs are split there.  The metrical prior gets a second use: a candidate
    # split near a strong slot is accepted at a lower bar than one mid-beat,
    # because that is where a singer restates a note.
    out: List[Dict[str, Any]] = []
    min_fr = max(2, int(MIN_NOTE_S * FPS))
    i = 0
    while i < n_fr:
        if path[i] == n_pitch:
            i += 1
            continue
        j = i
        while j < n_fr and path[j] == path[i]:
            j += 1
        p_idx = int(path[i])
        seg = A[p_idx, i:j]

        cuts = [i]
        if seg.size > 2 * min_fr:
            peak = float(seg.max()) or 1.0
            k = min_fr
            while k < len(seg) - min_fr:
                window = seg[max(0, k - min_fr):k + 1]
                dip = float(window.min())
                rise = float(seg[k]) - dip
                strong = metre[slot[i + k]] >= metre.mean()
                if rise > (REARTIC_RISE * (0.75 if strong else 1.0)) * peak \
                        and i + k - cuts[-1] >= min_fr:
                    cuts.append(i + k)
                    k += min_fr
                else:
                    k += 1
        cuts.append(j)

        for a0, b0 in zip(cuts[:-1], cuts[1:]):
            if (b0 - a0) / FPS < MIN_NOTE_S:
                continue
            s = A[p_idx, a0:b0]
            out.append({
                "pitch": MEL_LO + p_idx,
                "onset": a0 / FPS,
                "offset": b0 / FPS,
                "_act": float(s.max()) if s.size else 0.0,
            })
        i = j

    # velocity from activation, same 40..110 range the projection uses
    if out:
        acts = np.array([n.pop("_act") for n in out])
        lo, hi = np.percentile(acts, 5), np.percentile(acts, 95)
        for n, v in zip(out, acts):
            f = 0.0 if hi <= lo else float(np.clip((v - lo) / (hi - lo), 0, 1))
            n["velocity"] = int(round(40 + 70 * f))
    return out
