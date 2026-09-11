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

# Ranges used when decoding an ISOLATED stem, where there is no other
# instrument to be confused with and the pitch floor is no longer doing any
# separating work.
#
# VOCAL_LO is 40, not 56.  On a real upload the vocals stem put roughly 30% of
# its notes below G#3 — a male rock vocal sits far lower than POP909's corpus
# average of 0.7% below that line — and every one of those was being handed to
# the accompaniment rebuild and overwritten.
VOCAL_LO, VOCAL_HI = 40, 96    # E2-C7
BASS_LO, BASS_HI = 24, 55      # C1-G3
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
# Superseded by the sustained-dip test below; kept so the old behaviour can be
# compared.  It measured vibrato as re-articulation — see the note in the run
# splitter.
REARTIC_RISE = 0.50

# Swept against the POP909 melody targets on a stem-routed vocal
# (unison 23.6% of intervals, median duration 291 ms):
#
#   frac  min_s   notes  unison%  zero-gap  dur p50
#   0.30   0.03     873    17.5%     17.3%    210ms
#   0.45   0.03     920    21.8%     21.5%    210ms
#   0.60   0.05     986    27.0%     26.8%    190ms
#   0.60   0.08     942    23.6%     23.4%    200ms   <- ships
#   old rise test   1115    35.5%     35.2%    116ms
REARTIC_DIP_FRAC = 0.60    # activation must fall below this share of the peak
REARTIC_DIP_MIN_S = 0.08   # ...and stay there this long to count as a re-strike

# A re-strike RELEASES the key.  Even at the right unison count the repeats sat
# back-to-back with no gap — zero-gap 23.4% against unison 23.6%, so essentially
# every repeat was two notes touching, which is what reads as a double press on
# the roll and as a stutter by ear.  POP909's same-pitch gaps have p25 0.068 and
# p50 0.121, so a human lifts the key for about an eighth of a second.  The
# first note of a pair now ends early by this much.
REARTIC_RELEASE_S = 0.07

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


def _resample(H: np.ndarray, low: int, lo: int, hi: int):
    """Slice H to [lo, hi] and put it on the priors' frame rate. None if OOB."""
    from app.services.piano_projection import FPS as PROJ_FPS
    a, b = lo - low, hi - low + 1
    if a < 0 or b > H.shape[0]:
        return None
    T = H.shape[1]
    n_fr = max(2, int(T / PROJ_FPS * FPS))
    src = np.clip((np.arange(n_fr) / FPS * PROJ_FPS).astype(int), 0, T - 1)
    return H[a:b][:, src]                               # (n_pitch, n_fr)


def _observations(A: np.ndarray):
    """Per-frame log-likelihood of each pitch, and the level silence must beat.

    gamma < 1 deliberately.  Raw activations span orders of magnitude, and at
    gamma 1 the loudest frame overwhelms every prior, which turns the decode
    back into per-frame argmax — exactly the thing being replaced.
    """
    ref = float(np.percentile(A[A > 0], 60)) if np.any(A > 0) else 1.0
    obs = OBS_GAMMA * np.log(A + 1e-9)
    sil = OBS_GAMMA * float(np.log(ref * SIL_FLOOR + 1e-9))
    return obs, sil


def _viterbi(obs: np.ndarray, sil: float, P: Dict[str, Any],
             slot: np.ndarray) -> np.ndarray:
    """Most likely monophonic pitch path through `obs`.  Silence is state n_pitch.

    Extracted from decode() so that every voice goes through the SAME decoder.
    Two voices decoded by two different code paths is how the melody and the
    bass ended up disagreeing about the register they share.
    """
    n_pitch, n_fr = obs.shape
    iv = P["interval"]
    max_iv = P["max_iv"]
    log_iv = np.log(iv[1:-1] + 1e-9)                    # -max_iv..+max_iv
    log_stay = float(np.log(P["p_stay"]))
    log_change_base = float(np.log(1.0 - P["p_stay"]))

    metre = P["metre"]
    metre_w = np.log(metre / metre.mean() + 1e-9)       # 0-mean-ish multiplier

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
    return path


def _path_to_notes(path: np.ndarray, A: np.ndarray, lo: int,
                   P: Dict[str, Any], slot: np.ndarray) -> List[Dict[str, Any]]:
    """Turn a pitch path into notes, re-articulating where the key was struck
    again.

    Shared by every voice for the same reason the decoder is.
    """
    metre = P["metre"]
    n_pitch, n_fr = A.shape
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
            # A SUSTAINED dip, not a momentary one.
            #
            # The previous test asked only whether activation rose sharply after
            # a local minimum, and vibrato satisfies that on every cycle: vibrato
            # IS amplitude modulation, so a ~5 Hz wobble offers a dip-and-rise
            # every 200 ms while the minimum split spacing is 60 ms.  Measured on
            # a stem-routed melody that produced 392 zero-gap same-pitch pairs —
            # 35% of the line doubled — and halved median note length to 116 ms
            # against a human 291 ms.  The bass, which does not vibrate, sat at
            # 22.9% against a human 23.6%: the contrast was the tell.
            #
            # A real re-strike releases the key, so the activation stays down
            # while the string decays.  A vibrato trough passes straight through.
            # So the test is duration, not slope: the level must fall below
            # DIP_FRAC of the run's peak and REMAIN there for DIP_MIN_S before
            # the note counts as struck again.  The cut goes where it comes back.
            low = seg < REARTIC_DIP_FRAC * peak
            dip_fr = max(2, int(REARTIC_DIP_MIN_S * FPS))
            k = 0
            while k < len(low):
                if not low[k]:
                    k += 1
                    continue
                run_end = k
                while run_end < len(low) and low[run_end]:
                    run_end += 1
                # a strong metrical position is where a singer restates a note,
                # so a slightly shorter dip is accepted there
                need = dip_fr
                if run_end < len(seg) and metre[slot[min(i + run_end, n_fr - 1)]] >= metre.mean():
                    need = max(2, int(dip_fr * 0.6))
                if (run_end - k) >= need and run_end < len(seg)                         and (i + run_end) - cuts[-1] >= min_fr:
                    cuts.append(i + run_end)
                k = run_end + 1
        cuts.append(j)

        rel_fr = max(1, int(REARTIC_RELEASE_S * FPS))
        for idx_c, (a0, b0) in enumerate(zip(cuts[:-1], cuts[1:])):
            # Lift the key before the next strike, but only between two notes of
            # this run — the last one keeps its full length.
            end = b0 - rel_fr if idx_c < len(cuts) - 2 else b0
            if (end - a0) / FPS < MIN_NOTE_S:
                continue
            s = A[p_idx, a0:end]
            out.append({
                "pitch": lo + p_idx,
                "onset": a0 / FPS,
                "offset": end / FPS,
                "_act": float(s.max()) if s.size else 0.0,
            })
        i = j

    # velocity from activation, same 40..110 range the projection uses.
    # `q_lo`/`q_hi`, not lo/hi — those name the register and are still live.
    if out:
        acts = np.array([n.pop("_act") for n in out])
        q_lo, q_hi = np.percentile(acts, 5), np.percentile(acts, 95)
        for n, v in zip(out, acts):
            f = (0.0 if q_hi <= q_lo
                 else float(np.clip((v - q_lo) / (q_hi - q_lo), 0, 1)))
            n["velocity"] = int(round(40 + 70 * f))
    return out


def decode(H: np.ndarray, low: int, beat_times: Optional[np.ndarray] = None,
           downbeats: Optional[np.ndarray] = None,
           beats_per_bar: int = 4,
           lo: Optional[int] = None, hi: Optional[int] = None,
           floor: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
    """Viterbi one monophonic line through the activation matrix.

    `H` is (88, T) activations from the projection, `low` its lowest MIDI pitch.
    `lo`/`hi` bound the decoded register.  The default is the melody window, so
    existing callers are unaffected.

    `floor` is an optional per-frame minimum ABSOLUTE MIDI pitch, on the priors'
    frame grid.  It is how the second voice is kept above the first: pitches
    below it are made unreachable rather than merely unlikely, so the two lines
    cannot cross.  A hard register split used to do this job and could not,
    because melody and bass genuinely share E2-G3 and a fixed boundary has to
    give that band to one of them.

    The priors are melody priors for both voices, which is right for a vocal and
    only approximately right for a bass: the structure (one note at a time, a
    dwell cost, onsets pulled to strong beats) transfers, but the interval and
    duration distributions of a bass line are not a singer's.  POP909 has no
    bass track; its PIANO track's lowest voice IS a human left-hand bass and is
    where proper bass priors should come from.
    """
    P = priors()
    if not DECODE_MELODY or P is None or H.size == 0:
        return []
    lo = MEL_LO if lo is None else int(lo)
    hi = MEL_HI if hi is None else int(hi)
    A = _resample(H, low, lo, hi)
    if A is None:
        return []
    obs, sil = _observations(A)
    if floor is not None:
        # unreachable, not just expensive
        n_pitch, n_fr = obs.shape
        f = np.clip(np.asarray(floor, float)[:n_fr] - lo, 0, n_pitch)
        rows = np.arange(n_pitch)[:, None]
        obs = np.where(rows < f[None, :], -1e18, obs)
    slot = _slot_of_frame(A.shape[1], beat_times, downbeats,
                          P["slots"], beats_per_bar)
    path = _viterbi(obs, sil, P, slot)
    return _path_to_notes(path, A, lo, P, slot)


# Semitones the melody must clear the bass by.  Not 1: a melody sitting a
# semitone above the bass is almost always the decoder splitting one note
# between the two voices rather than two parts genuinely that close.
VOICE_MIN_SEP = 3

# OFF: built to close a gap that turned out not to be measured.  See the note
# in reconcile_slots — the per-bar metric it was aimed at was estimator noise.
JOINT_SLOTS = False
JOINT_SUBDIVISIONS = 4     # sixteenths at 4/4, matching the downstream snap

# Place onsets by the OVERLAY instead of by the grid.
#
# Every note is already 99.3% on-grid and they are still heard in the wrong
# place, so sitting on the grid is plainly not what makes a note right.  The
# song's own activation is the reference the grid was only ever a proxy for, so
# use it directly: find where this pitch actually starts sounding, and put the
# note there.
# OFF: measured as a no-op, and for a structural reason worth keeping.
#
#                        H-based overlay      audio onsets (independent)
#     before                     90.9%      66.3% within 40 ms, median 29.3
#     after                      96.6%      65.2% within 40 ms, median 34.8
#     chance                     19.8%      46.6% within 40 ms, median 43.4
#
# Against H it looks decisive; against spectral flux taken straight off the
# waveform it is slightly WORSE.  The gain was circular — this moves notes to
# where d(H) peaks and the overlay metric rewards notes sitting where d(H)
# peaks.  Move magnitudes give it away: median 6 ms, p90 16 ms.
#
# The reason is structural.  The Viterbi already placed these onsets FROM this
# same H, so H's information about when notes start is spent; re-reading it
# cannot recover what was never there.  What little motion appears is mostly
# re-quantising from the decoder's 100 Hz grid onto H's 43 Hz one.
#
# Fixing timing needs a signal the decoder did not already consume: a
# broadband onset envelope at a finer hop.  H has 23 ms frames, so ±12 ms is
# the floor on everything downstream regardless of algorithm.
REFINE_ONSETS = False
REFINE_Q = 60              # a rise must beat this percentile of its own pitch's
REFINE_MAX_MOVE_SUB = 1.0  # ...and may not move further than this many sixteenths


# OFF, because it silently disabled the grid snap that was validated by ear.
#
# This stamps every note it moves with `_timed`, and snap_to_beat_grid honours
# that stamp — a guard added in the same commit on the reasoning that snapping
# was "a no-op, since the notes were already 99.3% on grid".  That was true
# BEFORE this existed.  This is the thing that moves notes off the grid, so the
# guard disabled the correction for exactly the notes that now needed it.
#
# Measured on Complication.mp3, 161 downbeats, bars with melody AND bass both
# on the downbeat — the metric the user fixed and listened to:
#
#     before the grid fix            15 / 161
#     grid fix (REFINE_ENV off)      49 / 161
#     with REFINE_ENV on             24 / 161
#
# 80% of notes carry `_timed`, so only 23.5% survive on the grid, median 23 ms
# off.  Envelope timing and grid snapping are in direct conflict: one places a
# note by audio evidence, the other by metre, and both cannot win.  Envelope
# timing was justified by a 21% improvement in distance-to-onset — the metric
# it optimises — against a grid fix that had been judged by ear.  So the ear
# wins until an A/B says otherwise.
REFINE_ENV = False

# How far a note may be moved.  Chosen musically, not by the metric: a
# sixteenth is ~81 ms at this tempo, so 90 ms lets a note cross into the
# NEIGHBOURING slot, which changes the rhythm rather than tightening it.  60 ms
# keeps every correction inside the note's own slot.
#
# The metric does prefer 90 ms — median distance to a full-mix onset 17.4 ms
# against 23.2 ms at 60 ms, and 919 notes placed against 759.  That is partly
# mechanical, since moving more notes onto onsets improves distance-to-onset by
# construction.  Worth an A/B by ear; it is this one constant.
REFINE_ENV_MAX_MOVE_S = 0.06
REFINE_ENV_PITCH_Q = 25    # H must show at least this much presence at the target


def refine_onsets_env(notes: List[Dict[str, Any]], onsets, H: np.ndarray,
                      low: int, max_move: float = REFINE_ENV_MAX_MOVE_S,
                      proj_fps: Optional[float] = None) -> List[Dict[str, Any]]:
    """Take the TIME from the audio's onset envelope and the PITCH from H.

    This is the split that makes the correction possible.  H decided which
    pitches sound and when, at 23 ms resolution, and refining against it again
    recovers nothing — the decoder already used it.  The broadband envelope is
    a different signal at 5.8 ms, so it knows things about attack timing that
    never entered the decode, but it says nothing about pitch.  So each is
    asked only what it can answer: the envelope supplies the instant, H vetoes
    instants where that pitch is not actually present.

    The veto matters.  Without it a vocal note would happily snap to a guitar
    strum 40 ms away, because a broadband onset does not know whose it is.

    Bounded to +-60 ms — under one sixteenth at most tempos — so this stays a
    correction rather than a search for the best nearby onset.  Notes with no
    qualifying onset in range keep their decoded time.
    """
    if not REFINE_ENV or not notes or onsets is None or len(onsets) == 0:
        return notes
    if H is None or H.size == 0:
        return notes
    if proj_fps is None:
        from app.services.piano_projection import FPS as proj_fps  # noqa: N806

    ons = np.sort(np.asarray(onsets, dtype=float))
    n_fr = H.shape[1]
    pres = np.zeros(H.shape[0])
    for r in range(H.shape[0]):
        row = H[r][H[r] > 0]
        pres[r] = np.percentile(row, REFINE_ENV_PITCH_Q) if row.size else np.inf

    out, moved, vetoed = [], 0, 0
    for n in notes:
        m = dict(n)
        t = float(m["onset"])
        r = int(m["pitch"]) - low
        i = int(np.searchsorted(ons, t))
        best, bd = None, max_move
        for j in (i - 1, i):
            if 0 <= j < len(ons) and abs(ons[j] - t) <= bd:
                best, bd = float(ons[j]), abs(ons[j] - t)
        if best is not None and 0 <= r < H.shape[0]:
            f = int(best * proj_fps)
            a, b = max(0, f - 1), min(n_fr, f + 2)
            if b > a and float(H[r, a:b].max()) >= pres[r]:
                dur = float(m["offset"]) - t
                m["onset"] = best
                m["offset"] = best + max(dur, MIN_NOTE_S)
                m["_timed"] = True
                moved += 1
            else:
                vetoed += 1
        out.append(m)

    out.sort(key=lambda x: (float(x["onset"]), int(x["pitch"])))
    print(f"[decode] envelope timing: {moved}/{len(notes)} placed from onsets, "
          f"{vetoed} vetoed by pitch presence")
    return out


def refine_onsets(notes: List[Dict[str, Any]], H: np.ndarray, low: int,
                  beat_times, subdivisions: int = JOINT_SUBDIVISIONS,
                  proj_fps: Optional[float] = None) -> List[Dict[str, Any]]:
    """Move each onset to where its own energy actually starts.

    NEAREST qualifying rise, not the largest.  Taking the largest is what broke
    the first attempt at measuring this: over a +-250 ms window the biggest
    activation rise at a given pitch is frequently the NEXT note's attack, so
    the note gets dragged onto its neighbour and the error is reported as the
    distance between them.  Bounding the search to one subdivision and
    preferring proximity keeps a correction a correction.

    A rise has to clear the REFINE_Q percentile of that pitch's own positive
    differences, so the bar adapts per pitch — a quiet register is not held to a
    loud one's standard.  Where nothing qualifies the note is left exactly where
    it was, because inventing motion from weak evidence is how a timing fix
    becomes a timing problem.

    Notes it moves are stamped `_timed`, which the downstream quantisers honour.
    Without that they are simply re-snapped to the grid and this does nothing.
    """
    if not REFINE_ONSETS or not notes or H is None or H.size == 0:
        return notes
    grid = _subdiv_grid(beat_times, subdivisions)
    if grid is None:
        return notes
    if proj_fps is None:
        from app.services.piano_projection import FPS as proj_fps  # noqa: N806

    sub_s = float(np.median(np.diff(grid)))
    max_move = REFINE_MAX_MOVE_SUB * sub_s
    win = max(1, int(max_move * proj_fps))
    n_fr = H.shape[1]
    d = np.diff(H, axis=1, prepend=H[:, :1])

    # per-pitch bar for what counts as an attack
    thr = np.zeros(H.shape[0])
    for r in range(H.shape[0]):
        pos = d[r][d[r] > 0]
        thr[r] = np.percentile(pos, REFINE_Q) if pos.size else np.inf

    out, moved = [], 0
    for n in notes:
        m = dict(n)
        r = int(m["pitch"]) - low
        t = float(m["onset"])
        f = int(t * proj_fps)
        if 0 <= r < H.shape[0]:
            a, b = max(0, f - win), min(n_fr, f + win + 1)
            if b > a:
                seg = d[r, a:b]
                ok = np.where(seg >= thr[r])[0]
                if ok.size:
                    # nearest qualifying rise to where we already are
                    best = ok[int(np.argmin(np.abs((a + ok) - f)))]
                    nt = (a + int(best)) / proj_fps
                    if abs(nt - t) > 1e-9:
                        dur = float(m["offset"]) - t
                        m["onset"] = nt
                        m["offset"] = nt + max(dur, MIN_NOTE_S)
                        m["_timed"] = True
                        moved += 1
        out.append(m)

    out.sort(key=lambda x: (float(x["onset"]), int(x["pitch"])))
    print(f"[decode] onset refinement: {moved}/{len(notes)} moved "
          f"(window +-{max_move*1000:.0f} ms)")
    return out


def _subdiv_grid(beat_times, subdivisions: int = JOINT_SUBDIVISIONS):
    bt = np.asarray(beat_times, dtype=float)
    if bt.size < 2:
        return None
    g = [a + (b - a) * k / subdivisions
         for a, b in zip(bt[:-1], bt[1:]) for k in range(subdivisions)]
    g.append(float(bt[-1]))
    return np.asarray(g)


def reconcile_slots(notes: List[Dict[str, Any]], H: np.ndarray, low: int,
                    beat_times, subdivisions: int = JOINT_SUBDIVISIONS,
                    proj_fps: Optional[float] = None) -> List[Dict[str, Any]]:
    """Make notes at the same musical moment share a slot, across voices.

    The two-voice decode couples the voices in PITCH — the floor makes the
    melody depend on the bass, which is why crossings measure at zero — and not
    at all in TIME.  They are two independent Viterbi passes, each with its own
    metrical prior, never asked to agree with each other.  Measured on a real
    arrangement, the two voices' best per-bar alignment differed by more than
    40 ms in 60.2% of bars, against an 81 ms subdivision, while every POOLED
    number said the timing was perfect: melody peak +0 ms, bass peak +0 ms,
    difference +0 ms.  Aggregates hid it completely because the errors are
    inconsistent rather than offset, so they cancel.

    Reconciling per note would not fix this — that is a per-note decision and
    two notes of one musical moment can still choose different slots.  The
    decision has to be JOINT: group onsets that are close enough to be the same
    moment, then pick one slot for the whole group by summed evidence.

    Conservative on purpose, in two ways.  Only notes already within one
    subdivision are grouped, so genuine syncopation — bass on the 'and' against
    melody on the beat — is a full subdivision away and never collapsed.  And a
    group only moves if the move IMPROVES its total evidence, so where the
    activation is ambiguous nothing happens.

    DEFAULTED OFF, because the 60.2% did not survive a control.  Splitting ONE
    voice in half at random and comparing it against itself — zero desync by
    construction — scores 66.9% (melody) and 60.1% (bass) on the same metric.
    Melody vs bass at 57.3% is BELOW that floor, so the number was estimator
    noise: about five onsets per voice per bar, argmax over 33 candidate
    shifts.  A/B'd anyway: 57.3% before, 57.3% after, 182 of 1059 onsets moved,
    onset containment 90.9% -> 91.4%.  No effect on the target, and the small
    containment gain is not what it was built for.

    Kept rather than deleted because the reasoning still holds — the voices
    genuinely are coupled in pitch and not in time — and because a metric that
    can actually resolve voice-to-voice timing would make this testable.  Do
    not turn it on without one.
    """
    grid = _subdiv_grid(beat_times, subdivisions)
    if grid is None or not notes or H is None or H.size == 0:
        return notes
    if proj_fps is None:
        from app.services.piano_projection import FPS as proj_fps  # noqa: N806

    d = np.diff(H, axis=1, prepend=H[:, :1])       # positive = attack
    n_fr = H.shape[1]

    def support(pitch: int, t: float) -> float:
        """Evidence that `pitch` is struck at time `t`, +-1 frame."""
        r = int(pitch) - low
        if not (0 <= r < H.shape[0]):
            return 0.0
        f = int(t * proj_fps)
        a, b = max(0, f - 1), min(n_fr, f + 2)
        return float(max(0.0, d[r, a:b].max())) if b > a else 0.0

    tol = float(np.median(np.diff(grid)))          # one subdivision
    order = sorted(range(len(notes)), key=lambda i: float(notes[i]["onset"]))
    out = [dict(n) for n in notes]

    i = 0
    while i < len(order):
        grp = [order[i]]
        t0 = float(out[order[i]]["onset"])
        j = i + 1
        while j < len(order) and float(out[order[j]]["onset"]) - t0 <= tol:
            grp.append(order[j])
            j += 1
        # only worth deciding jointly if more than one VOICE is involved
        roles = {out[k].get("role") for k in grp}
        if len(grp) > 1 and len(roles) > 1:
            cand = set()
            for k in grp:
                gi = int(np.argmin(np.abs(grid - float(out[k]["onset"]))))
                for dd in (-1, 0, 1):
                    if 0 <= gi + dd < len(grid):
                        cand.add(gi + dd)
            cur = sum(support(out[k]["pitch"], float(out[k]["onset"])) for k in grp)
            best_t, best_s = None, cur
            for gi in cand:
                t = float(grid[gi])
                s = sum(support(out[k]["pitch"], t) for k in grp)
                if s > best_s:
                    best_t, best_s = t, s
            if best_t is not None:
                for k in grp:
                    dur = float(out[k]["offset"]) - float(out[k]["onset"])
                    out[k]["onset"] = best_t
                    out[k]["offset"] = best_t + max(dur, MIN_NOTE_S)
        i = j

    out.sort(key=lambda n: (float(n["onset"]), int(n["pitch"])))
    return out


def decode_two_voices(H: np.ndarray, low: int,
                      beat_times: Optional[np.ndarray] = None,
                      downbeats: Optional[np.ndarray] = None,
                      beats_per_bar: int = 4,
                      onsets: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
    """Decode a bass and a melody from ONE activation matrix, in that order.

    Replaces per-stem routing.  That routing ran two different decoders over two
    different stems at two different FFT sizes and then merged the results, so
    the two parts were never solved against the same evidence and had no way to
    agree about the register they share.  Here there is one decomposition, one
    frame grid, one decoder, called twice.

    Bass first, because it is the more reliable of the two — on every listening
    pass so far the bass has come back close to right while the melody was
    cluttered — and because a floor derived from a good line is worth more than
    a ceiling derived from a poor one.  The melody is then decoded above it with
    a per-frame floor of bass + VOICE_MIN_SEP, which is the non-crossing
    constraint.  Where the bass is silent there is no floor and the melody may
    go as low as its own register allows.

    KNOWN GAP, not yet measured: the floor stops the melody landing ON the bass,
    but the bass's own upper partials sit above the floor, so the melody can
    still latch onto a bass harmonic — most likely the octave, which the
    dictionary already confuses (mean cos(p, p+12) = 0.446).  Subtracting the
    bass's explained spectrum before the second decode is the obvious fix and is
    deliberately NOT done here, because it is a guess until it is measured.
    """
    P = priors()
    if not DECODE_MELODY or P is None or H.size == 0:
        return []

    bass = decode(H, low, beat_times, downbeats, beats_per_bar,
                  lo=BASS_LO, hi=BASS_HI)
    for n in bass:
        n["role"] = "bass"

    from app.services.piano_projection import FPS as PROJ_FPS
    n_fr = max(2, int(H.shape[1] / PROJ_FPS * FPS))
    floor = np.full(n_fr, float(VOCAL_LO))
    for n in bass:
        a = int(n["onset"] * FPS)
        b = min(n_fr, int(n["offset"] * FPS))
        if b > a:
            floor[a:b] = np.maximum(floor[a:b], n["pitch"] + VOICE_MIN_SEP)
    floor = np.minimum(floor, float(VOCAL_HI))

    mel = decode(H, low, beat_times, downbeats, beats_per_bar,
                 lo=VOCAL_LO, hi=VOCAL_HI, floor=floor)
    for n in mel:
        n["role"] = "melody"

    out = bass + mel
    out.sort(key=lambda n: (n["onset"], n["pitch"]))

    # The voices are coupled in pitch by the floor above; this couples them in
    # time.  Without it they are two independent decodes that happen to share a
    # grid, and they disagreed about where the beat was in most bars.
    if JOINT_SLOTS and beat_times is not None:
        out = reconcile_slots(out, H, low, beat_times)
    # Last, so it places notes by evidence rather than by the grid, and nothing
    # after it re-grids them.
    if REFINE_ONSETS and beat_times is not None:
        out = refine_onsets(out, H, low, beat_times)
    if REFINE_ENV and onsets is not None:
        out = refine_onsets_env(out, onsets, H, low)
    return out
