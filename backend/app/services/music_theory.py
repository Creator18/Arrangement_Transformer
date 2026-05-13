"""
music_theory.py — Music-theory post-processing for piano arrangements.

Applied as the final pass in the NMF transformation pipeline, after
compression, quantisation, outlier removal, and chord-tone snapping.

Processing order (all adjustments are at most a few semitones):
  1. smooth_bass_voice_leading  — step-wise motion + contrary motion
  2. prepare_cadences           — dominant→tonic at 4-bar phrase boundaries
  3. avoid_parallel_motion      — parallel 5ths / octaves between bass & melody
  4. resolve_tendency_tones     — leading-tone and ♭7 obligations in melody
  5. avoid_augmented_second     — aug 2nd in minor keys
  6. snap_strays_to_scale       — final guardrail: nudge stragglers to scale tones
  7. apply_music_theory         — orchestrates 1-6
"""
from __future__ import annotations

import librosa
import numpy as np
from typing import Dict, List, Optional, Tuple

# ── constants ─────────────────────────────────────────────────────────────────

# Notes below this pitch are treated as the bass voice
BASS_SPLIT = 56          # MIDI 56 = G#3 (matches nmf_voice_leading.BASS_MAX + 1)

# Default MIDI bounds searched when re-pitching bass notes
BASS_LO_DEFAULT = 28     # E1
BASS_HI_DEFAULT = 55     # G3

# Root-preference scoring for bass chord tones (lower = preferred)
# Root (0) and its octave (12) tied at 0; fifth (7) at 1; third (3/4) at 3
_BASS_INTERVAL_PRIORITY: Dict[int, int] = {
    0: 0, 7: 1, 12: 0,
    5: 2,          # fourth / inverted fifth
    4: 3, 3: 3,    # major / minor third
    10: 4, 11: 4,  # minor / major seventh
}

# Cost-function weights for bass voice leading
_PRIORITY_W  = 3   # interval-priority contribution
_DISTANCE_W  = 1   # step-wise proximity contribution
_CONTRARY_W  = 2   # contrary-motion preference contribution


# ── internal helpers ──────────────────────────────────────────────────────────

def _split_voices(notes: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Partition notes into (bass, upper) by BASS_SPLIT."""
    bass  = sorted([n for n in notes if int(n["pitch"]) <  BASS_SPLIT],
                   key=lambda n: float(n["onset"]))
    upper = sorted([n for n in notes if int(n["pitch"]) >= BASS_SPLIT],
                   key=lambda n: float(n["onset"]))
    return bass, upper


def _melody_trend(upper: List[Dict], t_center: float, window: float) -> float:
    """
    Signed pitch change (semitones) of the upper voice in a time window
    centred on t_center.  Positive = ascending, negative = descending.
    Clamped to [−12, +12].
    """
    half = window / 2
    wnd  = [n for n in upper
            if t_center - half <= float(n["onset"]) <= t_center + half]
    if len(wnd) < 2:
        return 0.0
    pitches = [float(n["pitch"])
               for n in sorted(wnd, key=lambda n: float(n["onset"]))]
    return float(np.clip(pitches[-1] - pitches[0], -12.0, 12.0))


def _pitch_at(t: float, voice: List[Dict]) -> Optional[int]:
    """Return the pitch of the voice note sounding at time t, or None."""
    for n in voice:
        onset  = float(n["onset"])
        offset = float(n.get("offset", onset + 0.25))
        if onset <= t < offset:
            return int(n["pitch"])
    return None


def _nearest_pitch_class(
    pitch:     int,
    target_pc: int,
    lo:        int,
    hi:        int,
    max_dist:  int = 6,
) -> Optional[int]:
    """
    Find the MIDI pitch in [lo, hi] with pitch-class == target_pc that is
    closest to `pitch`.  Returns None if nothing is within max_dist semitones.
    """
    best: Optional[int] = None
    best_d = max_dist + 1
    for p in range(lo, hi + 1):
        if p % 12 == target_pc:
            d = abs(p - pitch)
            if d < best_d:
                best_d, best = d, p
    return best


# ── 1. Smooth bass voice leading ──────────────────────────────────────────────

def smooth_bass_voice_leading(
    bass:        List[Dict],
    upper:       List[Dict],
    chord_map:   Dict[int, tuple],
    measure_dur: float,
    lo: int = BASS_LO_DEFAULT,
    hi: int = BASS_HI_DEFAULT,
) -> List[Dict]:
    """
    Re-pitch each bass note so that consecutive notes prefer step-wise
    motion while still landing on chord tones (root > fifth > third).

    Contrary-motion bias: when the upper voice is ascending the cost
    function penalises high bass pitches, and vice-versa, so the two
    voices tend to move in opposite directions.

    The first bass note is kept as an anchor; every subsequent note is
    chosen by minimising:

        cost = PRIORITY_W × interval_priority
             + DISTANCE_W × |p − prev_pitch|
             + CONTRARY_W × (melody_trend_norm × pitch_norm)

    where melody_trend_norm ∈ [−1, +1] and pitch_norm ∈ [0, 1].
    """
    if not bass:
        return bass

    result = [dict(bass[0])]

    for i in range(1, len(bass)):
        n  = bass[i]
        t  = float(n["onset"])
        mi = int(t / measure_dur)

        chord_info = (chord_map.get(mi)
                      or chord_map.get(mi - 1)
                      or chord_map.get(mi + 1))
        if chord_info is None:
            result.append(dict(n))
            continue

        root_pc, chord_pcs = chord_info
        prev_pitch = int(result[-1]["pitch"])

        # Melody trend normalised to [−1, +1]
        trend_norm = _melody_trend(upper, t, 2 * measure_dur) / 12.0

        best_p    = int(n["pitch"])
        best_cost = float("inf")

        for p in range(lo, hi + 1):
            if p % 12 not in chord_pcs:
                continue

            interval = (p - root_pc) % 12
            priority = _BASS_INTERVAL_PRIORITY.get(interval, 5)
            dist     = abs(p - prev_pitch)

            # Contrary-motion cost: high when bass and melody move together
            p_norm        = (p - lo) / max(1, hi - lo)   # 0 = low, 1 = high
            contrary_cost = trend_norm * p_norm

            cost = (_PRIORITY_W * priority
                    + _DISTANCE_W * dist
                    + _CONTRARY_W * contrary_cost)

            if cost < best_cost:
                best_cost, best_p = cost, p

        if best_p != int(n["pitch"]):
            result.append({**n, "pitch": best_p,
                           "pitch_name": librosa.midi_to_note(best_p)})
        else:
            result.append(dict(n))

    return result


# ── 2. Cadential preparation ──────────────────────────────────────────────────

def prepare_cadences(
    bass:        List[Dict],
    key_root:    int,
    measure_dur: float,
    lo: int = BASS_LO_DEFAULT,
    hi: int = BASS_HI_DEFAULT,
) -> List[Dict]:
    """
    At measure 3 of every 4-bar phrase (phrase-position index 2), nudge
    the bass toward the dominant root (key_root + 7 semitones).
    At measure 4 (index 3), nudge toward the tonic (key_root).

    A nudge is applied only when the target pitch is within 5 semitones
    of the current note, preventing jarring leaps.  This enforces the
    V→I cadential motion that gives each phrase a sense of closure.
    """
    dominant_pc = (key_root + 7) % 12
    tonic_pc    = key_root

    result = []
    for n in bass:
        mi         = int(float(n["onset"]) / measure_dur)
        phrase_pos = mi % 4                    # position within 4-bar phrase

        target_pc: Optional[int] = None
        if phrase_pos == 2:
            target_pc = dominant_pc            # pre-cadential → dominant
        elif phrase_pos == 3:
            target_pc = tonic_pc              # cadential → tonic

        if target_pc is not None:
            new_p = _nearest_pitch_class(int(n["pitch"]), target_pc, lo, hi,
                                         max_dist=5)
            if new_p is not None and new_p != int(n["pitch"]):
                n = {**n, "pitch": new_p, "pitch_name": librosa.midi_to_note(new_p)}

        result.append(n)

    return result


# ── 3. Parallel fifths / octaves avoidance ────────────────────────────────────

def avoid_parallel_motion(
    bass:  List[Dict],
    upper: List[Dict],
    lo: int = BASS_LO_DEFAULT,
    hi: int = BASS_HI_DEFAULT,
) -> List[Dict]:
    """
    For each consecutive pair of bass notes, find the simultaneously
    sounding melody pitch.  If both voices move in the same direction and
    the resulting interval class is a perfect fifth (7 st mod 12) or
    octave / unison (0 st mod 12), shift the bass by ±1–2 semitones to
    break the parallel motion.

    Parallel fifths/octaves make the two voices sound like a single line
    rather than independent parts — even on piano this muddies the texture.
    """
    if len(bass) < 2 or not upper:
        return bass

    result = [dict(bass[0])]

    for i in range(1, len(bass)):
        curr_n    = dict(bass[i])
        prev_bass = int(result[-1]["pitch"])
        curr_bass = int(curr_n["pitch"])

        prev_mel = _pitch_at(float(result[-1]["onset"]), upper)
        curr_mel = _pitch_at(float(curr_n["onset"]),     upper)

        if prev_mel is not None and curr_mel is not None:
            bass_dir  = curr_bass - prev_bass
            mel_dir   = curr_mel  - prev_mel
            prev_ivl  = abs(prev_bass - prev_mel) % 12
            curr_ivl  = abs(curr_bass - curr_mel) % 12

            same_dir    = (bass_dir != 0 and mel_dir != 0
                           and (bass_dir > 0) == (mel_dir > 0))
            bad_ivl     = curr_ivl in (0, 7)
            is_parallel = same_dir and bad_ivl and (prev_ivl == curr_ivl)

            if is_parallel:
                for delta in (-1, 1, -2, 2):
                    adj = curr_bass + delta
                    if lo <= adj <= hi:
                        new_ivl = abs(adj - curr_mel) % 12
                        # Accept if the new interval is not a 5th/octave
                        if new_ivl not in (0, 7):
                            curr_n = {**curr_n, "pitch": adj,
                                      "pitch_name": librosa.midi_to_note(adj)}
                            break

        result.append(curr_n)

    return result


# ── 4. Tendency tone resolution ───────────────────────────────────────────────

def resolve_tendency_tones(
    upper:      List[Dict],
    key_root:   int,
    scale_name: str,
) -> List[Dict]:
    """
    Enforce two melodic obligation rules:

    (a) Leading-tone rule — the major 7th (key_root + 11) has a strong
        upward pull toward the tonic.  If the note following a leading
        tone is exactly one semitone away from the tonic and that semitone
        move would reach the tonic, apply it.

    (b) Harmonic-minor cadence rule — in minor keys, the natural ♭7
        (key_root + 10) resolving down to ♭6 (key_root + 8) across a
        phrase weakens the cadence.  Raise the ♭7 retroactively to the
        harmonic leading tone (key_root + 11).

    Only small corrections (≤ 1 semitone) are made to avoid distorting
    the melody.
    """
    if not upper:
        return upper

    leading_pc = (key_root + 11) % 12
    tonic_pc   = key_root
    flat7_pc   = (key_root + 10) % 12
    flat6_pc   = (key_root + 8)  % 12

    result = list(upper)

    for i in range(1, len(result)):
        prev       = result[i - 1]
        curr       = result[i]
        prev_pc    = int(prev["pitch"]) % 12
        curr_pitch = int(curr["pitch"])
        curr_pc    = curr_pitch % 12

        # (a) Leading tone → tonic
        if prev_pc == leading_pc and curr_pc != tonic_pc:
            candidate = curr_pitch + 1
            if candidate % 12 == tonic_pc and 21 <= candidate <= 108:
                result[i] = {**curr, "pitch": candidate,
                             "pitch_name": librosa.midi_to_note(candidate)}

        # (b) Minor ♭7 → ♭6 at cadence: raise ♭7 to ^7
        elif ("minor" in scale_name
              and prev_pc == flat7_pc
              and curr_pc == flat6_pc):
            raised = int(prev["pitch"]) + 1
            if raised % 12 == leading_pc and 21 <= raised <= 108:
                result[i - 1] = {**prev, "pitch": raised,
                                 "pitch_name": librosa.midi_to_note(raised)}

    return result


# ── 5. Augmented 2nd avoidance ────────────────────────────────────────────────

def avoid_augmented_second(
    upper:      List[Dict],
    key_root:   int,
    scale_name: str,
) -> List[Dict]:
    """
    In minor keys the interval ♭6→^7 (ascending) or ^7→♭6 (descending)
    spans 3 semitones — an augmented 2nd — which sounds awkward in
    stepwise melodic lines.

    Fix:
      Ascending  ♭6→^7 : replace ♭6 with natural 6 (key_root + 9), so the
                          approach to the leading tone is stepwise.
      Descending ^7→♭6 : replace ^7 with ♭7 (key_root + 10), so the descent
                          stays within the natural minor scale.

    Corrections are only ±1 semitone and only applied in minor keys.
    """
    if not upper or "minor" not in scale_name:
        return upper

    flat6_pc = (key_root + 8)  % 12   # ♭6
    nat6_pc  = (key_root + 9)  % 12   # natural 6
    lead_pc  = (key_root + 11) % 12   # ^7 (harmonic leading tone)
    flat7_pc = (key_root + 10) % 12   # ♭7

    result = list(upper)

    for i in range(1, len(result)):
        prev    = result[i - 1]
        curr    = result[i]
        prev_pc = int(prev["pitch"]) % 12
        curr_pc = int(curr["pitch"]) % 12

        # Ascending aug 2nd: ♭6 → ^7 → replace ♭6 with natural 6
        if prev_pc == flat6_pc and curr_pc == lead_pc:
            new_p = int(prev["pitch"]) + 1
            if new_p % 12 == nat6_pc and 21 <= new_p <= 108:
                result[i - 1] = {**prev, "pitch": new_p,
                                 "pitch_name": librosa.midi_to_note(new_p)}

        # Descending aug 2nd: ^7 → ♭6 → replace ^7 with ♭7
        elif prev_pc == lead_pc and curr_pc == flat6_pc:
            new_p = int(prev["pitch"]) - 1
            if new_p % 12 == flat7_pc and 21 <= new_p <= 108:
                result[i - 1] = {**prev, "pitch": new_p,
                                 "pitch_name": librosa.midi_to_note(new_p)}

    return result


# ── 6. Physical hand limits ───────────────────────────────────────────────────

# MIDI boundary between left-hand (bass) and right-hand (melody) territory
HAND_SPLIT   = 60    # middle C
MAX_PER_HAND = 5     # fingers per hand
MAX_SPAN     = 15    # semitones — comfortable 10th (e.g. C→E an octave up)
ONSET_TOL    = 0.05  # seconds — window for grouping simultaneous attacks


def _trim_hand(
    hand_notes: List[Dict],
    keep_low:   bool,
) -> List[Dict]:
    """
    Reduce a single hand's simultaneously attacked notes to at most
    MAX_PER_HAND pitches spanning at most MAX_SPAN semitones.

    keep_low=True  → left hand:  anchor = lowest note (bass root)
    keep_low=False → right hand: anchor = highest note (melody)

    Trimming strategy
    -----------------
    1. If > MAX_PER_HAND notes: drop from the *non-anchor* end first, keeping
       the anchor note and the closest (MAX_PER_HAND − 1) neighbours.
    2. If span still > MAX_SPAN after step 1: keep dropping from the
       non-anchor end one at a time until the span fits.
    """
    if len(hand_notes) <= 1:
        return hand_notes

    # Work with a sorted copy (ascending pitch)
    by_pitch = sorted(hand_notes, key=lambda n: int(n["pitch"]))

    # Step 1 — enforce note count
    if len(by_pitch) > MAX_PER_HAND:
        if keep_low:
            # Left hand: keep lowest + nearest 4 above it
            by_pitch = by_pitch[:MAX_PER_HAND]
        else:
            # Right hand: keep highest + nearest 4 below it
            by_pitch = by_pitch[-MAX_PER_HAND:]

    # Step 2 — enforce span
    while len(by_pitch) > 1:
        span = int(by_pitch[-1]["pitch"]) - int(by_pitch[0]["pitch"])
        if span <= MAX_SPAN:
            break
        if keep_low:
            by_pitch = by_pitch[:-1]   # drop topmost (farthest from bass)
        else:
            by_pitch = by_pitch[1:]    # drop bottommost (farthest from melody)

    return by_pitch


def enforce_hand_limits(notes: List[Dict]) -> List[Dict]:
    """
    Enforce physical piano-performance constraints:

      Left hand  (pitch < middle C = 60): ≤ 5 notes, span ≤ 15 semitones.
      Right hand (pitch ≥ 60)           : ≤ 5 notes, span ≤ 15 semitones.
      Total                             : ≤ 10 simultaneous notes.

    Only *simultaneously attacked* notes count against the finger budget.
    A long held bass note does not pin a finger while the right hand runs a
    phrase — only the initial attack moment matters.

    Notes that are not part of a simultaneous cluster (solo single notes or
    sequential lines) are always preserved unchanged.

    When trimming is needed:
      • Left hand  keeps the lowest note (bass anchor) + nearest neighbours.
      • Right hand keeps the highest note (melody)    + nearest neighbours.
    """
    if not notes:
        return notes

    # Sort by onset so we can scan linearly
    by_onset = sorted(notes, key=lambda n: (float(n["onset"]), int(n["pitch"])))
    n_total  = len(by_onset)

    # Build groups of simultaneously attacking notes (within ONSET_TOL)
    groups: List[List[int]] = []   # each entry = list of indices into by_onset
    cur_group: List[int]    = [0]
    cur_t: float            = float(by_onset[0]["onset"])

    for i in range(1, n_total):
        t = float(by_onset[i]["onset"])
        if t - cur_t <= ONSET_TOL:
            cur_group.append(i)
        else:
            groups.append(cur_group)
            cur_group = [i]
            cur_t     = t
    groups.append(cur_group)

    # Decide which indices to keep
    keep: set = set()

    for group_idxs in groups:
        group = [by_onset[i] for i in group_idxs]

        if len(group) == 1:
            # Solo note — always keep
            keep.add(group_idxs[0])
            continue

        # Split into hands
        lh = [(i, by_onset[i]) for i in group_idxs if int(by_onset[i]["pitch"]) < HAND_SPLIT]
        rh = [(i, by_onset[i]) for i in group_idxs if int(by_onset[i]["pitch"]) >= HAND_SPLIT]

        # Trim each hand independently
        lh_notes = _trim_hand([n for _, n in lh], keep_low=True)
        rh_notes = _trim_hand([n for _, n in rh], keep_low=False)

        # Build a pitch→index map for recovery (pitch is unique within a hand group)
        pitch_to_idx = {int(by_onset[i]["pitch"]): i for i, _ in lh + rh}

        for n in lh_notes + rh_notes:
            idx = pitch_to_idx.get(int(n["pitch"]))
            if idx is not None:
                keep.add(idx)

    result = [by_onset[i] for i in range(n_total) if i in keep]
    result.sort(key=lambda n: (float(n["onset"]), int(n["pitch"])))
    return result


# ── orchestrator ──────────────────────────────────────────────────────────────

# ── 6. Final scale guardrail ──────────────────────────────────────────────────

# Scale-degree pitch classes (intervals from tonic).  Covers the modes we
# care about for pop/classical piano transcriptions.
_SCALE_PCS = {
    "major":            (0, 2, 4, 5, 7, 9, 11),
    "ionian":           (0, 2, 4, 5, 7, 9, 11),
    "minor":            (0, 2, 3, 5, 7, 8, 10),   # natural minor
    "natural minor":    (0, 2, 3, 5, 7, 8, 10),
    "harmonic minor":   (0, 2, 3, 5, 7, 8, 11),
    "melodic minor":    (0, 2, 3, 5, 7, 9, 11),
    "dorian":           (0, 2, 3, 5, 7, 9, 10),
    "phrygian":         (0, 1, 3, 5, 7, 8, 10),
    "lydian":           (0, 2, 4, 6, 7, 9, 11),
    "mixolydian":       (0, 2, 4, 5, 7, 9, 10),
    "locrian":          (0, 1, 3, 5, 6, 8, 10),
}


def _scale_pitch_classes(key_root: int, scale_name: str) -> set:
    """
    Return the set of allowed pitch classes for (key_root, scale_name).
    Falls back to major when scale_name is unrecognised.
    """
    intervals = _SCALE_PCS.get((scale_name or "major").lower().strip(),
                               _SCALE_PCS["major"])
    return {(key_root + i) % 12 for i in intervals}


def _choose_nudge(
    pitch:        int,
    candidates:   List[Tuple[int, int, int]],   # list of (|d|, d, cand_pitch)
    prev_pitch:   Optional[int],
    next_pitch:   Optional[int],
    chord_pcs:    Optional[set],
) -> int:
    """
    Pick one nudge from the equal-distance candidates.

    Tiebreak order (only invoked when ≥2 candidates share the minimum |d|):
      1. Chord-tone preference — if exactly one candidate sits on a current
         chord tone, prefer it.  Keeps the nudge consistent with the local
         harmony where possible.
      2. Melodic contour — if (prev_pitch, next_pitch) define an ascending
         or descending line through this note, prefer the candidate that
         preserves that direction.  This is the "fit the original melody"
         rule: the strayed note is presumed to be an off-by-one transcription
         error of a passing/neighbour tone, so the line's contour is the
         best evidence of which direction it was meant to go.
      3. Lower-Hz neighbour — when there's no contour info (edge of voice,
         flat trend), fall back to the candidate with the lower MIDI value.
         In Hz a downward semitone is closer than an upward one (~6 cents),
         so this is the deterministic frequency-distance answer.

    The first rule that yields a unique winner returns it; otherwise we fall
    through to the next rule.
    """
    # 1. Chord-tone preference
    if chord_pcs:
        chord_matches = [c for c in candidates if (c[2] % 12) in chord_pcs]
        if len(chord_matches) == 1:
            return chord_matches[0][2]
        if len(chord_matches) >= 2:
            candidates = chord_matches    # narrow the field and continue tiebreaking

    # 2. Melodic contour from local neighbours
    trend = 0
    if prev_pitch is not None and next_pitch is not None:
        trend = next_pitch - prev_pitch
    elif next_pitch is not None:
        trend = next_pitch - pitch
    elif prev_pitch is not None:
        trend = pitch - prev_pitch

    if trend > 0:
        ups = [c for c in candidates if c[1] > 0]
        if ups:
            return ups[0][2]
    elif trend < 0:
        downs = [c for c in candidates if c[1] < 0]
        if downs:
            return downs[0][2]

    # 3. Lower MIDI value (frequency-closer)
    candidates_sorted = sorted(candidates, key=lambda c: c[1])
    return candidates_sorted[0][2]


def snap_strays_to_scale(
    notes:      List[Dict],
    chord_map:  Dict[int, tuple],
    measure_dur: float,
    key_root:   int,
    scale_name: str,
    max_nudge:  int = 1,
) -> List[Dict]:
    """
    Final guardrail: any note that's both (a) outside the detected scale AND
    (b) outside the current measure's chord tones gets nudged to the closest
    scale tone within `max_nudge` semitones.  Notes that are scale tones,
    chord tones, or too far from any scale tone to nudge safely are left
    alone — we don't want to fight legitimate borrowed chords or chromatic
    passing tones from a confident chord-tone-snap upstream.

    Tiebreak (when ±d both land on scale tones at equal distance):
        chord-tone preference → melodic contour → lower-Hz neighbour.
        See _choose_nudge for the full rule order.

    Why max_nudge=1 by default
    --------------------------
    The most common straggler is a transcription error of ±1 semitone
    (pYIN locking onto a strong harmonic, Basic Pitch confusing F/F#).
    Capping at 1 catches those without warping intentional accidentals or
    blues notes (e.g. ♭3 over a major chord), which are typically further
    from a scale tone than that.

    Note ordering
    -------------
    Internally we sort by onset so the melodic-contour tiebreak can look
    at prev / next notes in the same voice.  The output list is in onset
    order; callers (apply_music_theory) re-sort by (onset, pitch) anyway.
    """
    if not notes:
        return notes

    scale_pcs = _scale_pitch_classes(key_root, scale_name)
    if not scale_pcs:
        return notes

    # Sort by onset so neighbour lookups for the contour tiebreak are correct
    sorted_notes = sorted(notes, key=lambda n: float(n["onset"]))

    out: List[Dict] = []
    for i, n in enumerate(sorted_notes):
        pitch = int(n["pitch"])
        pc    = pitch % 12

        if pc in scale_pcs:
            out.append(n); continue

        # Check chord-tone exemption for this measure
        t  = float(n["onset"])
        mi = int(t / measure_dur) if measure_dur > 0 else 0
        chord_info = (chord_map.get(mi)
                      or chord_map.get(mi - 1)
                      or chord_map.get(mi + 1))
        chord_pcs: Optional[set] = None
        if chord_info is not None:
            _, chord_pcs = chord_info
            if pc in chord_pcs:
                # Genuine chord-tone outside the scale (borrowed chord etc.) — keep it
                out.append(n); continue

        # Collect ALL candidates at the minimum |d|.  d=0 is excluded since
        # the in-scale early-return above already handled that case.
        all_cands: List[Tuple[int, int, int]] = []   # (|d|, d, cand_pitch)
        for d in range(-max_nudge, max_nudge + 1):
            if d == 0:
                continue
            cand = pitch + d
            if 0 <= cand <= 127 and (cand % 12) in scale_pcs:
                all_cands.append((abs(d), d, cand))

        if not all_cands:
            out.append(n); continue

        min_d = min(c[0] for c in all_cands)
        tied  = [c for c in all_cands if c[0] == min_d]

        prev_pitch = int(sorted_notes[i - 1]["pitch"]) if i > 0 else None
        next_pitch = (int(sorted_notes[i + 1]["pitch"])
                      if i + 1 < len(sorted_notes) else None)

        if len(tied) == 1:
            best_p = tied[0][2]
        else:
            best_p = _choose_nudge(pitch, tied, prev_pitch, next_pitch, chord_pcs)

        if best_p == pitch:
            out.append(n); continue

        nudged = {**n, "pitch": int(best_p),
                  "pitch_name": librosa.midi_to_note(int(best_p))}
        out.append(nudged)

    return out


def apply_chord_and_scale_snap(
    notes:      List[Dict],
    chord_map:  Dict[int, tuple],
    tempo:      float,
    key_root:   int,
    scale_name: str,
    bass_lo:    int = BASS_LO_DEFAULT,
    bass_hi:    int = BASS_HI_DEFAULT,
) -> List[Dict]:
    """
    Cleanup pass on the ORIGINAL transcription view (pre-transform).

    Applies:
      - snap_strays_to_scale       (scale-floor on everything; the chord-tone
                                    exemption *inside* this pass still keeps
                                    legitimate borrowed-chord notes from
                                    being snapped)
      - enforce_hand_limits        (≤5 per hand, ≤15-semitone span,
                                    ≤10 total — drops outlier notes that
                                    couldn't be played by a human)

    Deliberately skips:
      - smooth_bass_voice_leading — chord-tone snapping was too aggressive
        on the original bass (it re-pitches every bass note to root/3rd/5th
        of the per-measure chord, which on the 88-key transcription
        flattens real bass-line motion).  The scale-snap alone catches
        off-by-one transcription errors without rewriting the line.
      - prepare_cadences, avoid_parallel_motion, resolve_tendency_tones,
        avoid_augmented_second — stylistic / re-voicing edits, appropriate
        only for the transformed 5-octave arrangement.

    Note `chord_map` is still accepted (and used inside snap_strays_to_scale
    for the borrowed-chord exemption) — only the standalone bass chord-tone
    snap is removed.

    enforce_hand_limits is destructive (it can drop notes), so the ORIGINAL
    list shrinks if the raw transcription had simultaneity outliers (e.g.
    Basic Pitch false positives that produced 6+ note clusters).

    Idempotent in pitch: a second pass leaves pitches unchanged.  Returns a
    new sorted-by-(onset, pitch) list; input is not mutated.

    Function name kept as `apply_chord_and_scale_snap` for caller
    compatibility — "chord" here refers to the chord-tone *exemption*
    inside snap_strays_to_scale rather than the dropped bass chord snap.
    """
    if not notes or tempo <= 0:
        return notes

    measure_dur = (60.0 / tempo) * 4
    bass, upper = _split_voices(notes)

    # 1 — Scale-floor on both voices (with chord-tone exemption inside)
    bass  = snap_strays_to_scale(bass,  chord_map, measure_dur, key_root, scale_name)
    upper = snap_strays_to_scale(upper, chord_map, measure_dur, key_root, scale_name)

    # 2 — Hand-rule pruning (per the user's explicit request)
    merged = sorted(bass + upper,
                    key=lambda n: (float(n["onset"]), int(n["pitch"])))
    return enforce_hand_limits(merged)


def apply_music_theory(
    notes:      List[Dict],
    chord_map:  Dict[int, tuple],
    tempo:      float,
    key_root:   int,
    scale_name: str,
    bass_lo:    int = BASS_LO_DEFAULT,
    bass_hi:    int = BASS_HI_DEFAULT,
) -> List[Dict]:
    """
    Full music-theory correction pass for a piano arrangement.

    Accepts the transformed note list (output of the NMF pipeline) together
    with harmonic context derived from the original transcription.  Pitch
    values may shift by a few semitones.  Note events may be removed if they
    exceed physical hand limits (step 6); all other durations and velocities
    are unchanged.

    Parameters
    ----------
    notes       : transformed note list
    chord_map   : {measure_idx: (root_pc, chord_pcs_set)} — section-aware
                  chord map computed from the original transcription
    tempo       : BPM
    key_root    : tonic pitch class (0 = C … 11 = B)
    scale_name  : 'major', 'minor', 'dorian', etc. (from detect_scale)
    bass_lo/hi  : MIDI search bounds for bass re-pitching
    """
    if not notes or tempo <= 0:
        return notes

    measure_dur = (60.0 / tempo) * 4
    bass, upper = _split_voices(notes)

    # 1 — Smooth bass voice leading (step-wise + contrary motion)
    bass = smooth_bass_voice_leading(
        bass, upper, chord_map, measure_dur, bass_lo, bass_hi
    )

    # 2 — Cadential preparation: V→I at every 4-bar phrase boundary
    bass = prepare_cadences(bass, key_root, measure_dur, bass_lo, bass_hi)

    # 3 — Parallel fifths / octaves between bass and melody
    bass = avoid_parallel_motion(bass, upper, bass_lo, bass_hi)

    # 4 — Tendency-tone resolution in the melody
    upper = resolve_tendency_tones(upper, key_root, scale_name)

    # 5 — Augmented 2nd avoidance (minor keys only)
    upper = avoid_augmented_second(upper, key_root, scale_name)

    # 6 — Scale-floor guardrail (catches stragglers that escape steps 1–5,
    #     typically bass notes in measures where chord_map.get(mi) is None
    #     and a transcription-noise pitch slipped through unchanged)
    bass  = snap_strays_to_scale(bass,  chord_map, measure_dur, key_root, scale_name)
    upper = snap_strays_to_scale(upper, chord_map, measure_dur, key_root, scale_name)

    # 7 — Physical hand limits (≤ 5 per hand, ≤ 15-semitone span, ≤ 10 total)
    merged = sorted(bass + upper, key=lambda n: (float(n["onset"]), int(n["pitch"])))
    return enforce_hand_limits(merged)
