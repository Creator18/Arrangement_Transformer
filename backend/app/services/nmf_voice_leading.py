"""
NMF + Voice Leading arrangement transformation.

Pipeline:
  1. Build piano roll (pitch × time) from note list
  2. NMF decompose into W (pitch patterns) × H (temporal activations)
  3. Compress each pitch-pattern column of W into the target octave range
  4. Reconstruct compressed piano roll
  5. Convert back to note events
  6. Voice-lead consecutive chords: Hungarian assignment minimises total
     semitone movement between successive chord voicings
"""
from __future__ import annotations

import numpy as np
import librosa
from sklearn.decomposition import NMF
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Any, Optional

from pathlib import Path

from app.services.music_theory import apply_music_theory

# ── constants ─────────────────────────────────────────────────────────────────
PIANO_LOW   = 21    # A0
PIANO_HIGH  = 108   # C8
N_PITCHES   = PIANO_HIGH - PIANO_LOW + 1
FRAME_RATE  = 50    # frames per second  (20 ms resolution)

# Default target: 5-octave keyboard C2–C7
TARGET_LOW  = 36    # C2
TARGET_HIGH = 96    # C7

# Bass lives below G3; melody/voice above G#3
BASS_MAX    = 55
MELODY_MIN  = 56


# ── piano roll helpers ─────────────────────────────────────────────────────────

def build_piano_roll(notes: List[Dict], duration: float) -> np.ndarray:
    """Return float32 array (N_PITCHES, n_frames) with velocity values 0–1."""
    n_frames = int(np.ceil(duration * FRAME_RATE)) + 1
    roll = np.zeros((N_PITCHES, n_frames), dtype=np.float32)
    for n in notes:
        p = int(n["pitch"]) - PIANO_LOW
        if not (0 <= p < N_PITCHES):
            continue
        f0 = int(float(n["onset"]) * FRAME_RATE)
        f1 = max(f0 + 1, int(float(n.get("offset", n["onset"] + 0.25)) * FRAME_RATE))
        vel = float(n.get("velocity", 80)) / 127.0
        roll[p, f0:f1] = np.maximum(roll[p, f0:f1], vel)
    return roll


def piano_roll_to_notes(roll: np.ndarray, min_dur: float = 0.06) -> List[Dict]:
    """Convert piano roll back to note list, dropping notes shorter than min_dur."""
    notes: List[Dict] = []
    thresh = 0.08
    for p_idx in range(N_PITCHES):
        pitch = p_idx + PIANO_LOW
        row = roll[p_idx]
        in_note = False
        onset_f = 0
        vel_acc: List[float] = []

        for f, v in enumerate(row):
            if v >= thresh and not in_note:
                in_note = True
                onset_f = f
                vel_acc = [v]
            elif v >= thresh and in_note:
                vel_acc.append(v)
            elif v < thresh and in_note:
                dur = (f - onset_f) / FRAME_RATE
                if dur >= min_dur:
                    notes.append({
                        "pitch": pitch,
                        "pitch_name": librosa.midi_to_note(pitch),
                        "onset":  round(onset_f / FRAME_RATE, 4),
                        "offset": round(f / FRAME_RATE, 4),
                        "velocity": int(np.mean(vel_acc) * 127),
                    })
                in_note = False

        if in_note:
            dur = (len(row) - onset_f) / FRAME_RATE
            if dur >= min_dur:
                notes.append({
                    "pitch": pitch,
                    "pitch_name": librosa.midi_to_note(pitch),
                    "onset":  round(onset_f / FRAME_RATE, 4),
                    "offset": round(len(row) / FRAME_RATE, 4),
                    "velocity": int(np.mean(vel_acc) * 127),
                })

    notes.sort(key=lambda n: (n["onset"], n["pitch"]))
    return notes


# ── NMF compression ───────────────────────────────────────────────────────────

def _compress_W_column(col: np.ndarray, tgt_low: int, tgt_high: int) -> np.ndarray:
    """
    Fold each active pitch into the target range by shifting up/down in
    whole octaves (12 semitones).  This preserves every pitch class exactly
    (C stays C, F# stays F#) — only the octave changes, so harmony and
    scale are never distorted.
    """
    active = np.where(col > 1e-4)[0]
    if len(active) == 0:
        return col.copy()

    tgt_lo_idx = tgt_low  - PIANO_LOW
    tgt_hi_idx = tgt_high - PIANO_LOW

    new_col = np.zeros_like(col)
    for p in active:
        new_p = int(p)
        # Shift up by octaves until inside or above target
        while new_p < tgt_lo_idx:
            new_p += 12
        # Shift down by octaves until inside or below target
        while new_p > tgt_hi_idx:
            new_p -= 12
        new_p = max(tgt_lo_idx, min(tgt_hi_idx, new_p))
        new_col[new_p] += col[p]

    return new_col


def nmf_compress(
    roll: np.ndarray,
    tgt_low: int = TARGET_LOW,
    tgt_high: int = TARGET_HIGH,
    n_components: int = 16,
) -> np.ndarray:
    """
    Decompose roll via NMF, compress each basis vector to target pitch range,
    reconstruct. Returns compressed piano roll of same shape.
    """
    # sklearn NMF expects (samples, features) → transpose to (time, pitch)
    X = roll.T  # (T, P)
    K = min(n_components, X.shape[0] - 1, X.shape[1] - 1, 64)

    model = NMF(n_components=K, max_iter=500, random_state=0, init="nndsvda")
    H = model.fit_transform(X)  # (T, K)
    W = model.components_       # (K, P)

    # Compress each basis vector
    W_compressed = np.stack([_compress_W_column(W[k], tgt_low, tgt_high) for k in range(K)])

    # Reconstruct
    X_compressed = H @ W_compressed   # (T, P)
    X_compressed = np.clip(X_compressed, 0.0, 1.0)
    return X_compressed.T             # (P, T)


# ── voice leading ─────────────────────────────────────────────────────────────

def _chord_events(notes: List[Dict], tol: float = 0.025) -> List[Dict]:
    """
    Group notes with onsets within `tol` seconds into chord events.
    Returns list of {onset, pitches, velocities}.
    """
    if not notes:
        return []
    sorted_notes = sorted(notes, key=lambda n: n["onset"])
    events: List[Dict] = []
    cur_onset = sorted_notes[0]["onset"]
    cur_pitches: List[int] = []
    cur_vels: List[int] = []

    for n in sorted_notes:
        if n["onset"] - cur_onset <= tol:
            cur_pitches.append(int(n["pitch"]))
            cur_vels.append(int(n.get("velocity", 80)))
        else:
            events.append({"onset": cur_onset, "pitches": cur_pitches, "velocities": cur_vels})
            cur_onset = n["onset"]
            cur_pitches = [int(n["pitch"])]
            cur_vels = [int(n.get("velocity", 80))]

    events.append({"onset": cur_onset, "pitches": cur_pitches, "velocities": cur_vels})
    return events


def _voice_lead_pair(prev: List[int], curr: List[int]) -> List[int]:
    """
    Given previous chord `prev` and current chord `curr` (both sorted),
    return `curr` reordered so total semitone movement is minimised.
    Uses Hungarian algorithm on |prev[i] - curr[j]| cost matrix.
    If sizes differ, return curr unchanged.
    """
    if len(prev) != len(curr) or len(curr) == 0:
        return curr

    cost = np.abs(np.array(prev)[:, None] - np.array(curr)[None, :]).astype(float)
    _, col_ind = linear_sum_assignment(cost)
    return [curr[j] for j in col_ind]


def apply_voice_leading(notes: List[Dict], tol: float = 0.025) -> List[Dict]:
    """
    Post-process notes: for each chord onset, reassign pitches from
    the same set to minimise voice movement from previous chord.
    """
    events = _chord_events(notes, tol)
    if not events:
        return notes

    # Build a lookup: (onset_rounded, pitch) → note index
    note_lookup: Dict[tuple, int] = {}
    for i, n in enumerate(notes):
        key = (round(float(n["onset"]), 3), int(n["pitch"]))
        note_lookup[key] = i

    prev_pitches: List[int] = []
    pitch_remap: Dict[int, int] = {}  # old_pitch_at_onset → new_pitch

    for ev in events:
        onset = ev["onset"]
        pitches = sorted(ev["pitches"])

        if not prev_pitches:
            prev_pitches = pitches
            continue

        reordered = _voice_lead_pair(prev_pitches, pitches)

        # Build remap for notes at this onset
        for old_p, new_p in zip(pitches, reordered):
            if old_p != new_p:
                pitch_remap[(round(onset, 3), old_p)] = new_p

        prev_pitches = reordered

    # Apply remap to note list
    result = []
    for n in notes:
        key = (round(float(n["onset"]), 3), int(n["pitch"]))
        if key in pitch_remap:
            new_pitch = pitch_remap[key]
            result.append({
                **n,
                "pitch": new_pitch,
                "pitch_name": librosa.midi_to_note(new_pitch),
            })
        else:
            result.append(n)

    result.sort(key=lambda n: (n["onset"], n["pitch"]))
    return result


# ── quantization ──────────────────────────────────────────────────────────────

def quantize_notes(notes: List[Dict], tempo: float, subdivisions: int = 16) -> List[Dict]:
    """
    Snap every note onset and offset to the nearest rhythmic grid position.
    subdivisions=16 means 16th-note grid (4 per beat).
    Notes whose snapped onset == snapped offset are extended by one grid unit.
    """
    if not notes or tempo <= 0:
        return notes

    grid = (60.0 / tempo) / (subdivisions / 4)  # seconds per grid unit

    def snap(t: float) -> float:
        return round(t / grid) * grid

    result = []
    for n in notes:
        onset_s  = snap(float(n["onset"]))
        offset_s = snap(float(n.get("offset", n["onset"] + grid)))
        if offset_s <= onset_s:
            offset_s = onset_s + grid
        result.append({
            **n,
            "onset":  round(onset_s,  4),
            "offset": round(offset_s, 4),
        })

    result.sort(key=lambda n: (n["onset"], n["pitch"]))
    return result


# ── scale & chord detection ───────────────────────────────────────────────────

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

_SCALES: Dict[str, List[int]] = {
    "major":            [0, 2, 4, 5, 7, 9, 11],
    "minor":            [0, 2, 3, 5, 7, 8, 10],
    "pentatonic_major": [0, 2, 4, 7, 9],
    "pentatonic_minor": [0, 3, 5, 7, 10],
    "blues":            [0, 3, 5, 6, 7, 10],
    "dorian":           [0, 2, 3, 5, 7, 9, 10],
    "mixolydian":       [0, 2, 4, 5, 7, 9, 10],
}

_CHORDS: Dict[str, List[int]] = {
    "maj":   [0, 4, 7],
    "min":   [0, 3, 7],
    "dom7":  [0, 4, 7, 10],
    "min7":  [0, 3, 7, 10],
    "maj7":  [0, 4, 7, 11],
    "dim":   [0, 3, 6],
    "aug":   [0, 4, 8],
    "sus2":  [0, 2, 7],
    "sus4":  [0, 5, 7],
    "hdim7": [0, 3, 6, 10],
}


def detect_scale(notes: List[Dict]) -> tuple:
    """Return (root, scale_name, scale_pcs) via weighted chroma dot-product."""
    chroma = np.zeros(12, dtype=np.float32)
    for n in notes:
        pc  = int(n["pitch"]) % 12
        dur = float(n.get("offset", n["onset"] + 0.25)) - float(n["onset"])
        vel = float(n.get("velocity", 80)) / 127.0
        chroma[pc] += max(dur, 0.0) * vel
    total = chroma.sum()
    if total > 0:
        chroma /= total

    best_score, best_root, best_name = -1.0, 0, "major"
    for name, intervals in _SCALES.items():
        mask = np.zeros(12, dtype=np.float32)
        for iv in intervals:
            mask[iv] = 1.0
        for root in range(12):
            score = float(np.dot(chroma, np.roll(mask, root)))
            if score > best_score:
                best_score, best_root, best_name = score, root, name

    scale_pcs = {(best_root + iv) % 12 for iv in _SCALES[best_name]}
    return best_root, best_name, scale_pcs


def _infer_chord_pcs(pitch_classes: set) -> set:
    """
    Return the chord pitch-class set that best fits the given pitch classes.
    Scores each (root, chord_type) by a weighted combination of coverage
    (how many chord tones are present) and precision (how many present pcs
    are chord tones). Falls back to accepting all pitch classes if no chord
    scores above a minimum threshold.
    """
    if not pitch_classes:
        return pitch_classes

    best_score, best_pcs = -1.0, pitch_classes
    for name, intervals in _CHORDS.items():
        for root in range(12):
            chord_pcs = {(root + i) % 12 for i in intervals}
            hits      = len(pitch_classes & chord_pcs)
            coverage  = hits / len(intervals)
            precision = hits / len(pitch_classes)
            score     = 0.6 * coverage + 0.4 * precision
            if score > best_score:
                best_score = score
                best_pcs   = chord_pcs

    # If best match is weak (< 40% coverage) keep all pcs — sparse beat
    if best_score < 0.4:
        return pitch_classes
    return best_pcs


def _group_by_beat(notes: List[Dict], tempo: float) -> Dict[int, List[int]]:
    """Return {beat_idx: [note_indices]} grouping by quarter-note beat."""
    beat_dur = 60.0 / tempo
    groups: Dict[int, List[int]] = {}
    for i, n in enumerate(notes):
        beat_idx = int(float(n["onset"]) / beat_dur)
        groups.setdefault(beat_idx, []).append(i)
    return groups


# ── layer 1: per-beat outlier removal ────────────────────────────────────────

def remove_beat_outliers(notes: List[Dict], tempo: float) -> List[Dict]:
    """
    Drop notes that are spatial outliers within their beat AND have no
    melodic continuation into adjacent beats.

    Outlier criterion (both must be true to drop):
      1. |pitch - beat_median| > max(1.5 * IQR, 18 semitones)
      2. No note within ±5 semitones exists in the ±2 surrounding beats
    Beats with fewer than 3 notes are left untouched.
    """
    if not notes or tempo <= 0:
        return notes

    groups = _group_by_beat(notes, tempo)
    drop: set = set()

    for beat_idx, idxs in groups.items():
        if len(idxs) < 3:
            continue

        pitches = np.array([int(notes[i]["pitch"]) for i in idxs], dtype=float)
        median  = float(np.median(pitches))
        q75, q25 = float(np.percentile(pitches, 75)), float(np.percentile(pitches, 25))
        threshold = max(1.5 * (q75 - q25), 18.0)

        for idx in idxs:
            pitch = int(notes[idx]["pitch"])
            if abs(pitch - median) <= threshold:
                continue

            # Check melodic continuation in ±2 beats
            continuation = any(
                abs(int(notes[j]["pitch"]) - pitch) <= 5
                for adj in range(beat_idx - 2, beat_idx + 3)
                if adj != beat_idx
                for j in groups.get(adj, [])
            )
            if not continuation:
                drop.add(idx)

    return [n for i, n in enumerate(notes) if i not in drop]


# ── layer 2: per-beat chord inference + chord-tone snapping ──────────────────

def snap_to_chord_tones(notes: List[Dict], tempo: float) -> List[Dict]:
    """
    Infer the most likely chord for each beat, then move any note whose
    pitch class falls outside that chord to the nearest chord tone
    (tries ±1 … ±6 semitones, prefers smallest move).
    This is strictly more precise than global scale snapping because it
    respects local harmonic context — an E♭ is wrong in C major globally
    but correct in a C7 or Cm chord locally.
    """
    if not notes or tempo <= 0:
        return notes

    groups = _group_by_beat(notes, tempo)

    # Infer chord pcs for each beat
    beat_chord: Dict[int, set] = {}
    for beat_idx, idxs in groups.items():
        pcs = {int(notes[i]["pitch"]) % 12 for i in idxs}
        beat_chord[beat_idx] = _infer_chord_pcs(pcs)

    beat_dur = 60.0 / tempo
    result   = []
    for n in notes:
        beat_idx  = int(float(n["onset"]) / beat_dur)
        chord_pcs = beat_chord.get(beat_idx)
        pitch     = int(n["pitch"])

        if not chord_pcs or pitch % 12 in chord_pcs:
            result.append(n)
            continue

        best = pitch
        for delta in range(1, 7):
            found = False
            for d in (delta, -delta):
                candidate = pitch + d
                if 21 <= candidate <= 108 and candidate % 12 in chord_pcs:
                    best  = candidate
                    found = True
                    break
            if found:
                break

        if best != pitch:
            result.append({**n, "pitch": best, "pitch_name": librosa.midi_to_note(best)})
        else:
            result.append(n)

    result.sort(key=lambda n: (n["onset"], n["pitch"]))
    return result


# ── per-layer feel offset ─────────────────────────────────────────────────────

def apply_feel_offset(notes: List[Dict], feel_seconds: float) -> List[Dict]:
    """
    Shift all note onsets and offsets by feel_seconds.

    Used to stamp generated notes (walking bass, pYIN fill) with the same
    timing character as their layer family measured from the original audio.
    Positive feel = behind the beat; negative = ahead of the beat.
    Values smaller than 2 ms are ignored (below perceptual threshold).
    """
    if not notes or abs(feel_seconds) < 0.002:
        return notes
    result = []
    for n in notes:
        onset  = round(max(0.0, float(n['onset'])  + feel_seconds), 4)
        offset = round(max(0.0, float(n.get('offset', float(n['onset']) + 0.25)) + feel_seconds), 4)
        if offset <= onset:
            offset = round(onset + 0.25, 4)
        result.append({**n, 'onset': onset, 'offset': offset})
    return result


# ── note-level range compression ──────────────────────────────────────────────

def compress_notes_to_range(
    notes: List[Dict],
    tgt_low: int,
    tgt_high: int,
) -> List[Dict]:
    """
    Shift each note's pitch by whole octaves until it falls within
    [tgt_low, tgt_high].  Preserves pitch class exactly (C stays C, etc.).
    Used by the pattern-extraction pipeline as a clean alternative to the
    NMF roll-based compression.
    """
    if tgt_low <= PIANO_LOW and tgt_high >= PIANO_HIGH:
        return notes   # full range — nothing to do

    result = []
    for n in notes:
        pitch = int(n["pitch"])
        while pitch < tgt_low:
            pitch += 12
        while pitch > tgt_high:
            pitch -= 12
        pitch = max(tgt_low, min(tgt_high, pitch))
        if pitch != int(n["pitch"]):
            n = {**n, "pitch": pitch, "pitch_name": librosa.midi_to_note(pitch)}
        result.append(n)
    return result


# ── post-processing bundle (used by the pattern-extraction pipeline) ───────────

def post_process_notes(
    notes: List[Dict],
    chord_map: Dict,
    tempo: float,
    key_root: int = 0,
    scale_name: str = "major",
    bass_lo: int = TARGET_LOW,
    bass_hi: int = BASS_MAX,
) -> List[Dict]:
    """
    Apply all musical post-processing steps to a flat note list:
      1. Voice leading (Hungarian assignment)
      2. Quantise to 16th-note grid
      3. Remove per-beat pitch outliers
      4. Snap off-chord notes to nearest chord tone
      5. Music-theory pass (contrary motion, cadential prep, parallel avoidance)
    """
    notes = apply_voice_leading(notes)
    notes = quantize_notes(notes, tempo)
    notes = remove_beat_outliers(notes, tempo)
    notes = snap_to_chord_tones(notes, tempo)
    try:
        from app.services.music_theory import apply_music_theory
        notes = apply_music_theory(
            notes,
            chord_map  = chord_map,
            tempo      = tempo,
            key_root   = key_root,
            scale_name = scale_name,
            bass_lo    = bass_lo,
            bass_hi    = bass_hi,
        )
    except Exception:
        pass
    return notes


# ── Phase 6b: hand-aware playability with comfort-aware assignment ───────────

# Hard constraints reflect a normal hand on a piano:
#   • 5 fingers per hand
#   • thumb-to-pinky spread ≤ 1 octave (12 semitones)
#   • 10 fingers total
HAND_MAX_NOTES        = 5
HAND_MAX_SPAN         = 12      # absolute upper bound (semitones)
HAND_TOTAL_NOTES      = 10
HAND_SPLIT_PITCH      = 60      # MIDI C4 — coarse fallback when ambiguous
HAND_COMFORT_SPAN     = 8       # comfort soft-limit — costs ramp up past this
HAND_PHRASE_WINDOW_S  = 0.25    # within this many seconds → still in the phrase


def _span(active: set, pitches: Dict[int, int]) -> int:
    """Span (in semitones) of the pitches of all notes currently in `active`."""
    if len(active) < 2:
        return 0
    ps = [pitches[i] for i in active]
    return max(ps) - min(ps)


def _thumb_pitch(hand: str, active: set, pitches: Dict[int, int]) -> int | None:
    """
    The thumb is the *inner* finger of a piano hand: the highest pitch the
    LEFT hand is playing, or the lowest pitch the RIGHT hand is playing.
    Returns None when the hand is idle (no active notes).
    """
    if not active:
        return None
    ps = [pitches[i] for i in active]
    return max(ps) if hand == "lh" else min(ps)


def _note_importances(
    notes:           List[Dict],
    beat_times:      Optional[List[float]],
    downbeat_times:  Optional[List[float]],
) -> Dict[int, float]:
    """
    Per-note importance score in roughly [0, 3].  Higher = harder to drop.

    Composed from:
        • velocity / 127            — louder notes are more important
        • +1.0 within 50 ms of a downbeat  (strong beat → harmonic anchor)
        • +0.5 within 100 ms of a downbeat
        • +0.3 within 50 ms of a beat (any kind)
        • +0.15 within 100 ms of a beat

    Falls back to velocity alone when beat data isn't supplied.
    """
    db = list(downbeat_times) if downbeat_times is not None else []
    bt = list(beat_times)     if beat_times     is not None else []
    out: Dict[int, float] = {}
    for i, n in enumerate(notes):
        score  = float(n.get("velocity", 80)) / 127.0
        onset  = float(n["onset"])
        if db:
            d = min(abs(onset - x) for x in db)
            if   d < 0.05: score += 1.0
            elif d < 0.10: score += 0.5
        if bt:
            d = min(abs(onset - x) for x in bt)
            if   d < 0.05: score += 0.3
            elif d < 0.10: score += 0.15
        out[i] = score
    return out


def _assignment_cost(
    pitch:            int,
    active:           set,
    pitches:          Dict[int, int],
    last_thumb:       int | None,
    t:                float,
    last_onset:       float,
    max_count:        int,
    max_span:         int,
    comfort_span:     int,
    phrase_window_s:  float,
) -> float:
    """
    Cost of adding `pitch` to this hand at time `t`.

    Returns:
        infinity if it would violate a hard constraint (caller should reject);
        otherwise a finite cost combining
            • spread cost — quadratic ramp above `comfort_span`
            • movement cost — distance from previous thumb position when idle
            • phrase coherence — discount when within phrase_window_s of the
              hand's most recent onset
    """
    if len(active) >= max_count:
        return float("inf")

    if active:
        # Hand is anchored by sustained notes — span is fixed by them + new pitch.
        ps = [pitches[i] for i in active] + [pitch]
        new_span = max(ps) - min(ps)
        if new_span > max_span:
            return float("inf")
        cost = max(0, new_span - comfort_span) ** 2 * 0.5
    else:
        # Hand idle — can move freely; cost grows with distance moved.
        if last_thumb is None:
            cost = 0.0
        else:
            cost = abs(pitch - last_thumb) * 0.02

    # Phrase coherence: a note immediately following another in this hand
    # gets a 50 % discount on its assignment cost.
    if last_onset >= 0 and (t - last_onset) < phrase_window_s:
        cost *= 0.5

    return cost


def _resolve_per_hand(
    active:        set,
    pitches:       Dict[int, int],
    importance:    Dict[int, float],
    offsets:       Dict[int, float],
    sorted_notes:  List[Dict],
    dropped:       set,
    new_idx:       int,
    t:             float,
    max_count:     int,
    max_span:      int,
) -> bool:
    """
    Drop/truncate notes in `active` until count + span constraints hold.
    Returns True if the *new* note (new_idx) survived, False if it was dropped.
    """
    while True:
        count_violation = len(active) > max_count
        span_violation  = _span(active, pitches) > max_span
        if not count_violation and not span_violation:
            return True

        # Pick the worst note based on importance, biased by violation type
        if span_violation:
            # For a span violation: prefer dropping outermost-and-least-important
            ordered = sorted(pitches[i] for i in active)
            median  = ordered[len(ordered) // 2]
            worst   = max(active, key=lambda i: (abs(pitches[i] - median), -importance[i]))
        else:
            # Pure count: drop the least important
            worst = min(active, key=lambda i: importance[i])

        active.discard(worst)
        if worst == new_idx:
            dropped.add(new_idx)
            return False
        offsets[worst] = t
        sorted_notes[worst]["offset"] = round(t, 4)


def enforce_hand_constraints(
    notes:                  List[Dict],
    beat_times:             Optional[List[float]] = None,
    downbeat_times:         Optional[List[float]] = None,
    max_notes_per_hand:     int = HAND_MAX_NOTES,
    max_span_per_hand:      int = HAND_MAX_SPAN,
    max_total_notes:        int = HAND_TOTAL_NOTES,
    split_pitch:            int = HAND_SPLIT_PITCH,
    comfort_span:           int = HAND_COMFORT_SPAN,
    phrase_window_s:        float = HAND_PHRASE_WINDOW_S,
) -> List[Dict]:
    """
    Stamp each note with a `hand` field ('lh' / 'rh') and drop/truncate notes
    that violate the physical hard constraints of two-hand piano playing.
    The assignment is *comfort-aware*: it minimises hand movement and only
    spreads when necessary, prefers staying on the same hand within a phrase,
    and uses beat / downbeat data to drop weak-beat notes first when forced.

    Hard constraints (unconditional, drops or truncates if violated):
        • ≤ max_notes_per_hand notes simultaneously per hand
        • ≤ max_span_per_hand semitones from low to high per hand
        • ≤ max_total_notes notes total at any instant

    Soft optimisation goals (factored into the assignment cost):
        • prefer hand movement over spread     (`movement` < `spread` cost)
        • prefer the previously-active hand within `phrase_window_s` of an
          onset in the same hand                (`phrase coherence`)
        • free hand naturally helps a stretched hand because its movement cost
          is small and the other hand's spread cost is large

    Drop priority (when forced):
        • span violation → outermost note in the offending hand, tie-broken by
          lowest importance
        • count violation → lowest-importance note in the affected set
        • total violation → lowest-importance note across both hands

    "Importance" = velocity + bonus for proximity to a downbeat (strong harmonic
    anchor) + smaller bonus for proximity to any beat.  Downbeat / beat times
    can be left None — then importance reduces to velocity alone.

    "Drop" means: if the offending note is the new onset, it never appears in
    the output.  If it's an already-active note, its offset is truncated to
    the current time — the note ends early but cleanly, no gap or glitch.

    Phase 6b — adds the comfort layer over 6a's hard constraints.  Phase 6c
    will add cross-hand overlap detection and merging; phase 6d will model the
    sustain pedal so per-hand finger budgets relax on sustained content.
    """
    if not notes:
        return notes

    sorted_notes = sorted(
        ({**n} for n in notes),
        key=lambda n: (float(n["onset"]), int(n["pitch"])),
    )
    pitches  = {i: int(n["pitch"])                                    for i, n in enumerate(sorted_notes)}
    offsets  = {i: float(n.get("offset", float(n["onset"]) + 0.25))  for i, n in enumerate(sorted_notes)}
    importance = _note_importances(sorted_notes, beat_times, downbeat_times)

    events: List[tuple] = []
    for i, n in enumerate(sorted_notes):
        events.append((float(n["onset"]), 1, i, "on"))
        events.append((offsets[i],         0, i, "off"))
    events.sort()

    dropped: set = set()
    lh_active: set = set()
    rh_active: set = set()
    # Hand state for movement modelling — thumb position is retained even
    # while the hand is idle so the next onset's movement cost reflects how
    # far the hand has to actually travel.
    lh_thumb: int | None = None
    rh_thumb: int | None = None
    lh_last_onset = -1e9
    rh_last_onset = -1e9

    for t, _typ_rank, idx, typ in events:
        if idx in dropped:
            continue

        if typ == "off":
            lh_active.discard(idx)
            rh_active.discard(idx)
            # Refresh thumb only if the hand still has active notes; otherwise
            # keep the last-known thumb position as the hand's "rest" anchor.
            new_thumb = _thumb_pitch("lh", lh_active, pitches)
            if new_thumb is not None: lh_thumb = new_thumb
            new_thumb = _thumb_pitch("rh", rh_active, pitches)
            if new_thumb is not None: rh_thumb = new_thumb
            continue

        if offsets[idx] <= t:
            continue

        # ── Cost-based hand assignment ─────────────────────────────────────
        p = pitches[idx]

        lh_cost = _assignment_cost(
            p, lh_active, pitches, lh_thumb, t, lh_last_onset,
            max_notes_per_hand, max_span_per_hand,
            comfort_span, phrase_window_s,
        )
        rh_cost = _assignment_cost(
            p, rh_active, pitches, rh_thumb, t, rh_last_onset,
            max_notes_per_hand, max_span_per_hand,
            comfort_span, phrase_window_s,
        )

        # Pitch bias: bass register prefers LH, treble prefers RH.  Small but
        # nonzero so that ambiguous-zone notes don't flip-flop randomly.
        if p < split_pitch:
            rh_cost += 3.0
        else:
            lh_cost += 3.0

        # Pick the lower-cost hand.  If both are inf, the note can't fit
        # anywhere without first dropping something else — we tentatively
        # assign to the better default (pitch-based) and let _resolve_per_hand
        # handle the drop using importance.
        if lh_cost == float("inf") and rh_cost == float("inf"):
            chosen = "lh" if p < split_pitch else "rh"
        else:
            chosen = "lh" if lh_cost <= rh_cost else "rh"

        active = lh_active if chosen == "lh" else rh_active
        active.add(idx)
        sorted_notes[idx]["hand"] = chosen

        survived = _resolve_per_hand(
            active, pitches, importance, offsets, sorted_notes,
            dropped, idx, t,
            max_notes_per_hand, max_span_per_hand,
        )
        if not survived:
            continue

        # Update hand state on a successful onset
        if chosen == "lh":
            lh_last_onset = t
            lh_thumb = _thumb_pitch("lh", lh_active, pitches)
        else:
            rh_last_onset = t
            rh_thumb = _thumb_pitch("rh", rh_active, pitches)

        # Resolve total-polyphony budget across both hands.
        while len(lh_active) + len(rh_active) > max_total_notes:
            both  = lh_active | rh_active
            worst = min(both, key=lambda i: importance[i])
            lh_active.discard(worst); rh_active.discard(worst)
            if worst == idx:
                dropped.add(idx)
                break
            offsets[worst] = t
            sorted_notes[worst]["offset"] = round(t, 4)

    # Emit only notes that survived (and whose truncated offset is still > onset).
    result: List[Dict] = []
    for i, n in enumerate(sorted_notes):
        if i in dropped:
            continue
        if float(n.get("offset", n["onset"])) <= float(n["onset"]):
            continue
        result.append(n)
    return result
