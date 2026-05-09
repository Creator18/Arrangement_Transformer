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
from typing import List, Dict, Any

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


# ── main entry point ──────────────────────────────────────────────────────────

def transform_nmf(
    notes: List[Dict],
    target_low: int = TARGET_LOW,
    target_high: int = TARGET_HIGH,
    n_components: int = 16,
    tempo: float = 120.0,
) -> tuple:
    """
    Full NMF + voice-leading transform.
    Returns (transformed_notes, scale_info dict).

    Pipeline:
      1. Build piano roll
      2. NMF compress pitch range to [target_low, target_high]
      3. Reconstruct note list
      4. Apply voice-leading smoothing
      5. Quantize to 16th-note grid
      6. Remove per-beat outliers (isolated high/low stragglers)
      7. Snap remaining off-chord notes to nearest chord tone
    """
    if not notes:
        return notes, {}

    duration    = max(float(n.get("offset", n["onset"] + 0.5)) for n in notes) + 1.0
    roll        = build_piano_roll(notes, duration)
    compressed  = nmf_compress(roll, target_low, target_high, n_components)
    transformed = piano_roll_to_notes(compressed)
    transformed = apply_voice_leading(transformed)
    transformed = quantize_notes(transformed, tempo)
    transformed = remove_beat_outliers(transformed, tempo)
    transformed = snap_to_chord_tones(transformed, tempo)

    root, scale_name, scale_pcs = detect_scale(transformed)
    scale_info = {
        "root":  _NOTE_NAMES[root],
        "scale": scale_name,
        "notes": sorted(_NOTE_NAMES[pc] for pc in scale_pcs),
    }
    return transformed, scale_info
