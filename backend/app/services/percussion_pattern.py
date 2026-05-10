"""
Percussion pattern extraction and walking bass generation.

Pipeline
--------
1.  Load raw audio, bandpass-filter into three frequency bands:
      kick  (20–250 Hz)   — kick drum fundamental
      snare (250–4 kHz)   — snare body / clap
      hihat (4–10 kHz)    — cymbal / hi-hat transients
2.  Run spectral-flux onset detection on each band.
3.  Quantise onset times to a strict 16th-note grid derived from the tempo
    so no note "just before beat 2" survives — they all snap to beat 2.
4.  Build binary measure-pattern vectors (16 slots per measure) for each band.
5.  Section segmentation via a novelty curve on the kick patterns, then
    mode-vote inside each section to find its dominant kick pattern.
6.  Walking-bass generation:
      •  At every kick-hit slot → anchor chord tone (beat-weight rules)
      •  Beats 1 and 3 always get a note even if the kick is silent
      •  Optional passing tones on the 8th-note between wide intervals
      •  Stepwise voice-leading throughout (minimise leap from prev note)
"""
from __future__ import annotations

import numpy as np
import librosa
from scipy.signal import butter, sosfilt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── constants ─────────────────────────────────────────────────────────────────
RESOLUTION = 16          # 16th-note grid: 16 slots per 4/4 measure
SR_DEFAULT = 22050

BANDS: Dict[str, Tuple[float, float]] = {
    "kick":  (20.0,   250.0),
    "snare": (250.0,  4000.0),
    "hihat": (4000.0, 10000.0),
}

BASS_LO_DEFAULT = 36     # C2
BASS_HI_DEFAULT = 55     # G3

# Interval classes from chord root, ordered by bass preference
# root → 5th → maj3 / min3 → min7 → 4th → maj2 → maj6
_INTERVAL_PRIORITY: List[int] = [0, 7, 4, 3, 10, 5, 2, 9]

# Velocity by beat slot
_SLOT_VEL = {0: 92, 4: 68, 8: 78, 12: 65}


# ── audio helpers ─────────────────────────────────────────────────────────────

def _bandpass(y: np.ndarray, lo_hz: float, hi_hz: float, sr: int) -> np.ndarray:
    """4th-order Butterworth bandpass; clamps cutoffs safely away from Nyquist."""
    nyq = sr / 2.0
    lo  = max(lo_hz / nyq, 1e-4)
    hi  = min(hi_hz / nyq, 1.0 - 1e-4)
    if lo >= hi:
        return y.copy()
    sos = butter(4, [lo, hi], btype="band", output="sos")
    return sosfilt(sos, y).astype(np.float32)


# ── step 1-2: onset extraction ────────────────────────────────────────────────

def extract_percussion_onsets(
    audio_path: Path,
    sr: int = SR_DEFAULT,
) -> Dict[str, List[float]]:
    """
    Load audio and return onset times (seconds) for each frequency band.
    Returns {"kick": [...], "snare": [...], "hihat": [...]}.
    Returns empty lists on any error so the pipeline never crashes.
    """
    try:
        y, sr = librosa.load(str(audio_path), sr=sr, mono=True)
    except Exception:
        return {k: [] for k in BANDS}

    result: Dict[str, List[float]] = {}
    for band_name, (lo_hz, hi_hz) in BANDS.items():
        try:
            y_band = _bandpass(y, lo_hz, hi_hz, sr)
            frames = librosa.onset.onset_detect(
                y=y_band, sr=sr, units="time",
                hop_length=512,
                backtrack=True,
                pre_max=3, post_max=3,
                pre_avg=3, post_avg=5,
                delta=0.05, wait=4,
            )
            result[band_name] = frames.tolist()
        except Exception:
            result[band_name] = []
    return result


# ── step 3-4: grid quantisation & measure patterns ───────────────────────────

def _times_to_grid(
    times: List[float],
    tempo: float,
    resolution: int = RESOLUTION,
) -> List[Tuple[int, int]]:
    """Convert raw seconds to (measure_idx, slot_0_to_15) on the 16th-note grid."""
    if tempo <= 0:
        return []
    slot_dur = (60.0 / tempo) * 4 / resolution
    out: List[Tuple[int, int]] = []
    for t in times:
        if t < 0:
            continue
        slot_abs = int(round(t / slot_dur))
        out.append((slot_abs // resolution, slot_abs % resolution))
    return out


def build_measure_patterns(
    onsets: Dict[str, List[float]],
    n_measures: int,
    tempo: float,
    resolution: int = RESOLUTION,
) -> Dict[str, List[List[int]]]:
    """
    Return {"kick": [[0,1,0,…]×n_measures], "snare": …, "hihat": …}
    where each inner list has `resolution` binary elements (1 = hit).
    """
    patterns: Dict[str, List[List[int]]] = {
        band: [[0] * resolution for _ in range(n_measures)]
        for band in BANDS
    }
    for band_name, times in onsets.items():
        if band_name not in patterns:
            continue
        for measure, slot in _times_to_grid(times, tempo, resolution):
            if 0 <= measure < n_measures and 0 <= slot < resolution:
                patterns[band_name][measure][slot] = 1
    return patterns


# ── step 5: section segmentation + mode vote ──────────────────────────────────

def _pattern_mode(patterns: List[List[int]]) -> List[int]:
    """Return the most frequent pattern (mode) from a list of measure patterns."""
    if not patterns:
        return [0] * RESOLUTION
    counter: Dict[tuple, int] = {}
    for p in patterns:
        key = tuple(p)
        counter[key] = counter.get(key, 0) + 1
    return list(max(counter, key=counter.__getitem__))


def _detect_pattern_sections(
    kick_patterns: List[List[int]],
    min_section: int = 4,
) -> List[Tuple[int, int]]:
    """
    Novelty-curve section detection on the kick pattern matrix.
    Mirrors the approach in transcription._detect_sections but operates on
    16-bit binary rows instead of chroma.
    Returns [(start_measure, end_measure), …].
    """
    n = len(kick_patterns)
    if n <= min_section * 2:
        return [(0, n)]

    P = np.array(kick_patterns, dtype=float)   # (n, RESOLUTION)
    k = min(4, n // 4)
    novelty = np.zeros(n, dtype=float)
    for i in range(k, n - k):
        before = P[max(0, i - k):i].mean(axis=0)
        after  = P[i:min(n, i + k)].mean(axis=0)
        nb, na = np.linalg.norm(before), np.linalg.norm(after)
        if nb > 0 and na > 0:
            novelty[i] = 1.0 - float(np.dot(before, after) / (nb * na))

    if n > 2 * k:
        threshold = max(0.12, float(np.percentile(novelty[k:n - k], 65)))
    else:
        threshold = 1.0   # no valid window → single section

    boundaries = [0]
    i = k
    while i < n - k:
        window = novelty[max(0, i - 2):min(n, i + 3)]
        if novelty[i] >= threshold and float(novelty[i]) == float(window.max()):
            if i - boundaries[-1] >= min_section:
                boundaries.append(i)
                i += min_section
                continue
        i += 1
    boundaries.append(n)
    return [(boundaries[j], boundaries[j + 1]) for j in range(len(boundaries) - 1)]


def find_section_kick_patterns(
    kick_patterns: List[List[int]],
) -> Dict[int, List[int]]:
    """
    Segment kick patterns into sections and return the mode (dominant) pattern
    for each measure within its section.
    Returns {measure_idx: dominant_kick_pattern_16_elements}.
    Measures not covered get the global mode as fallback.
    """
    n = len(kick_patterns)
    if n == 0:
        return {}

    sections = _detect_pattern_sections(kick_patterns)
    result: Dict[int, List[int]] = {}
    for sec_start, sec_end in sections:
        mode = _pattern_mode(kick_patterns[sec_start:sec_end])
        for mi in range(sec_start, sec_end):
            result[mi] = mode

    global_mode = _pattern_mode(kick_patterns)
    for mi in range(n):
        result.setdefault(mi, global_mode)
    return result


# ── step 6: walking bass generation ──────────────────────────────────────────

def _chord_tones_in_range(
    root_pc: int,
    chord_pcs: set,
    lo: int,
    hi: int,
) -> List[int]:
    """
    Return all MIDI pitches in [lo, hi] that belong to chord_pcs,
    sorted by interval priority (root first, then 5th, then 3rd, …).
    """
    scored: List[Tuple[int, int]] = []
    for p in range(lo, hi + 1):
        if p % 12 in chord_pcs:
            interval = (p % 12 - root_pc) % 12
            pri = _INTERVAL_PRIORITY.index(interval) if interval in _INTERVAL_PRIORITY else 99
            scored.append((pri, p))
    scored.sort()
    return [p for _, p in scored]


def _pick_anchor(
    root_pc: int,
    chord_pcs: set,
    lo: int,
    hi: int,
    slot: int,
    prev_pitch: Optional[int],
    next_root_pc: Optional[int],
) -> int:
    """
    Select the best chord tone for anchor note at `slot` within a measure.

    Beat-weight rules (4/4, resolution=16):
      slot  0  → beat 1: root (bass downbeat)
      slot  8  → beat 3: 5th (or root if no 5th available)
      slot  4  → beat 2: closest chord tone to prev, prefer 3rd
      slot 12  → beat 4: approach tone toward next chord root
      other    → stepwise to nearest chord tone from prev_pitch
    """
    tones = _chord_tones_in_range(root_pc, chord_pcs, lo, hi)
    if not tones:
        mid = (lo + hi) // 2
        return min(range(lo, hi + 1), key=lambda p: abs(p - (prev_pitch or mid)))

    mid_range = (lo + hi) // 2

    if slot == 0:
        # Beat 1: root — lowest root in range (strong downbeat)
        roots = [p for p in tones if p % 12 == root_pc]
        return roots[0] if roots else tones[0]

    elif slot == 8:
        # Beat 3: perfect 5th (or root as fallback)
        fifth_pc = (root_pc + 7) % 12
        fifths = [p for p in tones if p % 12 == fifth_pc]
        if fifths:
            ref = prev_pitch if prev_pitch is not None else mid_range
            return min(fifths, key=lambda p: abs(p - ref))
        roots = [p for p in tones if p % 12 == root_pc]
        return roots[0] if roots else tones[0]

    elif slot == 12 and next_root_pc is not None:
        # Beat 4: approach note — half/whole step below next chord root
        for p in range(lo, hi + 1):
            if p % 12 == next_root_pc:
                # Try semitone below, then whole-tone below
                for delta in (1, 2):
                    approach = p - delta
                    if lo <= approach <= hi:
                        return approach
                break
        # Fallback: closest chord tone to prev
        ref = prev_pitch if prev_pitch is not None else mid_range
        return min(tones, key=lambda p: abs(p - ref))

    else:
        # Off-beat / beat 2: stepwise voice leading from prev
        ref = prev_pitch if prev_pitch is not None else mid_range
        return min(tones, key=lambda p: abs(p - ref))


def generate_walking_bass(
    chord_map: Dict[int, tuple],
    measure_kick_patterns: Dict[int, List[int]],
    tempo: float,
    lo: int = BASS_LO_DEFAULT,
    hi: int = BASS_HI_DEFAULT,
    resolution: int = RESOLUTION,
    add_passing_tones: bool = True,
) -> List[Dict]:
    """
    Generate a walking bass line driven by the kick drum pattern and chord map.

    For each measure:
      1. Collect kick-hit slots from the section's dominant pattern.
      2. Always include slots 0 (beat 1) and 8 (beat 3) as mandatory anchors.
      3. At each anchor slot, call _pick_anchor() to choose the chord tone.
      4. Optionally inject a passing tone halfway between wide leaps.
      5. Maintain running prev_pitch for smooth voice leading across measures.

    Returns a list of note dicts ready to merge with the upper voices.
    """
    if tempo <= 0 or not chord_map:
        return []

    measure_dur = (60.0 / tempo) * 4
    slot_dur    = measure_dur / resolution

    all_measures = set(chord_map.keys()) | set(measure_kick_patterns.keys())
    if not all_measures:
        return []
    max_measure = max(all_measures)

    notes: List[Dict] = []
    prev_pitch: Optional[int] = None

    for mi in range(max_measure + 1):
        # Chord for this measure (fall back to adjacent measure)
        chord_info = (chord_map.get(mi)
                      or chord_map.get(mi - 1)
                      or chord_map.get(mi + 1))
        if chord_info is None:
            continue

        root_pc, chord_pcs = chord_info

        # Next chord's root (for beat-4 approach notes)
        next_chord  = chord_map.get(mi + 1) or chord_map.get(mi + 2)
        next_root   = next_chord[0] if next_chord else None

        # Kick pattern for this measure; default = 4-on-the-floor if nothing
        kick = measure_kick_patterns.get(mi)
        if not kick or not any(kick):
            kick = [1 if s % 4 == 0 else 0 for s in range(resolution)]

        # Hit slots: actual kick + mandatory beats 1 & 3
        hit_slots = sorted({s for s, v in enumerate(kick) if v} | {0, 8})

        # Resolve each anchor pitch
        anchor: Dict[int, int] = {}
        for slot in hit_slots:
            p = _pick_anchor(root_pc, chord_pcs, lo, hi, slot,
                             prev_pitch, next_root)
            anchor[slot] = p
            prev_pitch = p

        # Emit note events
        for idx, slot in enumerate(hit_slots):
            pitch = anchor[slot]
            t0    = mi * measure_dur + slot * slot_dur

            # Duration = gap to next hit (or end of measure)
            if idx + 1 < len(hit_slots):
                next_slot = hit_slots[idx + 1]
            else:
                next_slot = resolution
            dur = (next_slot - slot) * slot_dur

            # Velocity by beat position
            vel = _SLOT_VEL.get(slot, 60)

            # Optional passing tone halfway through wide gaps
            if add_passing_tones and (next_slot - slot) >= 4:
                next_p = anchor.get(next_slot)
                if next_p is not None and abs(pitch - next_p) >= 3:
                    pass_slot = slot + (next_slot - slot) // 2
                    if pass_slot not in anchor:
                        # Chromatic midpoint, nudged to nearest chord tone
                        mid_p = int(round((pitch + next_p) / 2))
                        tones = _chord_tones_in_range(root_pc, chord_pcs, lo, hi)
                        if tones:
                            mid_p = min(tones, key=lambda p: abs(p - mid_p))
                        mid_p = max(lo, min(hi, mid_p))
                        pt0   = mi * measure_dur + pass_slot * slot_dur
                        pt_dur = (next_slot - pass_slot) * slot_dur
                        notes.append({
                            "pitch":      mid_p,
                            "pitch_name": librosa.midi_to_note(mid_p),
                            "onset":      round(pt0, 4),
                            "offset":     round(pt0 + pt_dur * 0.82, 4),
                            "velocity":   max(40, vel - 20),
                        })

            notes.append({
                "pitch":      pitch,
                "pitch_name": librosa.midi_to_note(pitch),
                "onset":      round(t0, 4),
                "offset":     round(t0 + dur * 0.88, 4),
                "velocity":   vel,
            })

    notes.sort(key=lambda n: (n["onset"], n["pitch"]))
    return notes
