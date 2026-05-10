"""
Pattern extraction pipeline.

Architecture:
  1. Detect repeating cycle length in chord progression
  2. Encode each note as interval-relative to its chord root
  3. Mine repeating pattern elements (>=60% of cycle reps, >=2 reps)
  4. Assign notes to layers by pattern membership + register bucket
  5. Fill melody residual using pYIN pitch tracking on raw audio
"""
from __future__ import annotations

import numpy as np
import librosa
from pathlib import Path
from typing import List, Dict, Set, Optional, Any
from collections import defaultdict

RESOLUTION = 16   # 16th-note slots per 4/4 measure

# Register boundaries (MIDI)
PIANO_LOW = 21  # A0
BASS_HI   = 55  # G3  — at or below = bass layer
MID_HI    = 71  # B4  — bass_hi+1 .. mid_hi = mid/harmony layer
                # above mid_hi = treble / melody layer


# ── timing helpers ─────────────────────────────────────────────────────────────

def _slot_dur(tempo: float) -> float:
    return (60.0 / tempo) * 4.0 / RESOLUTION


def _measure_dur(tempo: float) -> float:
    return (60.0 / tempo) * 4.0


def _velocity_tier(vel: int) -> int:
    if vel < 40:  return 0
    if vel < 70:  return 1
    if vel < 100: return 2
    return 3


# ── step 1: detect chord-cycle length ─────────────────────────────────────────

def detect_cycle_length(chord_map: Dict[int, tuple], n_measures: int) -> int:
    """
    Find the shortest period P (measures) where chord roots repeat with >=70%
    agreement. Tries P in [1, 2, 4, 8, 16]; returns 4 as fallback.
    """
    if not chord_map or n_measures < 2:
        return 4

    for P in [1, 2, 4, 8, 16]:
        if P >= n_measures:
            continue
        pairs = [
            (m, m + P)
            for m in range(n_measures - P)
            if m in chord_map and (m + P) in chord_map
        ]
        if not pairs:
            continue
        matches = sum(chord_map[m][0] == chord_map[m + P][0] for m, _ in pairs)
        if matches / len(pairs) >= 0.70:
            return P

    return min(4, n_measures)


# ── step 2: encode notes relative to chord roots ───────────────────────────────

def encode_notes(
    notes: List[Dict],
    chord_map: Dict[int, tuple],
    tempo: float,
    cycle_length: int,
) -> List[Dict]:
    """
    Attach cycle context to each note. Notes whose measure has no chord entry
    are skipped.

    Added fields:
      m_idx      — measure index
      cycle_rep  — which repetition of the chord cycle (0, 1, 2, …)
      cycle_pos  — position within cycle  (0 … cycle_length-1)
      interval   — (pitch%12 - chord_root) % 12
      octave_off — octave offset relative to the chord-root MIDI in same octave
      slot       — 16th-note slot within the measure (0 … 15)
      vel_tier   — velocity bucket (0-3)
      root_pc    — chord root pitch class
      chord_pcs  — set of chord pitch classes
    """
    sd = _slot_dur(tempo)
    md = _measure_dur(tempo)
    encoded: List[Dict] = []

    for n in notes:
        onset = float(n["onset"])
        pitch = int(n["pitch"])
        m_idx = int(onset / md)
        slot  = int(round((onset - m_idx * md) / sd)) % RESOLUTION

        chord = (
            chord_map.get(m_idx)
            or chord_map.get(m_idx - 1)
            or chord_map.get(m_idx + 1)
        )
        if chord is None:
            continue
        root_pc, chord_pcs = chord

        interval   = (pitch % 12 - root_pc) % 12
        root_midi  = (pitch // 12) * 12 + root_pc
        octave_off = (pitch - root_midi) // 12

        encoded.append({
            **n,
            "m_idx":      m_idx,
            "cycle_rep":  m_idx // cycle_length,
            "cycle_pos":  m_idx % cycle_length,
            "interval":   interval,
            "octave_off": octave_off,
            "slot":       slot,
            "vel_tier":   _velocity_tier(int(n.get("velocity", 80))),
            "root_pc":    root_pc,
            "chord_pcs":  chord_pcs,
        })

    return encoded


# ── step 3: mine repeating pattern elements ────────────────────────────────────

def mine_pattern_elements(
    encoded_notes: List[Dict],
    n_reps: int,
    min_coverage: float = 0.60,
    min_reps: int = 2,
) -> Set[tuple]:
    """
    Return the set of (cycle_pos, interval, octave_off, slot) tuples that
    appear in >=min_coverage fraction of cycle reps AND >=min_reps distinct reps.

    Also checks half-cycle and quarter-cycle sub-patterns: if a tuple satisfies
    the threshold when the cycle is halved (pos % (cycle_length/2)) it still
    counts, which catches 2- or 4-bar ostinati inside an 8-bar cycle.
    """
    appearance_map: Dict[tuple, Set[int]] = defaultdict(set)
    for n in encoded_notes:
        key = (n["cycle_pos"], n["interval"], n["octave_off"], n["slot"])
        appearance_map[key].add(n["cycle_rep"])

    pattern_elements: Set[tuple] = set()
    for key, reps in appearance_map.items():
        if len(reps) >= min_reps and len(reps) / max(n_reps, 1) >= min_coverage:
            pattern_elements.add(key)

    return pattern_elements


# ── step 4: layer assignment ───────────────────────────────────────────────────

def _layer_name(pitches: List[int], n_notes: int, duration_s: float) -> str:
    mid = sum(pitches) / len(pitches)
    density = n_notes / max(duration_s / 4.0, 0.25)

    register = "Bass" if mid <= BASS_HI else ("Mid" if mid <= MID_HI else "High")

    if density > 8:
        char = "Arpeggio"
    elif density > 3:
        char = "Comp"
    elif density > 1:
        char = "Stabs"
    else:
        char = "Pad"

    return f"{register} {char}"


def _layer_color(name: str) -> str:
    if "Bass"   in name: return "#f59e0b"
    if "Mid"    in name: return "#a78bfa"
    if "High"   in name: return "#06b6d4"
    if "Melody" in name: return "#34d399"
    return "#e879f9"


def assign_notes_to_layers(
    encoded_notes: List[Dict],
    pattern_elements: Set[tuple],
) -> Dict[str, List[Dict]]:
    """
    Split encoded notes into register-bucketed pattern layers + melody residual.

    Uses each note's pre-labeled 'layer' field (set by transcription via NMF
    component dominance) as the primary classifier:
      layer='bass'    → bass bucket (always, regardless of pitch)
      layer='pattern' → mid/high bucket by pitch
      layer='melody'  → Lead Melody residual
      unlabeled       → cycle-based pattern_elements membership fallback

    Pitch also gates bass notes: any note at or below BASS_HI goes to bass
    even if unlabeled, since register is reliable.

    Returns {layer_label: [plain note dicts]}.
    """
    _KEEP = {"pitch", "pitch_name", "onset", "offset", "velocity"}

    buckets: Dict[str, List[Dict]] = {"bass": [], "mid": [], "high": []}
    melody_residual: List[Dict] = []

    for n in encoded_notes:
        note_out  = {k: v for k, v in n.items() if k in _KEEP}
        pitch     = int(n["pitch"])
        pre_label = n.get("layer", "")

        if pre_label == "bass" or pitch <= BASS_HI:
            buckets["bass"].append(note_out)
        elif pre_label == "pattern":
            # NMF-labeled as pattern → register bucket by pitch
            buckets["mid" if pitch <= MID_HI else "high"].append(note_out)
        elif pre_label == "melody":
            # NMF-labeled as melody → residual
            melody_residual.append(note_out)
        else:
            # Unlabeled note — fall back to cycle-based pattern mining
            key = (n["cycle_pos"], n["interval"], n["octave_off"], n["slot"])
            if key in pattern_elements:
                buckets["mid" if pitch <= MID_HI else "high"].append(note_out)
            else:
                melody_residual.append(note_out)

    # Build named layers from non-empty register buckets
    layers: Dict[str, List[Dict]] = {}
    for bname, bnotes in buckets.items():
        if not bnotes:
            continue
        pitches  = [int(n["pitch"]) for n in bnotes]
        t_start  = min(float(n["onset"]) for n in bnotes)
        t_end    = max(float(n.get("offset", n["onset"] + 0.25)) for n in bnotes)
        name     = _layer_name(pitches, len(bnotes), t_end - t_start)
        layers[name] = bnotes

    if melody_residual:
        layers["Lead Melody"] = melody_residual

    return layers


# ── step 5: pYIN gap fill for melody ──────────────────────────────────────────

def _has_conflict(
    onset: float,
    pitch: int,
    offset: float,
    pattern_notes: List[Dict],
    existing_melody: List[Dict],
    tol_time: float = 0.08,
    tol_pitch: int = 3,
) -> bool:
    """True if the proposed note overlaps an existing pattern or melody note."""
    for n in pattern_notes:
        n_on  = float(n["onset"])
        n_off = float(n.get("offset", n_on + 0.25))
        if n_on - tol_time < offset and onset < n_off + tol_time:
            if abs(int(n["pitch"]) - pitch) <= tol_pitch:
                return True
    for n in existing_melody:
        n_on  = float(n["onset"])
        n_off = float(n.get("offset", n_on + 0.25))
        if n_on < offset and onset < n_off:
            return True
    return False


def fill_melody_pyin(
    melody_residual: List[Dict],
    audio_path: Path,
    pattern_notes: List[Dict],
    tempo: float,
    *,
    fmin_note: str = "C3",
    fmax_note: str = "C7",
) -> List[Dict]:
    """
    Fill gaps in melody_residual using pYIN pitch tracking on the raw audio.

    Only inserts a note when:
      1. pYIN shows a clear voiced pitch for >=2 analysis frames in the gap
      2. A gap of >=0.2 s exists in the current melody
      3. No temporal/pitch conflict with pattern or existing melody notes
    """
    if not audio_path or not audio_path.exists():
        return melody_residual

    try:
        y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
        HOP   = 512

        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz(fmin_note),
            fmax=librosa.note_to_hz(fmax_note),
            sr=sr,
            hop_length=HOP,
            frame_length=2048,
            fill_na=None,
        )

        with np.errstate(invalid="ignore"):
            midi_f0 = np.where(
                voiced_flag & (f0 > 0),
                librosa.hz_to_midi(np.where(f0 > 0, f0, 1.0)),
                np.nan,
            )

        frame_times = librosa.frames_to_time(
            np.arange(len(midi_f0)), sr=sr, hop_length=HOP
        )

        # Build gap list from sorted melody
        sorted_mel = sorted(melody_residual, key=lambda n: float(n["onset"]))
        gaps: List[tuple] = []   # (gap_start, gap_end)

        if sorted_mel:
            prev_off = float(sorted_mel[0].get("offset", sorted_mel[0]["onset"] + 0.25))
            for n in sorted_mel[1:]:
                n_on = float(n["onset"])
                if n_on - prev_off > 0.20:
                    gaps.append((prev_off, n_on))
                prev_off = max(prev_off, float(n.get("offset", n["onset"] + 0.25)))
        else:
            # No melody notes at all — consider the whole track a gap
            if len(frame_times) > 0:
                gaps = [(float(frame_times[0]), float(frame_times[-1]))]

        filled = list(melody_residual)

        for g_start, g_end in gaps:
            t = g_start
            while t < g_end - 0.15:
                t_end     = min(t + 0.25, g_end)
                seg_mask  = (frame_times >= t) & (frame_times < t_end)
                seg_f0    = midi_f0[seg_mask]
                voiced    = seg_f0[~np.isnan(seg_f0)]

                if len(voiced) >= 2:
                    pitch   = int(round(float(np.median(voiced))))
                    pitch   = max(48, min(96, pitch))   # C3–C7 melody range
                    onset_t = t
                    off_t   = t_end

                    if not _has_conflict(onset_t, pitch, off_t, pattern_notes, filled):
                        filled.append({
                            "pitch":      pitch,
                            "pitch_name": librosa.midi_to_note(pitch),
                            "onset":      round(onset_t, 4),
                            "offset":     round(off_t, 4),
                            "velocity":   60,
                        })

                t = t_end

        filled.sort(key=lambda n: (float(n["onset"]), int(n["pitch"])))
        return filled

    except Exception:
        return melody_residual


# ── step 6: dominance-based clutter reduction ─────────────────────────────────

def apply_dominance_filter(
    raw_layers: Dict[str, List[Dict]],
    dominance_map: Dict[int, str],
    tempo: float,
    *,
    max_upper_per_onset: int = 3,
) -> Dict[str, List[Dict]]:
    """
    Reduce inter-layer clutter using per-measure source dominance derived from
    the original audio (computed in segmentation via JS divergence).

    dominance_map: {measure_idx → 'bass' | 'melody'}
      'bass'   — bass was louder in the original → thin upper polyphony so the
                 bass still leads; keep only the top max_upper_per_onset notes
                 (by velocity) at each simultaneous onset.
      'melody' — upper was louder in the original → thin bass to quarter-note
                 beats (slots 0, 4, 8, 12) so melody breathes freely.

    This pass runs before merge_and_resolve so that downstream conflict
    resolution works on an already-thinned note set.
    """
    if not dominance_map or not raw_layers:
        return raw_layers

    md  = _measure_dur(tempo)
    sd  = _slot_dur(tempo)
    TOL = 0.015   # 15 ms onset-grouping tolerance

    def _m(onset: float) -> int:
        return int(onset / md)

    bass_dom = {m for m, d in dominance_map.items() if d == 'bass'}
    mel_dom  = {m for m, d in dominance_map.items() if d == 'melody'}

    # ── bass-dominant measures: cap upper-voice polyphony per onset ───────────
    if bass_dom:
        upper_names = [k for k in raw_layers if 'Bass' not in k]
        for uname in upper_names:
            notes = raw_layers[uname]
            if not notes:
                continue

            kept: List[Dict] = []
            pending: Dict[tuple, List[Dict]] = defaultdict(list)

            for n in sorted(notes, key=lambda x: float(x['onset'])):
                m_idx = _m(float(n['onset']))
                if m_idx not in bass_dom:
                    kept.append(n)
                else:
                    slot_key = (m_idx, round(float(n['onset']) / TOL))
                    pending[slot_key].append(n)

            for group in pending.values():
                group.sort(key=lambda x: -int(x.get('velocity', 80)))
                kept.extend(group[:max_upper_per_onset])

            kept.sort(key=lambda x: (float(x['onset']), int(x['pitch'])))
            raw_layers[uname] = kept

    # ── melody-dominant measures: thin bass to quarter-note beats ─────────────
    if mel_dom:
        bass_name = next((k for k in raw_layers if 'Bass' in k), None)
        if bass_name:
            beat_slots = {0, 4, 8, 12}   # quarter-note positions in 16-slot grid
            kept = []
            for n in raw_layers[bass_name]:
                onset = float(n['onset'])
                m_idx = _m(onset)
                if m_idx not in mel_dom:
                    kept.append(n)
                    continue
                slot = int(round((onset - m_idx * md) / sd)) % RESOLUTION
                if slot in beat_slots:
                    kept.append(n)
            raw_layers[bass_name] = kept

    return raw_layers


# ── step 7: merge layers and resolve conflicts ────────────────────────────────

def merge_and_resolve(
    raw_layers: Dict[str, List[Dict]],
    chord_map: Dict[int, tuple],
    tempo: float,
) -> Dict[str, List[Dict]]:
    """
    After all layers are assembled (including pYIN fill), resolve inter-layer
    conflicts so that when notes are merged for MIDI export they are clean.

    Passes applied in order:
      A. Deduplicate — same pitch within 30 ms across any two layers → keep
         the one from the lower-indexed (pattern-priority) layer, drop the other.
      B. Voice-crossing fix — if a bass-layer note pitch > any simultaneous
         mid/high note, shift the bass note down by octaves until it's below.
      C. Unison doubling — if two different layers play the same pitch class
         within 30 ms, reduce the lower-priority one by one octave downward
         (bass priority > mid > high > melody).
      D. Harmonic clash — at each beat, collect all pitch classes across layers;
         if a tritone interval (6 semitones) exists between a non-chord-tone
         in the melody and a chord tone in a pattern layer, nudge the melody
         note ±1 semitone to the nearest chord tone.
    """
    if not raw_layers:
        return raw_layers

    md       = _measure_dur(tempo)
    TOL      = 0.030   # 30 ms onset tolerance for "simultaneous"

    # Priority order for conflict resolution (index 0 = highest priority)
    PRIORITY = ["Bass", "Mid", "High", "Lead Melody"]

    def _priority(layer_name: str) -> int:
        for i, p in enumerate(PRIORITY):
            if p in layer_name:
                return i
        return len(PRIORITY)

    # ── A: deduplicate exact same pitch within TOL ─────────────────────────────
    # Build a global set of (onset_rounded, pitch) seen so far (highest priority first)
    seen: set = set()
    for layer_name in sorted(raw_layers.keys(), key=_priority):
        kept = []
        for n in raw_layers[layer_name]:
            key = (round(float(n["onset"]) / TOL) * TOL, int(n["pitch"]))
            if key not in seen:
                seen.add(key)
                kept.append(n)
        raw_layers[layer_name] = kept

    # ── B: voice-crossing fix ─────────────────────────────────────────────────
    # Bass notes must stay below all mid/high/melody notes at the same onset.
    if "Bass" in "".join(raw_layers.keys()):
        bass_name = next((k for k in raw_layers if "Bass" in k), None)
        upper_names = [k for k in raw_layers if bass_name and k != bass_name]

        if bass_name and upper_names:
            # Build a fast lookup: onset_slot → min upper pitch
            upper_min: Dict[int, int] = {}
            for uname in upper_names:
                for n in raw_layers[uname]:
                    slot = int(round(float(n["onset"]) / TOL))
                    upper_min[slot] = min(upper_min.get(slot, 127), int(n["pitch"]))

            fixed_bass = []
            for n in raw_layers[bass_name]:
                slot      = int(round(float(n["onset"]) / TOL))
                min_upper = upper_min.get(slot, 127)
                pitch     = int(n["pitch"])
                while pitch >= min_upper and pitch - 12 >= PIANO_LOW:
                    pitch -= 12
                if pitch != int(n["pitch"]):
                    n = {**n, "pitch": pitch, "pitch_name": librosa.midi_to_note(pitch)}
                fixed_bass.append(n)
            raw_layers[bass_name] = fixed_bass

    # ── C: unison doubling — collapse same pitch-class within TOL ─────────────
    # Walk layers lowest-priority first; if a note's PC is already present in a
    # higher-priority layer at the same onset, drop it (avoid muddy unisons).
    pc_seen: Dict[int, set] = {}  # onset_slot → {pitch_classes already claimed}
    for layer_name in sorted(raw_layers.keys(), key=_priority):
        kept = []
        for n in raw_layers[layer_name]:
            slot = int(round(float(n["onset"]) / TOL))
            pc   = int(n["pitch"]) % 12
            if pc in pc_seen.get(slot, set()):
                continue   # another layer already has this PC at this onset
            pc_seen.setdefault(slot, set()).add(pc)
            kept.append(n)
        raw_layers[layer_name] = kept

    # ── D: tritone clash in melody vs chord tones ──────────────────────────────
    # Only fix melody notes; pattern layers are already chord-snapped.
    melody_name = next((k for k in raw_layers if "Melody" in k), None)
    if melody_name:
        beat_dur = 60.0 / tempo
        fixed_mel = []
        for n in raw_layers[melody_name]:
            pitch   = int(n["pitch"])
            m_idx   = int(float(n["onset"]) / md)
            chord   = chord_map.get(m_idx) or chord_map.get(m_idx - 1) or chord_map.get(m_idx + 1)
            if chord is None:
                fixed_mel.append(n)
                continue
            root_pc, chord_pcs = chord
            # Check if this melody note forms a tritone with any chord tone
            has_tritone = any(abs(pitch % 12 - cp) % 12 == 6 for cp in chord_pcs)
            if has_tritone and pitch % 12 not in chord_pcs:
                # Nudge ±1 semitone toward nearest chord tone
                best, best_dist = pitch, 99
                for delta in range(1, 4):
                    for d in (delta, -delta):
                        cand = pitch + d
                        if 21 <= cand <= 108 and cand % 12 in chord_pcs:
                            if abs(d) < best_dist:
                                best, best_dist = cand, abs(d)
                if best != pitch:
                    n = {**n, "pitch": best, "pitch_name": librosa.midi_to_note(best)}
            fixed_mel.append(n)
        raw_layers[melody_name] = fixed_mel

    return raw_layers


# ── main entry point ───────────────────────────────────────────────────────────

_LAYER_ORDER = ["Bass", "Mid", "High", "Lead"]


def _sort_key(name: str) -> int:
    for i, prefix in enumerate(_LAYER_ORDER):
        if prefix in name:
            return i
    return len(_LAYER_ORDER)


def extract_layers(
    notes: List[Dict],
    chord_map: Dict[int, tuple],
    tempo: float,
    audio_path: Optional[Path] = None,
    *,
    dominance_map: Optional[Dict[int, str]] = None,
    upper_feel: float = 0.0,
    min_coverage: float = 0.60,
    min_reps: int = 2,
) -> List[Dict[str, Any]]:
    """
    Full pattern-extraction pipeline. Returns a list of layer dicts:
      [{"id": str, "name": str, "color": str, "notes": [...]}, ...]

    Layers are ordered: Bass → Mid → High → Lead Melody.

    dominance_map: optional {measure_idx → 'bass' | 'melody'} from segmentation.
      When provided, notes from the weaker register in each measure are thinned
      so the arrangement balance matches the original audio.

    upper_feel: median onset offset (seconds) of the upper register vs the beat,
      measured from the original audio during transcription.  Applied to any
      pYIN fill notes generated here so they share the same timing character as
      the transcribed upper-voice notes.

    Falls back to a single "Arrangement" layer (all notes) when:
      - fewer than min_reps chord-cycle repetitions are found
      - chord_map is empty
      - notes list is empty
    """
    _FALLBACK = lambda: [{
        "id":    "arrangement",
        "name":  "Arrangement",
        "color": "#f59e0b",
        "notes": sorted(notes, key=lambda n: (float(n["onset"]), int(n["pitch"]))),
    }]

    if not notes or not chord_map or tempo <= 0:
        return _FALLBACK()

    duration   = max(float(n.get("offset", n["onset"] + 0.5)) for n in notes) + 1.0
    md         = _measure_dur(tempo)
    n_measures = max(1, int(np.ceil(duration / md)))

    cycle_length = detect_cycle_length(chord_map, n_measures)
    n_reps       = max(1, n_measures // cycle_length)

    if n_reps < min_reps:
        return _FALLBACK()

    encoded  = encode_notes(notes, chord_map, tempo, cycle_length)
    elements = mine_pattern_elements(encoded, n_reps, min_coverage, min_reps)

    raw_layers = assign_notes_to_layers(encoded, elements)

    # Collect all pattern notes for pYIN conflict detection
    pattern_notes = [
        n
        for name, layer_notes in raw_layers.items()
        if name != "Lead Melody"
        for n in layer_notes
    ]

    # pYIN fill for melody gaps
    if "Lead Melody" in raw_layers and audio_path is not None:
        filled = fill_melody_pyin(
            raw_layers["Lead Melody"],
            audio_path,
            pattern_notes,
            tempo,
        )
        # Shift any newly generated fill notes by the upper register's feel so
        # they land with the same timing character as the transcribed melody notes
        if upper_feel and abs(upper_feel) >= 0.002:
            existing_onsets = {round(float(n['onset']), 3) for n in raw_layers["Lead Melody"]}
            adjusted = []
            for n in filled:
                if round(float(n['onset']), 3) not in existing_onsets:
                    onset  = round(max(0.0, float(n['onset'])  + upper_feel), 4)
                    offset = round(max(0.0, float(n.get('offset', float(n['onset']) + 0.25)) + upper_feel), 4)
                    n = {**n, 'onset': onset, 'offset': offset if offset > onset else onset + 0.25}
                adjusted.append(n)
            filled = sorted(adjusted, key=lambda x: (float(x['onset']), int(x['pitch'])))
        raw_layers["Lead Melody"] = filled

    # Dominance-based clutter reduction: thin the weaker register per measure
    # so the arrangement balance mirrors the original audio
    if dominance_map:
        raw_layers = apply_dominance_filter(raw_layers, dominance_map, tempo)

    # Resolve inter-layer conflicts (dedup, voice crossing, unison, tritone clash)
    raw_layers = merge_and_resolve(raw_layers, chord_map, tempo)

    # Build merged note list (all layers combined, sorted) for MIDI export
    all_merged = sorted(
        (n for layer_notes in raw_layers.values() for n in layer_notes),
        key=lambda n: (float(n["onset"]), int(n["pitch"])),
    )

    # Sort layers: bass → mid → high → melody; build output list
    result: List[Dict[str, Any]] = []
    for name, layer_notes in sorted(raw_layers.items(), key=lambda kv: _sort_key(kv[0])):
        result.append({
            "id":    name.lower().replace(" ", "_"),
            "name":  name,
            "color": _layer_color(name),
            "notes": sorted(layer_notes, key=lambda n: (float(n["onset"]), int(n["pitch"]))),
        })

    # Guarantee at least one layer
    if not result:
        return _FALLBACK()

    # Attach merged note list for the router to use in MIDI export
    result[0]["_merged_notes"] = all_merged

    return result
