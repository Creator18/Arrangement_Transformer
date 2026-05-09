from typing import List, Dict, Any, Optional
import copy
import librosa
import pretty_midi


# MIDI note number boundaries
A0_MIDI = 21   # lowest on 88-key
C8_MIDI = 108  # highest on 88-key
# 5-octave keyboard: C2-C7 (MIDI 36-96)
FIVE_OCT_LOW = 36
FIVE_OCT_HIGH = 96
# Register zones for 5-octave layout
BASS_ZONE_HIGH = 55      # G3
MELODY_ZONE_LOW = 48     # C3
MELODY_ZONE_HIGH = 84    # C6
VOICE_ZONE_LOW = 60      # C4


def classify_note_part(pitch: int) -> str:
    if pitch < MELODY_ZONE_LOW:
        return "bass"
    elif pitch <= VOICE_ZONE_LOW:
        return "melody"
    else:
        return "voice"


def _octave_shift(pitch: int, semitones: int) -> int:
    return max(FIVE_OCT_LOW, min(FIVE_OCT_HIGH, pitch + semitones))


def _compress_to_range(notes: List[Dict], low: int, high: int, shift_semitones: int) -> List[Dict]:
    out = []
    for n in notes:
        new_pitch = _octave_shift(n["pitch"], shift_semitones)
        new_pitch = max(low, min(high, new_pitch))
        out.append({**n, "pitch": new_pitch, "pitch_name": librosa.midi_to_note(new_pitch)})
    return out


def _resolve_collisions(
    bass_notes: List[Dict],
    melody_notes: List[Dict],
    voice_notes: List[Dict],
    tolerance: float = 0.05,
) -> tuple:
    """Bass root wins on timing collision. Melody is second. Voice most compressible."""
    bass_onsets = {round(n["onset"], 2): n for n in bass_notes}

    resolved_melody = []
    for n in melody_notes:
        onset_key = round(n["onset"], 2)
        collision = any(
            abs(onset_key - b_onset) < tolerance for b_onset in bass_onsets
        )
        if collision:
            resolved_melody.append({
                **n,
                "velocity": max(20, n["velocity"] - 30),
                "offset": n["offset"] + 0.05,
            })
        else:
            resolved_melody.append(n)

    resolved_voice = []
    melody_onsets = {round(n["onset"], 2) for n in resolved_melody}
    for n in voice_notes:
        onset_key = round(n["onset"], 2)
        bass_collision = any(abs(onset_key - b) < tolerance for b in bass_onsets)
        mel_collision = any(abs(onset_key - m) < tolerance for m in melody_onsets)
        if bass_collision or mel_collision:
            resolved_voice.append({
                **n,
                "velocity": max(15, n["velocity"] - 40),
                "offset": n["offset"] + 0.08,
            })
        else:
            resolved_voice.append(n)

    return bass_notes, resolved_melody, resolved_voice


def _apply_ghost_note_fill(
    bass_notes: List[Dict],
    tempo: float,
    segment_start: float,
    segment_end: float,
) -> List[Dict]:
    """
    If tempo breathing room allows, double hit frequency with ghost notes.
    Ghost = even hits at low velocity to re-trigger sustain.
    """
    seconds_per_beat = 60.0 / tempo
    gap_threshold = seconds_per_beat * 0.5   # half a beat gap needed

    result = []
    for i, note in enumerate(bass_notes):
        result.append(note)
        next_onset = bass_notes[i + 1]["onset"] if i + 1 < len(bass_notes) else segment_end
        gap = next_onset - note["onset"]
        if gap > gap_threshold:
            ghost_onset = note["onset"] + gap / 2
            ghost = {
                **note,
                "onset": ghost_onset,
                "offset": ghost_onset + (note["offset"] - note["onset"]) * 0.8,
                "velocity": max(20, note["velocity"] // 3),
                "_ghost": True,
            }
            result.append(ghost)
    return result


def transform_notes(
    notes: List[Dict],
    segments: List[Dict],
    source_octave_span: int,
    target_octave_span: int,
    tempo: float,
) -> List[Dict]:
    """
    Core transformation: compress note range from source to target octave span
    while respecting priority hierarchy and star-of-segment protection.
    """
    semitone_shift = (source_octave_span - target_octave_span) * 6  # approximate centering shift

    transformed = []

    for seg in segments:
        start = seg["start_time"]
        end = seg["end_time"]
        star = seg["star"]

        seg_notes = [n for n in notes if start <= n["onset"] < end]

        bass_notes = [n for n in seg_notes if n["pitch"] < MELODY_ZONE_LOW]
        melody_notes = [n for n in seg_notes if MELODY_ZONE_LOW <= n["pitch"] <= VOICE_ZONE_LOW]
        voice_notes = [n for n in seg_notes if n["pitch"] > VOICE_ZONE_LOW]

        # Compress non-star parts first
        if star != "bass":
            bass_notes = _compress_to_range(bass_notes, FIVE_OCT_LOW, BASS_ZONE_HIGH, 0)
        else:
            bass_notes = _compress_to_range(bass_notes, FIVE_OCT_LOW, BASS_ZONE_HIGH, 0)

        if star != "melody":
            melody_notes = _compress_to_range(
                melody_notes, MELODY_ZONE_LOW, MELODY_ZONE_HIGH, -semitone_shift // 2
            )
        else:
            melody_notes = _compress_to_range(melody_notes, MELODY_ZONE_LOW, MELODY_ZONE_HIGH, 0)

        # Voice is always most compressible
        voice_notes = _compress_to_range(voice_notes, VOICE_ZONE_LOW, FIVE_OCT_HIGH, -semitone_shift)

        bass_notes, melody_notes, voice_notes = _resolve_collisions(
            bass_notes, melody_notes, voice_notes
        )

        # Ghost note fill for bass when tempo allows
        if bass_notes:
            bass_notes = _apply_ghost_note_fill(bass_notes, tempo, start, end)

        transformed.extend(bass_notes)
        transformed.extend(melody_notes)
        transformed.extend(voice_notes)

    transformed.sort(key=lambda n: n["onset"])
    return transformed


def notes_to_pretty_midi(notes: List[Dict], tempo: float = 120.0) -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    piano = pretty_midi.Instrument(program=0)
    for n in notes:
        note = pretty_midi.Note(
            velocity=int(n.get("velocity", 80)),
            pitch=int(n["pitch"]),
            start=float(n["onset"]),
            end=float(n["offset"]),
        )
        piano.notes.append(note)
    pm.instruments.append(piano)
    return pm
