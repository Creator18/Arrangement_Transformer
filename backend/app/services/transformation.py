from typing import List, Dict, Any, Optional
import copy
import numpy as np
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


def notes_to_pretty_midi(
    notes: List[Dict],
    tempo: float = 120.0,
    beat_times: Optional[np.ndarray] = None,
    resolution: int = 480,
) -> pretty_midi.PrettyMIDI:
    """
    Convert a note list to a PrettyMIDI object at a strict, uniform tempo.

    When beat_times (actual beat positions from the original audio, in seconds)
    is provided, each note's floating-point onset is mapped to its fractional
    beat position in the original performance's variable-tempo grid, then
    snapped to the nearest 16th-note tick in the uniform-tempo MIDI grid.

    This separates two concerns:
      - Rhythmic intent  (which beat / subdivision the note targets) — preserved
      - Absolute spacing (the original's rubato / tempo drift)        — discarded

    Result: the MIDI always plays back at a perfectly consistent tempo that
    matches the overall BPM of the source track, with no jumped sections.
    """
    # 12 subdivisions per beat = LCM(4, 3) — captures duplet 16ths, triplet 8ths,
    # and dotted-beat compound meters (6/8, 12/8, 9/8) without needing to know
    # the time signature.  Resolution is set to a multiple of 12 so that every
    # subdivision maps to an exact integer tick count.
    SUBS = 12                           # subdivisions per beat
    resolution = SUBS * 40             # 480 ticks/beat, each sub = 40 ticks

    pm  = pretty_midi.PrettyMIDI(initial_tempo=tempo, resolution=resolution)
    spb = 60.0 / tempo                 # seconds per beat at uniform tempo
    tpb = float(resolution)            # ticks per beat
    tps = tpb / SUBS                   # ticks per subdivision (exactly 40)

    bt: Optional[np.ndarray] = (
        np.asarray(beat_times, dtype=float) if beat_times is not None and len(beat_times) >= 2
        else None
    )

    def _time_to_tick(t: float) -> int:
        t = float(t)

        if bt is None:
            # No beat grid — direct uniform conversion
            return max(0, round(t / spb * tpb))

        N = len(bt)

        if t <= bt[0]:
            dt   = bt[1] - bt[0]
            frac = (t - bt[0]) / dt
            i    = 0
        elif t >= bt[-1]:
            dt   = bt[-1] - bt[-2]
            frac = (t - bt[-1]) / dt
            i    = N - 1
        else:
            i    = int(np.searchsorted(bt, t, side='right')) - 1
            i    = min(i, N - 2)
            dt   = bt[i + 1] - bt[i]
            frac = (t - bt[i]) / dt

        # Fractional beat position in the original performance
        beat_pos = i + frac

        # Snap to nearest subdivision (works for duplet, triplet, compound meters)
        sub_idx = round(beat_pos * SUBS)
        return max(0, round(sub_idx * tps))

    def _tick_to_s(tick: int) -> float:
        return tick / tpb * spb

    min_ticks = max(1, round(tps))     # minimum note length = one subdivision

    piano = pretty_midi.Instrument(program=0)
    for n in notes:
        start_tick = _time_to_tick(float(n["onset"]))
        end_tick   = _time_to_tick(float(n.get("offset", float(n["onset"]) + spb / 4)))
        if end_tick <= start_tick:
            end_tick = start_tick + min_ticks

        piano.notes.append(pretty_midi.Note(
            velocity=max(1, min(127, int(n.get("velocity", 80)))),
            pitch=max(0, min(127, int(n["pitch"]))),
            start=_tick_to_s(start_tick),
            end=_tick_to_s(end_tick),
        ))

    pm.instruments.append(piano)
    return pm
