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


def enforce_uniform_tempo(
    notes: List[Dict],
    beat_times: Optional[np.ndarray],
    tempo: float,
    subdivisions: int = 12,
) -> List[Dict]:
    """
    Re-time merged notes onto a strict uniform-tempo grid while preserving
    each note's rhythmic position relative to the original audio's beat grid.

    For each note (assumed to be on the original-audio timeline in seconds):
      1. Find its fractional beat position in the supplied beat_times grid
         (linear interpolation between beats; linear extrapolation outside).
      2. Snap that fractional beat position to the nearest 1/subdivisions.
         12 subdivisions per beat (LCM of 4 and 3) accommodates duplet 16ths,
         triplet 8ths, and compound meters without time-signature detection.
      3. Convert the snapped beat position to seconds at the uniform tempo.

    The pass is deterministic and idempotent — running it again on the result
    produces the same output, so a single invocation is enough to smooth the
    entire piece.  Notes are returned sorted by (onset, pitch) and any
    zero/negative durations are bumped to one subdivision.
    """
    if not notes:
        return notes

    spb   = 60.0 / max(tempo, 1.0)            # uniform seconds per beat
    sub_s = spb / subdivisions                # seconds per subdivision
    bt = (np.asarray(beat_times, dtype=float)
          if beat_times is not None and len(beat_times) >= 2 else None)

    def _to_uniform(t: float) -> float:
        t = float(t)
        if bt is None:
            sub_idx = round(t / sub_s)
            return max(0.0, sub_idx * sub_s)
        N = len(bt)
        if t <= bt[0]:
            dt   = bt[1] - bt[0]
            frac = (t - bt[0]) / dt if dt > 0 else 0.0
            i    = 0
        elif t >= bt[-1]:
            dt   = bt[-1] - bt[-2]
            frac = (t - bt[-1]) / dt if dt > 0 else 0.0
            i    = N - 1
        else:
            i    = int(np.searchsorted(bt, t, side='right')) - 1
            i    = min(i, N - 2)
            dt   = bt[i + 1] - bt[i]
            frac = (t - bt[i]) / dt if dt > 0 else 0.0
        beat_pos = i + frac
        sub_idx  = round(beat_pos * subdivisions)
        return max(0.0, sub_idx * sub_s)

    out: List[Dict] = []
    for n in notes:
        onset_u  = _to_uniform(float(n['onset']))
        offset_u = _to_uniform(float(n.get('offset', float(n['onset']) + spb / 4.0)))
        if offset_u <= onset_u:
            offset_u = onset_u + sub_s
        out.append({**n, 'onset': round(onset_u, 4), 'offset': round(offset_u, 4)})

    out.sort(key=lambda x: (float(x['onset']), int(x['pitch'])))
    return out


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


# ── song-arrangement reduction ────────────────────────────────────────────────
#
# These two run on the PROJECTION path only (songs), never on a piano
# transcription.  The distinction matters: Onsets & Frames onsets are worth
# preserving to the millisecond (note F1 0.9677), so rule R2 protects them.
# Projection onsets come from peak-picking at precision 0.31-0.47, so
# preserving them exactly preserves noise.
#
# Compared against a human Synthesia arrangement of the same track, ours
# differed in three ways: no repeating figure, onsets scattered off the grid,
# and roughly ten times the note count.  These address the second and third.
# The first needs pattern reconstruction and is a bigger job.

SNAP_SUBDIVISIONS = 4      # sixteenths at 4/4
MAX_SOUNDING = 4           # the human arrangement never exceeded about this


def snap_to_beat_grid(notes: List[Dict[str, Any]],
                      beat_times: Optional[np.ndarray],
                      subdivisions: int = SNAP_SUBDIVISIONS,
                      min_dur: float = 0.05) -> List[Dict[str, Any]]:
    """Snap onsets to subdivisions of the MEASURED beat grid.

    Deliberately not `quantize_notes` or `enforce_uniform_tempo`: both impose a
    metronomic grid derived from a single tempo, which flattens real tempo
    drift and was the direct cause of the "rushing" complaint.  madmom's beat
    times follow the performance, so snapping between consecutive real beats
    keeps the piece's own timing while removing the sub-beat scatter that makes
    a chord read as a staircase.
    """
    if beat_times is None or len(beat_times) < 2 or not notes:
        return notes

    grid = []
    bt = np.asarray(beat_times, dtype=float)
    for a, b in zip(bt[:-1], bt[1:]):
        for k in range(subdivisions):
            grid.append(a + (b - a) * k / subdivisions)
    grid.append(float(bt[-1]))
    grid = np.asarray(grid)

    out = []
    for n in notes:
        m = dict(n)
        on = float(m["onset"])
        off = float(m.get("offset", on + 0.25))
        on_s = float(grid[int(np.argmin(np.abs(grid - on)))])
        off_s = float(grid[int(np.argmin(np.abs(grid - off)))])
        if off_s - on_s < min_dur:
            off_s = on_s + max(min_dur, off - on)
        m["onset"], m["offset"] = on_s, off_s
        out.append(m)
    out.sort(key=lambda x: (float(x["onset"]), int(x["pitch"])))
    return out


def cap_polyphony(notes: List[Dict[str, Any]],
                  max_sounding: int = MAX_SOUNDING) -> List[Dict[str, Any]]:
    """Keep at most `max_sounding` notes ringing at once, strongest first.

    enforce_hand_constraints already caps at five per hand and ten total, which
    is what a pianist can physically reach.  An arranger writes far less than
    they can reach: in the reference arrangement, two to four notes sound at
    any instant.  Velocity is the ranking, because on this path it comes from
    activation strength, so the weakest notes are also the least likely to be
    real.
    """
    if not notes:
        return notes
    ordered = sorted(notes, key=lambda n: (float(n["onset"]), -int(n.get("velocity", 80))))
    kept: List[Dict[str, Any]] = []
    for n in ordered:
        on = float(n["onset"])
        active = [k for k in kept if float(k["offset"]) > on + 1e-6]
        if len(active) < max_sounding:
            kept.append(n)
            continue
        weakest = min(active, key=lambda k: int(k.get("velocity", 80)))
        if int(n.get("velocity", 80)) > int(weakest.get("velocity", 80)):
            # Release the weaker note here rather than deleting it outright —
            # it has already been sounding, and truncating preserves the
            # attack the listener has heard.
            weakest["offset"] = on
            if float(weakest["offset"]) - float(weakest["onset"]) < 0.03:
                kept.remove(weakest)
            kept.append(n)
    kept.sort(key=lambda x: (float(x["onset"]), int(x["pitch"])))
    return kept


# ── scale enforcement ─────────────────────────────────────────────────────────

DROP_NON_SCALE = True     # remove every note outside the detected scale


def drop_non_scale(notes: List[Dict[str, Any]], scale_pcs) -> List[Dict[str, Any]]:
    """Remove every note whose pitch class is outside the detected scale.

    A blunt instrument on purpose.  The scale cleanup at /segment runs on the
    TRANSCRIPTION, so anything created later — the whole arpeggiated
    accompaniment — has never been checked against the key: measured, 20.3% of
    generated accompaniment notes sat outside it, and only 47% of bars were
    given a fully diatonic chord.  This closes that hole wherever notes come
    from.

    It is applied last, after generation and after the hand constraints, so
    nothing downstream can reintroduce an out-of-key note.
    """
    if not DROP_NON_SCALE or not notes or not scale_pcs:
        return notes
    pcs = set(int(p) % 12 for p in scale_pcs)
    return [n for n in notes if (int(n["pitch"]) % 12) in pcs]
