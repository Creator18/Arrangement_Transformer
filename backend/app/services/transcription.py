"""
Per-component transcription.

After separation.py (Demucs) has split the audio into a bass-register and
upper-register stem:

  bass.wav  → pYIN monophonic pitch tracking → bass notes (layer='bass')
  upper.wav → Basic Pitch polyphonic model  → upper notes (no layer label)

Upper notes carry no `layer` field — pattern_extraction.py classifies them
via cycle-based repetition mining instead of the old NMF-component-based
labeling, which is no longer available after the Demucs swap.
"""
from __future__ import annotations

import json
import tempfile
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import List, Dict, Any, Optional
from app.services.nmf_voice_leading import quantize_notes, apply_feel_offset


# ── pYIN bass transcription ───────────────────────────────────────────────────

def _pyin_to_notes(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
    fmin_note: str = 'A0',
    fmax_note: str = 'G3',
    min_duration: float = 0.06,
) -> List[Dict[str, Any]]:
    """
    Convert a pYIN pitch track on a bass-register audio signal into note events.

    Consecutive voiced frames within 2 semitones of each other are merged into
    a single note.  Gaps shorter than 80 ms are bridged.
    """
    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz(fmin_note),
        fmax=librosa.note_to_hz(fmax_note),
        sr=sr,
        hop_length=hop_length,
        frame_length=4096,
        fill_na=None,
    )

    with np.errstate(invalid='ignore'):
        midi_track = np.where(
            voiced_flag & (f0 > 0),
            librosa.hz_to_midi(np.where(f0 > 0, f0, 1.0)),
            np.nan,
        )

    frame_times = librosa.frames_to_time(np.arange(len(midi_track)), sr=sr, hop_length=hop_length)
    frame_dur   = float(frame_times[1] - frame_times[0]) if len(frame_times) > 1 else hop_length / sr

    notes: List[Dict[str, Any]] = []
    in_note       = False
    note_onset    = 0.0
    note_pitches: List[float] = []

    def _flush(onset: float, pitches: List[float], offset: float) -> Optional[Dict]:
        if offset - onset < min_duration or not pitches:
            return None
        pitch = int(round(float(np.median(pitches))))
        pitch = max(21, min(108, pitch))
        return {
            'pitch':      pitch,
            'pitch_name': librosa.midi_to_note(pitch),
            'onset':      round(onset, 4),
            'offset':     round(offset, 4),
            'velocity':   80,
            'layer':      'bass',
        }

    for i, midi in enumerate(midi_track):
        t = float(frame_times[i])
        if not np.isnan(midi):
            if not in_note:
                in_note    = True
                note_onset = t
                note_pitches = [float(midi)]
            else:
                # Bridge small gaps and tolerate ±2 semitone pitch variation
                note_pitches.append(float(midi))
        else:
            if in_note:
                n = _flush(note_onset, note_pitches, t)
                if n:
                    notes.append(n)
                in_note = False
                note_pitches = []

    if in_note and note_pitches:
        n = _flush(note_onset, note_pitches, float(frame_times[-1]))
        if n:
            notes.append(n)

    return notes


# ── Basic Pitch upper transcription ───────────────────────────────────────────
# (NMF-component-based layer labeling was removed when separation switched
# to Demucs.  pattern_extraction.py now classifies unlabeled upper notes via
# cycle mining, which works on register-mixed input without needing a
# precomputed NMF activation matrix.)

def _basic_pitch_notes(wav_path: Path) -> List[Dict[str, Any]]:
    """Run Basic Pitch on a wav file, return raw note list."""
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH

    _, midi_data, _ = predict(
        str(wav_path),
        ICASSP_2022_MODEL_PATH,
        onset_threshold=0.55,
        frame_threshold=0.35,
        minimum_note_length=60,
        minimum_frequency=float(librosa.note_to_hz('C2')),
        maximum_frequency=float(librosa.note_to_hz('C8')),
        melodia_trick=True,
    )

    notes: List[Dict[str, Any]] = []
    if not midi_data.instruments:
        return notes
    for note in midi_data.instruments[0].notes:
        notes.append({
            'pitch':      int(note.pitch),
            'pitch_name': librosa.midi_to_note(int(note.pitch)),
            'onset':      float(note.start),
            'offset':     float(note.end),
            'velocity':   int(note.velocity),
        })
    return notes


# ── post-processing: refine note list ─────────────────────────────────────────

def refine_notes(notes: List[Dict], tempo: float) -> List[Dict]:
    """
    Drop notes shorter than 60 ms, merge same-pitch notes with gap < 50 ms,
    remove exact duplicates within 10 ms.
    """
    if not notes:
        return notes

    notes = [n for n in notes if float(n['offset']) - float(n['onset']) >= 0.06]

    notes.sort(key=lambda n: (n['pitch'], n['onset']))
    merged: List[Dict] = []
    for n in notes:
        if (merged
                and merged[-1]['pitch'] == n['pitch']
                and float(n['onset']) - float(merged[-1]['offset']) < 0.05):
            merged[-1]['offset']   = max(merged[-1]['offset'],   float(n['offset']))
            merged[-1]['velocity'] = max(merged[-1]['velocity'],  int(n.get('velocity', 80)))
        else:
            merged.append(dict(n))
    notes = merged

    notes.sort(key=lambda n: (n['onset'], n['pitch']))
    deduped: List[Dict] = []
    for n in notes:
        if (deduped
                and deduped[-1]['pitch'] == n['pitch']
                and abs(float(n['onset']) - float(deduped[-1]['onset'])) < 0.01):
            continue
        deduped.append(n)

    deduped.sort(key=lambda n: (n['onset'], n['pitch']))
    return deduped


def get_octave_range(notes: List[Dict]) -> Dict[str, Any]:
    if not notes:
        return {'lowest': None, 'highest': None, 'octave_span': 0}
    pitches    = [int(n['pitch']) for n in notes]
    lowest_oct  = min(pitches) // 12 - 1
    highest_oct = max(pitches) // 12 - 1
    return {
        'lowest_midi':    min(pitches),
        'highest_midi':   max(pitches),
        'lowest_note':    librosa.midi_to_note(min(pitches)),
        'highest_note':   librosa.midi_to_note(max(pitches)),
        'lowest_octave':  lowest_oct,
        'highest_octave': highest_oct,
        'octave_span':    highest_oct - lowest_oct + 1,
    }


# ── beat-adaptive timing helpers ──────────────────────────────────────────────

def _compute_feel(notes: List[Dict], beat_times: np.ndarray) -> float:
    """
    Measure how far ahead or behind the beat this set of notes tends to land.

    For each note onset, find the nearest detected beat time and compute the
    offset (onset - beat_time).  The median of all such offsets is the "feel":
      positive → notes tend to land behind the beat
      negative → notes tend to land ahead of the beat

    Offsets beyond 500 ms are discarded (note is closer to the wrong beat).
    Returns 0.0 when there are too few notes or no beat times.
    """
    if not notes or len(beat_times) < 2:
        return 0.0

    offsets = []
    for n in notes:
        onset = float(n['onset'])
        nearest = float(beat_times[int(np.argmin(np.abs(beat_times - onset)))])
        delta = onset - nearest
        if abs(delta) < 0.5:
            offsets.append(delta)

    if not offsets:
        return 0.0
    return float(np.median(offsets))


def _beat_adaptive_quantize(
    notes: List[Dict],
    beat_times: np.ndarray,
    subdivisions: int = 4,
) -> List[Dict]:
    """
    Snap note onsets/offsets to the nearest sub-beat grid position.

    Unlike constant-tempo quantization, this uses the *actual* detected beat
    positions, so it handles tempo variation and rubato correctly.  Between
    consecutive beats the interval is divided into `subdivisions` equal slots
    (subdivisions=4 → 16th-note grid within each beat pair).

    Extrapolates using the last beat interval for notes beyond the final beat.
    Falls back to leaving times unchanged if fewer than 2 beats are detected.
    """
    if len(beat_times) < 2:
        return notes

    beat_times = np.sort(beat_times).astype(float)

    def _snap(t: float) -> float:
        t = float(t)

        # Before first beat — extrapolate using first interval
        if t < beat_times[0]:
            dt   = beat_times[1] - beat_times[0]
            sub  = dt / subdivisions
            diff = t - beat_times[0]
            idx  = round(diff / sub)
            return float(beat_times[0] + idx * sub)

        # After last beat — extrapolate using last interval
        if t >= beat_times[-1]:
            dt   = beat_times[-1] - beat_times[-2]
            sub  = dt / subdivisions
            diff = t - beat_times[-1]
            idx  = round(diff / sub)
            return float(beat_times[-1] + idx * sub)

        # Within the beat grid — find containing interval
        i    = int(np.searchsorted(beat_times, t, side='right')) - 1
        i    = min(i, len(beat_times) - 2)
        dt   = beat_times[i + 1] - beat_times[i]
        sub  = dt / subdivisions
        frac = (t - beat_times[i]) / dt
        idx  = round(frac * subdivisions)
        return float(beat_times[i] + idx * sub)

    result = []
    for n in notes:
        onset_s  = _snap(float(n['onset']))
        offset_s = _snap(float(n.get('offset', float(n['onset']) + 0.25)))

        if offset_s <= onset_s:
            # Extend by one sub-beat using the local interval
            i  = min(int(np.searchsorted(beat_times, onset_s, side='right')) - 1,
                     len(beat_times) - 2)
            dt = (beat_times[i + 1] - beat_times[i]) if i < len(beat_times) - 1 \
                 else (beat_times[-1] - beat_times[-2])
            offset_s = onset_s + dt / subdivisions

        result.append({
            **n,
            'onset':  round(onset_s,  4),
            'offset': round(offset_s, 4),
        })

    result.sort(key=lambda n: (float(n['onset']), int(n['pitch'])))
    return result


# ── main entry point ──────────────────────────────────────────────────────────

def transcribe_components(
    session_dir: Path,
) -> tuple:
    """
    Full per-component transcription pipeline.

    Reads:
      session_dir/components/bass.wav
      session_dir/components/upper.wav
      session_dir/components/H.npy
      session_dir/components.json   (component metadata from separation)

    Returns (notes, timing_feel) where:
      notes        — merged, labeled list sorted by (onset, pitch);
                     each note has a 'layer' field: 'bass' | 'pattern' | 'melody'
      timing_feel  — {'bass': float, 'upper': float} median offset in seconds
                     from the nearest beat for each register (positive = behind beat)
    """
    comp_dir   = session_dir / 'components'
    bass_path  = comp_dir / 'bass.wav'
    upper_path = comp_dir / 'upper.wav'
    meta_path  = session_dir / 'components.json'

    if not bass_path.exists() or not upper_path.exists():
        raise FileNotFoundError('Component audio files missing. Run /separate first.')

    with open(meta_path) as f:
        meta = json.load(f)

    tempo      = float(meta.get('tempo_bpm', 120.0))
    hop_length = int(meta.get('hop_length', 512))
    sr_comp    = int(meta.get('sr', 22050))

    # Actual beat positions saved during separation for beat-adaptive quantization
    raw_beat_frames = meta.get('beat_frames', [])
    beat_times = (
        librosa.frames_to_time(np.array(raw_beat_frames), sr=sr_comp, hop_length=hop_length)
        if raw_beat_frames else np.array([])
    )

    # ── bass: pYIN ────────────────────────────────────────────────────────────
    y_bass, sr_bass = librosa.load(str(bass_path), sr=sr_comp, mono=True)
    bass_notes      = _pyin_to_notes(y_bass, sr_bass, hop_length=hop_length)

    # ── upper: Basic Pitch ────────────────────────────────────────────────────
    # No more NMF-component-based labeling — Demucs separation doesn't produce
    # an H matrix.  Upper notes carry no `layer` field; pattern_extraction.py
    # falls back to its cycle-mining classifier on unlabeled notes, which is
    # exactly what we want for register-mixed input.
    upper_notes = _basic_pitch_notes(upper_path)

    # ── measure per-layer feel before any quantization ────────────────────────
    # Raw pYIN and Basic Pitch onsets reveal how far ahead/behind the beat each
    # register tends to land.  This is applied to generated notes later so that
    # walking bass and pYIN fill inherit the same timing character.
    bass_feel  = _compute_feel(bass_notes,  beat_times)
    upper_feel = _compute_feel(upper_notes, beat_times)

    # ── merge + deduplicate ───────────────────────────────────────────────────
    all_notes = sorted(
        bass_notes + upper_notes,
        key=lambda n: (float(n['onset']), int(n['pitch'])),
    )
    deduped: List[Dict] = []
    for n in all_notes:
        if any(
            abs(float(n['onset']) - float(x['onset'])) < 0.03
            and abs(int(n['pitch']) - int(x['pitch'])) <= 2
            for x in deduped[-40:]
        ):
            continue
        deduped.append(n)

    refined = refine_notes(deduped, tempo)

    # ── beat-adaptive quantization ────────────────────────────────────────────
    # Use actual beat frame positions (not a computed constant-tempo lattice) so
    # that tempo variation and rubato in the performance are handled correctly.
    if len(beat_times) >= 2:
        quantized = _beat_adaptive_quantize(refined, beat_times, subdivisions=4)
    else:
        quantized = quantize_notes(refined, tempo, subdivisions=16)

    timing_feel = {
        'bass':  round(bass_feel,  4),
        'upper': round(upper_feel, 4),
    }

    return quantized, timing_feel
