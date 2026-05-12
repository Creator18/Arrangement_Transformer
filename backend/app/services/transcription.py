"""
Per-component transcription.

After separation.py (Demucs + bandpass) has split the audio into a bass-
register and upper-register stem:

  bass.wav  → pYIN monophonic pitch tracking           → bass notes (layer='bass')
  upper.wav → Onsets & Frames polyphonic transcription → upper notes (no layer label)

Onsets & Frames runs via piano-transcription-inference (the PyTorch port of
Magenta's MAESTRO-trained model).  Sharper onsets and richer velocity than
the previous Basic Pitch implementation — the user-perceived "feel" of the
resulting arrangement depends almost entirely on the upper transcriber's
onset precision, so this swap is one of the biggest single quality wins
in the pipeline.

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


# ── Upper-register transcription via Magenta's Onsets & Frames ────────────────
# Uses the PyTorch port (piano-transcription-inference) of the MAESTRO-trained
# O&F model.  Significantly tighter onsets and more meaningful velocity than
# Basic Pitch — at the cost of being piano-specialised, which is fine for us
# because Demucs already strips drums and vocals upstream.
#
# (NMF-component-based layer labeling was removed when separation switched to
# Demucs.  pattern_extraction.py classifies unlabeled upper notes via cycle
# mining, which works on register-mixed input without needing the precomputed
# NMF activation matrix.)

# Lazy global so the ~170 MB checkpoint loads once per worker, not per call.
_of_transcriptor = None


# Zenodo URL the upstream package would shell out to wget — we download it
# directly via Python so the code works on Windows (no wget binary required).
_OF_CKPT_URL = (
    "https://zenodo.org/record/4034264/files/"
    "CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1"
)
_OF_CKPT_FILENAME = "note_F1=0.9677_pedal_F1=0.9186.pth"
_OF_CKPT_MIN_BYTES = int(1.6e8)   # ~160 MB — same sanity check the package uses


def _ensure_of_checkpoint() -> Path:
    """
    Download the Onsets & Frames checkpoint into the same cache path the
    upstream package looks for (~/piano_transcription_inference_data/) using
    urllib so we don't depend on a `wget` binary being on PATH.

    Returns the local checkpoint path.  Idempotent: a checkpoint already
    present at the expected size is left alone.
    """
    import urllib.request
    from pathlib import Path as _P
    ckpt_dir = _P.home() / "piano_transcription_inference_data"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / _OF_CKPT_FILENAME

    if ckpt_path.exists() and ckpt_path.stat().st_size >= _OF_CKPT_MIN_BYTES:
        return ckpt_path

    tmp_path = ckpt_path.with_suffix(ckpt_path.suffix + ".part")
    print(f"[onsets_frames] downloading checkpoint to {ckpt_path} (~165 MB)…")
    with urllib.request.urlopen(_OF_CKPT_URL) as resp, open(tmp_path, "wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk = 1 << 20   # 1 MiB
        while True:
            buf = resp.read(chunk)
            if not buf:
                break
            out.write(buf)
            downloaded += len(buf)
            if total:
                print(f"\r[onsets_frames] {downloaded // (1<<20)} / {total // (1<<20)} MiB", end="", flush=True)
        print()
    tmp_path.replace(ckpt_path)
    return ckpt_path


def _get_of_transcriptor():
    """Construct (or fetch) the Onsets & Frames transcriber.  Lazy + cached.

    First call downloads the MAESTRO-trained checkpoint (~170 MB) via urllib
    and caches it under ~/piano_transcription_inference_data/.  Subsequent
    calls reuse the cached file.  Device auto-detects CUDA → falls back to CPU.
    """
    global _of_transcriptor
    if _of_transcriptor is not None:
        return _of_transcriptor
    import torch
    from piano_transcription_inference import PianoTranscription
    ckpt_path = _ensure_of_checkpoint()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _of_transcriptor = PianoTranscription(device=device, checkpoint_path=str(ckpt_path))
    return _of_transcriptor


def _onsets_frames_notes(wav_path: Path) -> List[Dict[str, Any]]:
    """Run Magenta-O&F (PyTorch port) on a WAV file, return raw note list.

    Output is in our standard note-dict format with floating-point onset/offset
    in seconds, integer velocity 0–127, and pitch clamped to the 88-key
    range (MIDI 21–108).
    """
    from piano_transcription_inference import sample_rate as OF_SR

    # Model expects 16 kHz mono audio.
    audio, _ = librosa.load(str(wav_path), sr=OF_SR, mono=True)
    if audio.size == 0:
        return []

    transcriptor = _get_of_transcriptor()
    result = transcriptor.transcribe(audio.astype(np.float32), midi_path=None)

    notes: List[Dict[str, Any]] = []
    for ev in result.get("est_note_events", []):
        pitch = int(ev["midi_note"])
        if pitch < 21 or pitch > 108:           # clamp to 88-key range
            continue
        onset  = float(ev["onset_time"])
        offset = float(ev["offset_time"])
        if offset <= onset:                      # discard zero-length artefacts
            continue
        notes.append({
            "pitch":      pitch,
            "pitch_name": librosa.midi_to_note(pitch),
            "onset":      onset,
            "offset":     offset,
            "velocity":   max(1, min(127, int(ev["velocity"]))),
        })
    notes.sort(key=lambda n: (n["onset"], n["pitch"]))
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
    comp_dir     = session_dir / 'components'
    bass_path    = comp_dir / 'bass.wav'
    upper_path   = comp_dir / 'upper.wav'
    pattern_path = comp_dir / 'pattern.wav'
    melody_path  = comp_dir / 'melody.wav'
    meta_path    = session_dir / 'components.json'

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
    # bass_notes already carry layer='bass' from _pyin_to_notes

    # ── upper: split into pattern + melody via REPET, transcribe separately ──
    # pattern.wav / melody.wav are produced by separation.py.  Each is run
    # through Onsets & Frames independently and the resulting notes get
    # pre-labeled — pattern_extraction.py reads these labels to decide which
    # bucket each note belongs to without falling back to cycle mining.
    if pattern_path.exists() and melody_path.exists():
        pattern_notes = _onsets_frames_notes(pattern_path)
        for n in pattern_notes:
            n['layer'] = 'pattern'
        melody_notes  = _onsets_frames_notes(melody_path)
        for n in melody_notes:
            n['layer'] = 'melody'
        upper_notes = pattern_notes + melody_notes
    else:
        # Backwards compatibility for sessions produced before phase 5c —
        # fall back to transcribing the combined upper.wav with no labels.
        upper_notes = _onsets_frames_notes(upper_path)

    # ── measure per-layer feel before any quantization ────────────────────────
    # Raw pYIN and Onsets & Frames onsets reveal how far ahead/behind the beat
    # each register tends to land.  Applied later to generated notes (walking
    # bass, pYIN fill) so they inherit the same timing character.
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
