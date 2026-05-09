from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import librosa


def run_basic_pitch(wav_path: Path) -> List[Dict[str, Any]]:
    """Run Basic Pitch with tighter thresholds to reduce false positives."""
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH

    model_output, midi_data, note_events = predict(
        str(wav_path),
        ICASSP_2022_MODEL_PATH,
        onset_threshold=0.6,
        frame_threshold=0.4,
        minimum_note_length=80,
        minimum_frequency=27.5,
        maximum_frequency=4200.0,
        melodia_trick=True,
    )

    notes = []
    for note in midi_data.instruments[0].notes:
        notes.append({
            "pitch":      int(note.pitch),
            "pitch_name": librosa.midi_to_note(int(note.pitch)),
            "onset":      float(note.start),
            "offset":     float(note.end),
            "velocity":   int(note.velocity),
        })
    return notes


def refine_notes(notes: List[Dict], tempo: float) -> List[Dict]:
    """
    Post-process transcription:
    1. Drop notes shorter than 60 ms
    2. Merge same-pitch notes with gap < 50 ms
    3. Remove exact duplicates within 10 ms
    """
    if not notes:
        return notes

    MIN_DURATION = 0.06
    MERGE_GAP    = 0.05
    DEDUP_TOL    = 0.01

    notes = [n for n in notes if (n["offset"] - n["onset"]) >= MIN_DURATION]

    notes.sort(key=lambda n: (n["pitch"], n["onset"]))
    merged = []
    for n in notes:
        if (merged
                and merged[-1]["pitch"] == n["pitch"]
                and n["onset"] - merged[-1]["offset"] < MERGE_GAP):
            merged[-1]["offset"]   = max(merged[-1]["offset"], n["offset"])
            merged[-1]["velocity"] = max(merged[-1]["velocity"], n["velocity"])
        else:
            merged.append(dict(n))
    notes = merged

    notes.sort(key=lambda n: (n["onset"], n["pitch"]))
    deduped = []
    for n in notes:
        if (deduped
                and deduped[-1]["pitch"] == n["pitch"]
                and abs(n["onset"] - deduped[-1]["onset"]) < DEDUP_TOL):
            continue
        deduped.append(n)

    deduped.sort(key=lambda n: (n["onset"], n["pitch"]))
    return deduped


# ── pitch correction ──────────────────────────────────────────────────────────

def _chroma_from_audio(y: np.ndarray, sr: int) -> np.ndarray:
    """Return mean chroma vector (12,) from audio array."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, bins_per_octave=36)
    return chroma.mean(axis=1)


def _chroma_from_notes(notes: List[Dict], sr: int = 22050) -> np.ndarray:
    """Synthesise notes to audio with simple sine waves and get mean chroma."""
    if not notes:
        return np.zeros(12)
    duration = max(float(n.get("offset", n["onset"] + 0.5)) for n in notes) + 1.0
    total = int(duration * sr)
    audio = np.zeros(total, dtype=np.float32)
    for n in notes:
        freq  = 440.0 * (2.0 ** ((int(n["pitch"]) - 69) / 12.0))
        onset = int(float(n["onset"]) * sr)
        end   = min(int(float(n.get("offset", n["onset"] + 0.25)) * sr), total)
        length = end - onset
        if length <= 0:
            continue
        t = np.arange(length) / sr
        audio[onset:end] += (np.sin(2 * np.pi * freq * t) * 0.3).astype(np.float32)
    peak = np.abs(audio).max()
    if peak > 0:
        audio /= peak
    return _chroma_from_audio(audio, sr)


def _best_semitone_shift(chroma_orig: np.ndarray, chroma_trans: np.ndarray) -> int:
    """
    Try all 12 cyclic shifts of chroma_trans and return the one with the
    highest dot-product similarity to chroma_orig.
    Prefers smaller shifts: ties broken by |shift|.
    """
    best_shift = 0
    best_score = -np.inf
    for shift in range(12):
        score = float(np.dot(chroma_orig, np.roll(chroma_trans, shift)))
        # Prefer smaller absolute shift (wrap > 6 to negative)
        eff = shift if shift <= 6 else shift - 12
        if score > best_score or (score == best_score and abs(eff) < abs(best_shift)):
            best_score = score
            best_shift = eff
    return best_shift


def _apply_semitone_shift(notes: List[Dict], shift: int) -> List[Dict]:
    if shift == 0:
        return notes
    corrected = []
    for n in notes:
        new_pitch = int(n["pitch"]) + shift
        if 21 <= new_pitch <= 108:
            corrected.append({
                **n,
                "pitch":      new_pitch,
                "pitch_name": librosa.midi_to_note(new_pitch),
            })
        else:
            corrected.append(n)   # keep original if out of piano range
    return corrected


def _per_note_octave_snap(
    notes: List[Dict], y_orig: np.ndarray, sr: int
) -> List[Dict]:
    """
    For each note, test ±1 octave shift and pick the version whose
    frequency region has stronger energy in the original audio.
    Uses a simple constant-Q energy check in the note's time window.
    """
    cqt = np.abs(librosa.cqt(y_orig, sr=sr, bins_per_octave=12, n_bins=88))
    # cqt rows correspond to MIDI notes starting at C1 (MIDI 24) approximately
    CQT_BASE = 24

    corrected = []
    for n in notes:
        pitch = int(n["pitch"])
        onset_frame = int(float(n["onset"]) * sr / 512)
        offset_frame = max(onset_frame + 1,
                           int(float(n.get("offset", n["onset"] + 0.25)) * sr / 512))
        offset_frame = min(offset_frame, cqt.shape[1])

        def energy_at(p: int) -> float:
            idx = p - CQT_BASE
            if not (0 <= idx < cqt.shape[0]):
                return 0.0
            return float(cqt[idx, onset_frame:offset_frame].mean())

        base_energy  = energy_at(pitch)
        up_energy    = energy_at(pitch + 12)
        down_energy  = energy_at(pitch - 12)

        best_pitch = pitch
        best_energy = base_energy
        for candidate, e in [(pitch + 12, up_energy), (pitch - 12, down_energy)]:
            if e > best_energy * 1.4 and 21 <= candidate <= 108:
                best_energy = e
                best_pitch = candidate

        corrected.append({
            **n,
            "pitch":      best_pitch,
            "pitch_name": librosa.midi_to_note(best_pitch),
        })
    return corrected


def pitch_correct(notes: List[Dict], wav_path: Path) -> Tuple[List[Dict], int]:
    """
    Compare the chroma fingerprint of the original audio against the
    transcription and correct any systematic semitone offset.
    Then snap individual notes to their best octave based on CQT energy.

    Returns (corrected_notes, semitone_shift_applied).
    """
    sr = 22050
    y_orig, _ = librosa.load(str(wav_path), sr=sr)

    chroma_orig  = _chroma_from_audio(y_orig, sr)
    chroma_trans = _chroma_from_notes(notes, sr)

    shift = _best_semitone_shift(chroma_orig, chroma_trans)
    notes = _apply_semitone_shift(notes, shift)
    notes = _per_note_octave_snap(notes, y_orig, sr)

    return notes, shift


def get_octave_range(notes: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not notes:
        return {"lowest": None, "highest": None, "octave_span": 0}
    pitches     = [n["pitch"] for n in notes]
    lowest      = min(pitches)
    highest     = max(pitches)
    lowest_oct  = lowest  // 12 - 1
    highest_oct = highest // 12 - 1
    return {
        "lowest_midi":    lowest,
        "highest_midi":   highest,
        "lowest_note":    librosa.midi_to_note(lowest),
        "highest_note":   librosa.midi_to_note(highest),
        "lowest_octave":  lowest_oct,
        "highest_octave": highest_oct,
        "octave_span":    highest_oct - lowest_oct + 1,
    }
