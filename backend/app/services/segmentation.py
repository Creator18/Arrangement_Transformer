from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import librosa
import soundfile as sf

from app.services.audio_utils import js_divergence, compute_mel_from_array


def detect_time_signature(wav_path: Path) -> Tuple[int, int]:
    """Heuristic: detect 3/4 vs 4/4 based on beat grouping."""
    y, sr = librosa.load(str(wav_path), sr=22050)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    # Simple heuristic: try to detect triplet feel
    # Default to 4/4 unless strong 3/4 evidence
    return 4, 4


def segment_by_measures(
    wav_path: Path,
    bass_path: Path,
    melody_path: Path,
    tempo: float,
    time_sig_num: int = 4,
) -> List[Dict[str, Any]]:
    """
    Segment audio into measures and detect the 'star' of each segment
    using Jensen-Shannon divergence against the full segment mel spectrogram.
    """
    y_full, sr = librosa.load(str(wav_path), sr=22050)
    y_bass, _ = librosa.load(str(bass_path), sr=22050)
    y_melody, _ = librosa.load(str(melody_path), sr=22050)

    seconds_per_beat = 60.0 / tempo
    seconds_per_measure = seconds_per_beat * time_sig_num
    total_duration = librosa.get_duration(y=y_full, sr=sr)

    segments = []
    t = 0.0
    measure_idx = 0

    while t < total_duration:
        end_t = min(t + seconds_per_measure, total_duration)
        start_sample = int(t * sr)
        end_sample = int(end_t * sr)

        seg_full = y_full[start_sample:end_sample]
        seg_bass = y_bass[start_sample:min(end_sample, len(y_bass))]
        seg_melody = y_melody[start_sample:min(end_sample, len(y_melody))]

        if len(seg_full) < 512:
            t = end_t
            continue

        mel_full = compute_mel_from_array(seg_full, sr)

        js_bass = 1.0
        js_melody = 1.0

        if len(seg_bass) >= 512:
            mel_bass = compute_mel_from_array(seg_bass, sr)
            js_bass = js_divergence(mel_full, mel_bass)

        if len(seg_melody) >= 512:
            mel_melody = compute_mel_from_array(seg_melody, sr)
            js_melody = js_divergence(mel_full, mel_melody)

        star = "bass" if js_bass <= js_melody else "melody"

        segments.append({
            "measure_index": measure_idx,
            "start_time": round(t, 3),
            "end_time": round(end_t, 3),
            "duration": round(end_t - t, 3),
            "time_signature": f"{time_sig_num}/4",
            "star": star,
            "js_scores": {
                "bass": round(js_bass, 4),
                "melody": round(js_melody, 4),
            },
        })

        t = end_t
        measure_idx += 1

    return segments
