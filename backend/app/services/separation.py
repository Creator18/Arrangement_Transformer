from pathlib import Path
from typing import Dict
import numpy as np
import librosa
import soundfile as sf


def run_spleeter(wav_path: Path, session_dir: Path, stems: int = 2) -> Dict[str, Path]:
    """
    Frequency-band stem separation via librosa (replaces Spleeter).
    Bass stem: low-pass below 300 Hz.
    Melody stem: high-pass above 300 Hz.
    """
    stems_dir = session_dir / "stems" / wav_path.stem
    stems_dir.mkdir(parents=True, exist_ok=True)

    y, sr = librosa.load(str(wav_path), sr=44100, mono=True)

    # Build frequency-domain filter cutoff
    cutoff_hz = 300.0
    n_fft = 2048
    hop = 512

    D = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # Soft mask: sigmoid around cutoff
    cutoff_bin = np.searchsorted(freqs, cutoff_hz)
    mask_bass = np.zeros(len(freqs))
    mask_bass[:cutoff_bin] = 1.0
    # Smooth the transition over ±3 bins
    for b in range(max(0, cutoff_bin - 3), min(len(freqs), cutoff_bin + 3)):
        dist = (b - cutoff_bin) / 3.0
        mask_bass[b] = 1.0 / (1.0 + np.exp(5 * dist))

    mask_melody = 1.0 - mask_bass

    bass_D = D * mask_bass[:, np.newaxis]
    melody_D = D * mask_melody[:, np.newaxis]

    y_bass = librosa.istft(bass_D, hop_length=hop)
    y_melody = librosa.istft(melody_D, hop_length=hop)

    bass_path = stems_dir / "accompaniment.wav"
    melody_path = stems_dir / "vocals.wav"

    sf.write(str(bass_path), y_bass, sr, subtype="PCM_16")
    sf.write(str(melody_path), y_melody, sr, subtype="PCM_16")

    return {
        "bass": bass_path,
        "melody": melody_path,
    }
