import os
import subprocess
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path

# Prefer the bundled FFmpeg; fall back to system PATH
_REPO_ROOT = Path(__file__).parent.parent.parent.parent  # Arrangement_Transformer/
_FFMPEG_CANDIDATES = [
    _REPO_ROOT / "ffmpeg-8.1.1-essentials_build" / "ffmpeg-8.1.1-essentials_build" / "bin" / "ffmpeg.exe",
    _REPO_ROOT / "ffmpeg-8.1.1-full_build-shared" / "ffmpeg-8.1.1-full_build-shared" / "bin" / "ffmpeg.exe",
]
_FFMPEG_BIN = next((str(p) for p in _FFMPEG_CANDIDATES if p.exists()), "ffmpeg")


def convert_to_wav(input_path: Path, output_path: Path) -> Path:
    """Convert any audio format to 44100 Hz mono WAV via FFmpeg."""
    cmd = [
        _FFMPEG_BIN, "-y",
        "-i", str(input_path),
        "-ar", "44100",
        "-ac", "1",
        "-f", "wav",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        # Fallback: use librosa if FFmpeg fails for any reason
        y, sr = librosa.load(str(input_path), sr=44100, mono=True)
        sf.write(str(output_path), y, sr, subtype="PCM_16")
    return output_path


def get_audio_duration(wav_path: Path) -> float:
    y, sr = librosa.load(str(wav_path), sr=None)
    return librosa.get_duration(y=y, sr=sr)


def compute_mel_spectrogram(wav_path: Path, sr: int = 22050) -> np.ndarray:
    y, _ = librosa.load(str(wav_path), sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db


def compute_mel_from_array(y: np.ndarray, sr: int = 22050) -> np.ndarray:
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    return librosa.power_to_db(mel, ref=np.max)


def detect_tempo_and_beats(wav_path: Path):
    y, sr = librosa.load(str(wav_path), sr=22050)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    return float(tempo), [float(t) for t in beat_times]


def render_notes_to_audio(notes: list, sr: int = 22050) -> np.ndarray:
    """
    Piano-like additive synthesis.
    Uses multiple harmonics with individual exponential decays and slight
    inharmonicity (B coefficient) to approximate struck-string timbre.
    """
    if not notes:
        return np.zeros(sr, dtype=np.float32)

    max_time = max(float(n.get("offset", n["onset"] + 0.5)) for n in notes)
    total_samples = int((max_time + 2.0) * sr)
    audio = np.zeros(total_samples, dtype=np.float32)

    # Harmonic partials: (relative_amplitude, decay_rate_multiplier)
    # Mimics the bright attack and fast upper-harmonic decay of a piano
    PARTIALS = [
        (1.00, 1.0),   # fundamental
        (0.50, 1.8),   # 2nd harmonic (octave)
        (0.25, 2.8),   # 3rd
        (0.12, 4.0),   # 4th
        (0.06, 5.5),   # 5th
        (0.03, 7.5),   # 6th
    ]
    # Inharmonicity coefficient — slightly stretches upper partials like real strings
    B = 0.00015

    for n in notes:
        midi = float(n["pitch"])
        freq = 440.0 * (2.0 ** ((midi - 69.0) / 12.0))
        amp = float(n.get("velocity", 80)) / 127.0

        onset = int(float(n["onset"]) * sr)
        # Decay lasts longer than the nominal note length for natural ring-out
        note_dur = float(n.get("offset", n["onset"] + 0.5)) - float(n["onset"])
        decay_dur = note_dur + min(2.0, note_dur * 1.5)
        length = min(int(decay_dur * sr), total_samples - onset)
        if length <= 0:
            continue

        t = np.arange(length) / sr

        # Very short attack ramp (2 ms) — piano hammer is fast
        attack_len = min(int(0.002 * sr), length)
        attack_ramp = np.ones(length)
        attack_ramp[:attack_len] = np.linspace(0.0, 1.0, attack_len)

        # Release: fade last 30 ms
        release_len = min(int(0.03 * sr), length // 4)
        attack_ramp[-release_len:] *= np.linspace(1.0, 0.0, release_len)

        wave = np.zeros(length, dtype=np.float64)
        for k, (rel_amp, decay_mult) in enumerate(PARTIALS, start=1):
            # Inharmonic partial frequency
            f_k = freq * k * np.sqrt(1.0 + B * k * k)
            # Exponential decay — higher partials die faster
            # Lower notes decay more slowly than higher notes
            base_decay = 2.5 + (midi / 127.0) * 3.5
            decay = np.exp(-base_decay * decay_mult * t)
            wave += rel_amp * decay * np.sin(2.0 * np.pi * f_k * t)

        wave *= attack_ramp * amp
        end = onset + length
        audio[onset:end] += wave.astype(np.float32)

    # Soft limiter instead of hard clip
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / max(peak, 1.0) * 0.92
    return audio.astype(np.float32)


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence between two 1-D distributions."""
    from scipy.spatial.distance import jensenshannon
    p_flat = p.flatten()
    q_flat = q.flatten()

    # Normalize to same length via interpolation if needed
    if len(p_flat) != len(q_flat):
        target = min(len(p_flat), len(q_flat))
        p_flat = np.interp(np.linspace(0, 1, target), np.linspace(0, 1, len(p_flat)), p_flat)
        q_flat = np.interp(np.linspace(0, 1, target), np.linspace(0, 1, len(q_flat)), q_flat)

    # Shift to non-negative and normalize to probability distributions
    p_flat = p_flat - p_flat.min() + 1e-10
    q_flat = q_flat - q_flat.min() + 1e-10
    p_flat = p_flat / p_flat.sum()
    q_flat = q_flat / q_flat.sum()

    return float(jensenshannon(p_flat, q_flat))
