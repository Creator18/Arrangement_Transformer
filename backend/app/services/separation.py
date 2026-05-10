"""
NMF-based audio component separation for piano-only recordings.

Pipeline:
  1. Compute STFT magnitude spectrogram of the raw audio
  2. NMF decomposition → K spectral bases (W) × temporal activations (H)
  3. Classify each component k:
       - Periodicity: normalised autocorrelation of H[k] at beat-aligned lags
       - Register:    frequency centroid of W[k] mapped to MIDI pitch
  4. Group components into bass (< MIDI 56) and upper (>= MIDI 56)
  5. Reconstruct per-group audio via Wiener soft masking on the original STFT
  6. Save group audio files + H matrix + component metadata

The H matrix is saved as components/H.npy so that transcription can label
each transcribed note with the dominant component at its onset time,
connecting the audio-domain pattern detection to the note domain.
"""
from __future__ import annotations

import json
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Any
from sklearn.decomposition import NMF

# ── constants ─────────────────────────────────────────────────────────────────
SR           = 22050
N_FFT        = 4096
HOP          = 512
N_COMPONENTS = 12       # NMF components — enough voices for a full piano texture
BASS_MIDI    = 56       # G#3 — below this is bass register
PERIODICITY_THRESH = 0.30   # min autocorrelation to be "pattern"


# ── helpers ───────────────────────────────────────────────────────────────────

def _periodicity_score(h_row: np.ndarray, beat_period_frames: float) -> float:
    """
    Normalised autocorrelation of a single NMF activation row at beat-aligned
    lags (1, 2, 4, 8 beats).  Returns the maximum value found.
    A repeating rhythmic/harmonic pattern produces a high score; a freely
    moving melody line produces a low score.
    """
    # Smooth with a short uniform window to reduce quantisation noise
    envelope = np.convolve(h_row, np.ones(max(1, int(beat_period_frames // 8))), mode='same')
    envelope = envelope / (envelope.max() + 1e-8)

    mu = envelope.mean()
    centred = envelope - mu
    denom = float(np.dot(centred, centred)) + 1e-8

    scores: List[float] = []
    for mult in [1, 2, 4, 8]:
        lag = int(round(mult * beat_period_frames))
        if 0 < lag < len(centred):
            scores.append(float(np.dot(centred[:-lag], centred[lag:])) / denom)

    return max(scores) if scores else 0.0


def _centroid_midi(w_col: np.ndarray, freqs: np.ndarray) -> float:
    """Weighted-mean frequency of a spectral basis vector, expressed as MIDI."""
    total = w_col.sum() + 1e-8
    hz = float(np.dot(w_col, freqs) / total)
    return float(librosa.hz_to_midi(max(hz, 20.0)))


# ── main entry point ──────────────────────────────────────────────────────────

def separate_components(wav_path: Path, session_dir: Path) -> Dict[str, Any]:
    """
    Separate piano audio into bass and upper component groups using NMF.

    Saves to session_dir/components/:
      bass.wav    — reconstructed bass-register audio
      upper.wav   — reconstructed upper-register audio
      H.npy       — temporal activation matrix (K, T), float32
      W.npy       — spectral basis matrix  (F, K), float32

    Returns and writes components.json with metadata for downstream steps.
    """
    y, sr = librosa.load(str(wav_path), sr=SR, mono=True)

    # Tempo + actual beat frame positions for downstream beat-adaptive quantization
    tempo_arr, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP)
    tempo = float(np.atleast_1d(tempo_arr)[0])
    if tempo <= 0:
        tempo = 120.0
    beat_period_frames = (60.0 / tempo) * sr / HOP

    # STFT — complex spectrogram kept for Wiener masking
    D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP)
    S = np.abs(D).astype(np.float32)           # (F, T) magnitude
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)

    # NMF on transposed spectrogram: sklearn expects (n_samples, n_features)
    K = min(N_COMPONENTS, S.shape[0] - 1, S.shape[1] - 1)
    model = NMF(n_components=K, max_iter=500, random_state=0, init='nndsvda', tol=1e-4)
    H_sk = model.fit_transform(S.T)   # (T, K)
    W_sk = model.components_          # (K, F)

    # Reshape to natural convention: W (F, K),  H (K, T)
    W = W_sk.T.astype(np.float32)     # (F, K)
    H = H_sk.T.astype(np.float32)     # (K, T)

    # ── classify each component ───────────────────────────────────────────────
    components: List[Dict[str, Any]] = []
    for k in range(K):
        p_score      = _periodicity_score(H[k], beat_period_frames)
        midi_centroid = _centroid_midi(W[:, k], freqs)
        register     = 'bass' if midi_centroid < BASS_MIDI else 'upper'
        components.append({
            'k':              k,
            'periodicity':    round(float(p_score), 4),
            'centroid_midi':  round(float(midi_centroid), 1),
            'register':       register,
            'is_pattern':     bool(p_score >= PERIODICITY_THRESH),
        })

    bass_indices  = [c['k'] for c in components if c['register'] == 'bass']
    upper_indices = [c['k'] for c in components if c['register'] == 'upper']

    # ── Wiener soft masking ───────────────────────────────────────────────────
    # Reconstruct the full spectrogram approximation (denominator)
    V_approx = W @ H + 1e-8   # (F, T)

    def _group_audio(indices: List[int]) -> np.ndarray:
        if not indices:
            return np.zeros(len(y), dtype=np.float32)
        W_g = W[:, indices]         # (F, n)
        H_g = H[indices, :]         # (n, T)
        V_g = W_g @ H_g             # (F, T)
        mask = V_g / V_approx       # soft Wiener mask
        y_g  = librosa.istft(D * mask, hop_length=HOP)
        peak = np.abs(y_g).max()
        if peak > 0:
            y_g = y_g / peak
        return y_g.astype(np.float32)

    y_bass  = _group_audio(bass_indices)
    y_upper = _group_audio(upper_indices)

    # ── save ──────────────────────────────────────────────────────────────────
    comp_dir = session_dir / 'components'
    comp_dir.mkdir(exist_ok=True)

    bass_path  = comp_dir / 'bass.wav'
    upper_path = comp_dir / 'upper.wav'
    sf.write(str(bass_path),  y_bass,  SR, subtype='PCM_16')
    sf.write(str(upper_path), y_upper, SR, subtype='PCM_16')

    np.save(str(comp_dir / 'H.npy'), H)
    np.save(str(comp_dir / 'W.npy'), W)

    metadata: Dict[str, Any] = {
        'tempo_bpm':    round(tempo, 1),
        'n_components': K,
        'hop_length':   HOP,
        'sr':           SR,
        'beat_frames':  beat_frames.tolist(),
        'components':   components,
        'groups': {
            'bass':  {'path': str(bass_path),  'indices': bass_indices},
            'upper': {'path': str(upper_path), 'indices': upper_indices},
        },
    }

    with open(session_dir / 'components.json', 'w') as f:
        json.dump(metadata, f)

    return metadata
