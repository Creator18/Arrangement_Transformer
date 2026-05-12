"""
Demucs-based source separation.

Replaces the previous NMF pipeline.  Demucs is a neural source separator
that splits arbitrary audio into 4 instrument stems (drums, bass, vocals,
other).  For our piano-arrangement task we keep the harmonic content (bass
+ vocals + other, discarding drums) then bandpass-split that into a bass
register and an upper register so the downstream per-register transcribers
(pYIN on bass, Basic Pitch on upper) keep working unchanged.

This file is consciously a drop-in replacement for the NMF version: same
inputs, same outputs on disk (`components/bass.wav`, `components/upper.wav`,
`components.json` with tempo / sr / hop / beat_frames).  Downstream code in
transcription.py, segmentation.py, and routers/transform.py needs no
behaviour changes — just the H.npy/W.npy and per-component metadata are
gone, which transcription.py handles via a feature-flag-free path.

Why Demucs over Spleeter
------------------------
PyTorch-native, actively maintained, higher SDR on modern benchmarks.  No
TensorFlow dependency to drag into a PyTorch project.

Performance note
----------------
Demucs is ~30–60 s per 3-minute song on CPU, much faster on GPU.  Model
weights (~100 MB) download lazily on first call into the torch cache dir.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, sosfiltfilt


# ── constants ─────────────────────────────────────────────────────────────────
SR  = 22050          # output sample rate (must match downstream services)
HOP = 512            # output hop length  (must match downstream services)

# Bandpass split between bass and upper register.  G3 ≈ 196 Hz, so a cutoff
# at 240 Hz with a soft slope captures bass piano notes up to ~A3/B3.  pYIN
# narrows further with fmin/fmax inside transcription.py.
BAND_SPLIT_HZ = 240.0
FILTER_ORDER  = 6     # Butterworth — gentle slope, no ringing


# Lazy global so we only pay the model-construction cost once per worker.
_separator = None


def _get_separator():
    """Construct (or fetch) the Demucs separator.  Lazy + cached."""
    global _separator
    if _separator is not None:
        return _separator
    # `htdemucs` is the current default — 4-source, hybrid model.
    from demucs.api import Separator
    _separator = Separator(model="htdemucs")
    return _separator


def _butter_sos(cutoff_hz: float, sr: int, kind: str) -> np.ndarray:
    """
    Build a stable second-order-section Butterworth filter.  We use SOS form
    + sosfiltfilt (zero-phase) so the split doesn't introduce group delay
    that would shift onsets and confuse beat tracking later.
    """
    return butter(FILTER_ORDER, cutoff_hz / (sr / 2), btype=kind, output="sos")


def _to_mono(stereo: np.ndarray) -> np.ndarray:
    """(channels, samples) → (samples,) by channel-mean."""
    if stereo.ndim == 1:
        return stereo
    return stereo.mean(axis=0)


def _normalise(audio: np.ndarray) -> np.ndarray:
    peak = float(np.abs(audio).max())
    if peak > 1e-6:
        audio = audio / peak
    return audio.astype(np.float32)


def separate_components(wav_path: Path, session_dir: Path) -> Dict[str, Any]:
    """
    Run Demucs over the input, build bass/upper register stems, and save the
    same on-disk layout the rest of the pipeline expects.

    Side effects:
        session_dir/components/bass.wav
        session_dir/components/upper.wav
        session_dir/components.json
    """
    sep = _get_separator()

    # ── 1. Demucs separation ──────────────────────────────────────────────
    # The Separator handles loading, resampling, and chunked inference for us.
    # Returns (origin, stems_dict).  Each stem is a torch tensor shaped
    # (channels, samples) at the model's native sample rate.
    origin, stems = sep.separate_audio_file(str(wav_path))
    sr_model = int(sep.samplerate)

    def _stem_np(name: str) -> np.ndarray:
        t = stems[name].cpu().numpy()
        return _to_mono(t)

    # ── 2. Drop drums; keep harmonic content ───────────────────────────────
    # Vocals carry melody in real songs, so they stay.  Drums add transients
    # that confuse pitch trackers, so they go.
    harmonic = _stem_np("bass") + _stem_np("other") + _stem_np("vocals")

    # ── 3. Resample to our pipeline's working sample rate ─────────────────
    if sr_model != SR:
        harmonic = librosa.resample(harmonic, orig_sr=sr_model, target_sr=SR)

    # ── 4. Bandpass split into bass and upper register ────────────────────
    # zero-phase Butterworth → no onset shift in either band.
    sos_lo = _butter_sos(BAND_SPLIT_HZ, SR, "low")
    sos_hi = _butter_sos(BAND_SPLIT_HZ, SR, "high")
    y_bass  = sosfiltfilt(sos_lo, harmonic).astype(np.float32)
    y_upper = sosfiltfilt(sos_hi, harmonic).astype(np.float32)

    y_bass  = _normalise(y_bass)
    y_upper = _normalise(y_upper)

    # ── 5. Beat track on the harmonic mix ─────────────────────────────────
    # Drumless input gives a much cleaner beat curve than the raw audio,
    # which carries a lot of drum transients in real songs.
    tempo_arr, beat_frames = librosa.beat.beat_track(y=harmonic, sr=SR, hop_length=HOP)
    tempo = float(np.atleast_1d(tempo_arr)[0])
    if tempo <= 0:
        tempo = 120.0

    # ── 6. Write outputs ──────────────────────────────────────────────────
    comp_dir = session_dir / "components"
    comp_dir.mkdir(exist_ok=True)

    bass_path  = comp_dir / "bass.wav"
    upper_path = comp_dir / "upper.wav"
    sf.write(str(bass_path),  y_bass,  SR, subtype="PCM_16")
    sf.write(str(upper_path), y_upper, SR, subtype="PCM_16")

    metadata: Dict[str, Any] = {
        "method":           "demucs",
        "tempo_bpm":        round(tempo, 1),
        "hop_length":       HOP,
        "sr":               SR,
        "beat_frames":      [int(f) for f in beat_frames.tolist()],
        "band_split_hz":    BAND_SPLIT_HZ,
        "groups": {
            "bass":  {"path": str(bass_path),  "indices": []},  # kept for back-compat
            "upper": {"path": str(upper_path), "indices": []},
        },
        # Legacy fields kept as empty/zero so old transcription.py code that
        # peeks at them doesn't KeyError.  Phase-3 transcription.py update
        # removes those references entirely.
        "n_components":   0,
        "components":     [],
    }

    with open(session_dir / "components.json", "w") as f:
        json.dump(metadata, f)

    return metadata
