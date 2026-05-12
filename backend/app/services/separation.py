"""
Source separation + beat tracking.

    1. Demucs (htdemucs) splits the input into drums / bass / vocals / other.
    2. The drum stem is discarded; the remaining three are summed and
       bandpass-split at ~240 Hz into bass and upper register stems for the
       per-register transcribers (pYIN on bass, Basic Pitch on upper).
    3. The drum-less harmonic mix is fed through madmom's RNN + DBN tracker,
       which simultaneously produces beat positions, measure downbeats, an
       implied time signature, and a tempo estimate.

Outputs land on disk in the layout downstream code expects:
    session_dir/components/bass.wav
    session_dir/components/upper.wav
    session_dir/components.json   (tempo, beats, downbeats, sr, hop, …)

The components.json `method` field reads "demucs+madmom" so a downstream
log can tell which pipeline produced a given session's metadata.

Why this stack
--------------
Demucs is PyTorch-native, actively maintained, and outscores Spleeter on
modern source-separation benchmarks.  madmom's RNN beat tracker is roughly
2× more accurate than librosa.beat.beat_track on real music, and its DBN
downbeat tracker is the only off-the-shelf option that returns measure
boundaries rather than just beat positions.

Performance notes
-----------------
Demucs is ~30–60 s per 3-minute track on CPU, much faster on GPU.  Model
weights (~100 MB) download lazily on first call into the torch cache dir.
madmom adds another ~5–10 s for beat tracking; its models are bundled with
the pip package (~50 MB).
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

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


# Lazy globals for madmom processors — building these is non-trivial (each
# pulls a ~10 MB neural network checkpoint into memory) so we cache them
# across calls in the same worker.
_madmom_rnn  = None
_madmom_dbn  = None


def _get_madmom():
    """Construct (or fetch) the madmom downbeat tracker.  Lazy + cached."""
    global _madmom_rnn, _madmom_dbn
    if _madmom_rnn is not None and _madmom_dbn is not None:
        return _madmom_rnn, _madmom_dbn
    # madmom 0.16.1 needs collections/numpy alias restorations and an
    # in-place patch to DBNDownBeatTrackingProcessor.process — see
    # _madmom_compat for the why.  All idempotent.
    from app.services import _madmom_compat
    from madmom.features.downbeats import (
        RNNDownBeatProcessor,
        DBNDownBeatTrackingProcessor,
    )
    _madmom_compat.apply_madmom_runtime_patches()
    _madmom_rnn = RNNDownBeatProcessor()
    # beats_per_bar=[3, 4] lets the DBN choose between 3/4 and 4/4 per piece.
    # fps=100 matches madmom's RNN frame rate; do not change without retraining.
    _madmom_dbn = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
    return _madmom_rnn, _madmom_dbn


def _madmom_beats(audio: np.ndarray, sr: int) -> Tuple[List[int], List[int], float, int]:
    """
    Beat + downbeat tracking on the drum-less harmonic mix.

    Returns:
        beat_frames      list[int]  frame indices at hop_length=HOP
        downbeat_frames  list[int]  frame indices of measure starts (1-beats)
        tempo_bpm        float      median tempo across the piece
        beats_per_bar    int        whichever of {3, 4} madmom chose

    madmom's DBN tracker simultaneously decides beat positions AND the
    time signature (3/4 vs 4/4) on a per-section basis.  We extract the
    most common bar length as a single integer for downstream metadata —
    if a piece switches meters mid-way, this picks whichever dominates.

    Implementation note
    -------------------
    madmom's RNN processor is happiest with a file path.  When given a raw
    numpy array at a non-44100 sample rate it falls back to an internal
    ffmpeg-based resampler — which fails if ffmpeg isn't on the PATH.  We
    sidestep that by writing the harmonic mix to a temp WAV and handing
    madmom the path; WAVs route through scipy/libsndfile internally, no
    external binary required.
    """
    # _get_madmom installs the collections + numpy shims before any madmom import.
    rnn, dbn = _get_madmom()

    # madmom's RNNDownBeatProcessor hard-codes sample_rate=44100 in its
    # SignalProcessor.  When the file on disk doesn't match, load_wave_file
    # raises and madmom falls back to an ffmpeg subprocess — which we don't
    # have on PATH.  Pre-resample to 44100 so the WAV path is the happy path.
    MADMOM_SR = 44100
    if sr != MADMOM_SR:
        audio = librosa.resample(audio.astype(np.float32),
                                 orig_sr=sr, target_sr=MADMOM_SR)

    fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="madmom_in_")
    os.close(fd)
    try:
        sf.write(tmp_path, audio.astype(np.float32), MADMOM_SR, subtype="PCM_16")
        activations = rnn(tmp_path)                  # (T, 2)  beat + downbeat probs
        result      = dbn(activations)               # (N, 2)  [time_s, beat_pos]
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if len(result) == 0:
        return [], [], 120.0, 4

    beat_times     = result[:, 0]
    beat_positions = result[:, 1].astype(int)
    downbeat_times = beat_times[beat_positions == 1]

    # Tempo: median inter-beat-interval inverted.  Robust to occasional
    # mis-detections that would skew a simple mean.
    if len(beat_times) >= 2:
        median_ibi = float(np.median(np.diff(beat_times)))
        tempo = 60.0 / median_ibi if median_ibi > 1e-3 else 120.0
    else:
        tempo = 120.0

    # Time signature: the maximum beat position seen tells us the bar length.
    beats_per_bar = int(beat_positions.max()) if beat_positions.size else 4
    if beats_per_bar not in (3, 4):                  # belt-and-braces clamp
        beats_per_bar = 4

    # Convert seconds → frame indices in our pipeline's hop_length grid.
    # NB: SR (module constant) is the *output* frame grid — downstream code
    # expects beat_frames at SR/HOP resolution.  `sr`/`MADMOM_SR` above are
    # only the internal audio rate used to feed madmom's models.
    beat_frames     = [int(round(t * SR / HOP)) for t in beat_times]
    downbeat_frames = [int(round(t * SR / HOP)) for t in downbeat_times]
    return beat_frames, downbeat_frames, tempo, beats_per_bar


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

    # ── 5. Beat + downbeat tracking via madmom on the harmonic mix ────────
    # madmom's RNN beat tracker is roughly 2× more accurate than
    # librosa.beat.beat_track on real music, AND its DBN downbeat tracker
    # gives us measure starts (not just beat boundaries) and a time-signature
    # estimate in one pass.  Drumless input keeps drum transients from
    # confusing onset detection.
    beat_frames, downbeat_frames, tempo, beats_per_bar = _madmom_beats(harmonic, SR)

    # ── 6. Write outputs ──────────────────────────────────────────────────
    comp_dir = session_dir / "components"
    comp_dir.mkdir(exist_ok=True)

    bass_path  = comp_dir / "bass.wav"
    upper_path = comp_dir / "upper.wav"
    sf.write(str(bass_path),  y_bass,  SR, subtype="PCM_16")
    sf.write(str(upper_path), y_upper, SR, subtype="PCM_16")

    metadata: Dict[str, Any] = {
        "method":           "demucs+madmom",
        "tempo_bpm":        round(tempo, 1),
        "hop_length":       HOP,
        "sr":               SR,
        "beat_frames":      [int(f) for f in beat_frames],
        "downbeat_frames":  [int(f) for f in downbeat_frames],
        "beats_per_bar":    int(beats_per_bar),
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
