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
    """
    Construct (or fetch) the Demucs htdemucs model.  Lazy + cached.

    We use the low-level get_model / apply_model API rather than
    demucs.api.Separator because the api.py module wasn't shipped in
    demucs 4.0.1 (latest on PyPI); the high-level wrapper only exists
    on main.  The low-level API is stable across versions.
    """
    global _separator
    if _separator is not None:
        return _separator
    import torch
    from demucs.pretrained import get_model
    model = get_model("htdemucs")
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    _separator = model
    return _separator


def _demucs_separate(audio_44k_stereo, model):
    """Run the cached Demucs model over a (2, samples) tensor at 44100 Hz."""
    import torch
    from demucs.apply import apply_model
    device = next(model.parameters()).device
    mix = audio_44k_stereo.to(device)
    with torch.no_grad():
        stems = apply_model(model, mix.unsqueeze(0))
    stems = stems[0].cpu().numpy()                    # (4, 2, samples)
    return {name: stems[i] for i, name in enumerate(model.sources)}


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
    Beat + downbeat tracking. Give this the FULL MIX on songs: madmom is
    trained on music with drums, and a drumless mix is out of domain for it.

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


# One switch to put Demucs back on the critical path for everything.
SKIP_DEMUCS_FOR_PIANO = True


def _separate_piano_only(wav_path: Path, session_dir: Path,
                         content: Dict[str, Any]) -> Dict[str, Any]:
    """
    Solo-piano path: beat-track the raw mix and write the same on-disk layout,
    without running Demucs at all.

    `bass.wav` and `upper.wav` are the raw mix here rather than stems.  That
    is honest for this content — there is nothing to separate — and it keeps
    every downstream consumer working unchanged, since transcription reads the
    full mix anyway under of_fullmix.
    """
    import librosa
    import soundfile as sf

    audio, _ = librosa.load(str(wav_path), sr=SR, mono=True)
    beat_frames, downbeat_frames, tempo, beats_per_bar = _madmom_beats(audio, SR)

    comp_dir = session_dir / "components"
    comp_dir.mkdir(parents=True, exist_ok=True)
    for name in ("bass", "upper"):
        sf.write(str(comp_dir / f"{name}.wav"), audio, SR, subtype="PCM_16")

    meta = {
        "method": "raw+madmom",          # not "demucs+madmom" — say what ran
        "sr": SR,
        "hop_length": 512,
        "beat_frames": list(map(int, beat_frames)),
        "downbeat_frames": list(map(int, downbeat_frames)),
        "beats_per_bar": int(beats_per_bar),
        "tempo_bpm": float(tempo),
        "content": content,
        "separation_skipped": True,
    }
    with open(session_dir / "components.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[separation] solo piano — Demucs skipped, tempo {tempo:.1f}")
    return meta


def separate_components(wav_path: Path, session_dir: Path) -> Dict[str, Any]:
    """
    Run Demucs over the input, build bass/upper register stems, and save the
    same on-disk layout the rest of the pipeline expects.

    Side effects:
        session_dir/components/bass.wav
        session_dir/components/upper.wav
        session_dir/components.json
    """
    import torch
    import torchaudio
    import librosa as _lb

    # ── 0. Is there anything to separate? ─────────────────────────────────
    # On solo piano the "drum-less harmonic mix" Demucs produces IS the input:
    # measured, beat-tracking the raw mix reproduces the Demucs-derived grid
    # exactly (F = 1.000) for zero cost.  So Demucs is skipped when the
    # content is confidently piano-only, and runs otherwise.
    #
    # The detector is deliberately biased: anything not confidently piano
    # falls through to the slow, safe path, because a false "piano" degrades
    # the arrangement silently while a false "song" only costs ~10 s.
    from app.services.content_detect import analyse as _detect
    _probe, _psr = _lb.load(str(wav_path), sr=22050, mono=True)
    content = _detect(_probe, _psr)
    print(f"[separation] content: {content}")
    if SKIP_DEMUCS_FOR_PIANO and content.get("piano_only"):
        return _separate_piano_only(wav_path, session_dir, content)

    model = _get_separator()
    sr_model = int(model.samplerate)
    n_channels = int(model.audio_channels)

    # ── 1. Load audio at Demucs's native sample rate (44.1 kHz, stereo) ───
    audio, orig_sr = torchaudio.load(str(wav_path))     # (channels, samples)
    if orig_sr != sr_model:
        audio = torchaudio.functional.resample(audio, orig_sr, sr_model)
    if audio.shape[0] == 1:
        audio = audio.repeat(n_channels, 1)             # mono → stereo
    elif audio.shape[0] > n_channels:
        audio = audio[:n_channels]

    # ── 2. Run Demucs ──────────────────────────────────────────────────────
    stems = _demucs_separate(audio, model)              # {name: (2, samples)}

    def _stem_np(name: str) -> np.ndarray:
        return _to_mono(stems[name])

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

    # ── 5. Beat + downbeat tracking via madmom on the FULL MIX ────────────
    # madmom's RNN beat tracker is roughly 2x more accurate than
    # librosa.beat.beat_track on real music, AND its DBN downbeat tracker gives
    # us measure starts (not just beat boundaries) and a time-signature estimate
    # in one pass.
    #
    # It used to be fed `harmonic` — the mix with drums REMOVED — on the
    # reasoning that drum transients confuse onset detection.  That is true of
    # PITCH onset detection and false of beat tracking, where the drums are the
    # beat.  madmom's RNN is trained on real music with drums, so a drumless mix
    # is out of domain and it locks onto whatever periodicity the guitars and
    # vocals happen to offer.
    #
    # Measured (eval/beat_source_songs.py), against drum-stem attacks:
    #
    #     source              tempo   drum fit   median dev
    #     drumless (old)      136.4        36%      162.5 ms
    #     drums only          139.5        57%       46.4 ms
    #     full mix            139.5        56%       46.4 ms
    #
    # And the old grid scores beat F-measure 0.000 against all three others,
    # which agree with each other at 1.000 — same beat count and tempo, so it
    # was not mistracking the pulse, it was PHASE-SHIFTED about a third of a
    # beat off it.  Four stages hang on this grid (beat snapping, the
    # arpeggiator's bar edges, the melody decode's metrical prior, and the
    # 16-slot rhythm retrieval), so all four were quantising precisely to the
    # wrong phase.
    #
    # Full mix over drums-only because they are indistinguishable here
    # (F = 1.000) and the full mix needs no separation, so beat tracking no
    # longer depends on Demucs having succeeded.
    _mix_mono, _ = librosa.load(str(wav_path), sr=SR, mono=True)
    beat_frames, downbeat_frames, tempo, beats_per_bar = _madmom_beats(_mix_mono, SR)

    # ── 6. REPET-SIM split on BOTH bands ──────────────────────────────────
    # Upper band: pattern.wav + melody.wav (transcribed independently by
    #   Basic Pitch in transcription.py).
    # Bass  band: bass_pattern.wav + bass_melody.wav  (audio outputs for
    #   the Layers tab) plus a per-frame pattern_strength array that
    #   transcription.py uses to relabel pYIN notes as bass_pattern /
    #   bass_melody — pYIN stays the bass transcriber (more reliable than
    #   running Basic Pitch on bass), REPET only decides the labels.
    #
    # Running REPET independently on each band (rather than on the full
    # mix) keeps the similarity matrix focused on the right register and
    # avoids bass energy dominating the upper-band split.  See ROADMAP
    # notes from phase 5c on why full-mix REPET failed.
    from app.services.pattern_separation import repet_sim_separate

    def _safe_norm(a):
        peak = float(np.abs(a).max())
        return (a / peak * 0.9).astype(np.float32) if peak > 1e-6 else a.astype(np.float32)

    try:
        y_pattern, y_melody = repet_sim_separate(y_upper, SR)
        y_pattern = _safe_norm(y_pattern)
        y_melody  = _safe_norm(y_melody)
        upper_repet_ok = True
    except Exception as e:
        print(f"[separation] REPET-SIM (upper) failed ({e}); skipping pattern/melody split")
        y_pattern, y_melody = None, None
        upper_repet_ok = False

    try:
        y_bass_pattern, y_bass_melody, bass_pat_strength, bass_frame_times = \
            repet_sim_separate(y_bass, SR, return_mask=True)
        y_bass_pattern = _safe_norm(y_bass_pattern)
        y_bass_melody  = _safe_norm(y_bass_melody)
        bass_repet_ok = True
    except Exception as e:
        print(f"[separation] REPET-SIM (bass) failed ({e}); skipping bass-layer split")
        y_bass_pattern, y_bass_melody = None, None
        bass_pat_strength, bass_frame_times = None, None
        bass_repet_ok = False

    # ── 7. Write outputs ──────────────────────────────────────────────────
    comp_dir = session_dir / "components"
    comp_dir.mkdir(exist_ok=True)

    bass_path  = comp_dir / "bass.wav"
    upper_path = comp_dir / "upper.wav"
    sf.write(str(bass_path),  y_bass,  SR, subtype="PCM_16")
    sf.write(str(upper_path), y_upper, SR, subtype="PCM_16")

    # The drumless mix, with its band balance intact.
    #
    # bass.wav and upper.wav are each peak-normalised INDEPENDENTLY, which is
    # right for the transcribers that read one band at a time and wrong for
    # anything that adds them back together: measured on a real upload, that
    # sum carries 1.54x more bass energy than the mix it came from.
    # piano_projection was doing exactly that, and the low register swamped
    # everything — 7.29 notes/s below G#3 against 0.77 above it, where the eval
    # (which used the properly balanced signal) had them roughly equal.
    sf.write(str(comp_dir / "harmonic.wav"), _normalise(harmonic), SR,
             subtype="PCM_16")

    groups: Dict[str, Any] = {
        "bass":  {"path": str(bass_path),  "indices": []},
        "upper": {"path": str(upper_path), "indices": []},
    }

    if upper_repet_ok:
        pattern_path = comp_dir / "pattern.wav"
        melody_path  = comp_dir / "melody.wav"
        sf.write(str(pattern_path), y_pattern, SR, subtype="PCM_16")
        sf.write(str(melody_path),  y_melody,  SR, subtype="PCM_16")
        groups["pattern"] = {"path": str(pattern_path), "indices": []}
        groups["melody"]  = {"path": str(melody_path),  "indices": []}

    if bass_repet_ok:
        bass_pattern_path = comp_dir / "bass_pattern.wav"
        bass_melody_path  = comp_dir / "bass_melody.wav"
        sf.write(str(bass_pattern_path), y_bass_pattern, SR, subtype="PCM_16")
        sf.write(str(bass_melody_path),  y_bass_melody,  SR, subtype="PCM_16")
        groups["bass_pattern"] = {"path": str(bass_pattern_path), "indices": []}
        groups["bass_melody"]  = {"path": str(bass_melody_path),  "indices": []}

        # Persist the bass pattern_strength array so transcription.py can label
        # pYIN notes without re-running REPET.  Compact .npz alongside the WAVs.
        np.savez(
            comp_dir / "bass_pattern_mask.npz",
            strength=bass_pat_strength.astype(np.float32),
            times=bass_frame_times.astype(np.float32),
        )

    # ── 7b. Cluster each pattern stem into Pattern 1 … N ──────────────────
    # Bar-windowed agglomerative clustering on chroma + onset-shape features.
    # Lets the Layers UI present each recurring figure as its own piano roll
    # for review.  See pattern_clustering.py for the algorithm.
    from app.services.pattern_clustering import cluster_pattern_stem, write_cluster_audio

    downbeat_times = (np.array(downbeat_frames, dtype=float) * HOP / SR) \
        if downbeat_frames else np.array([])
    cluster_results: Dict[str, Any] = {}

    for stem_name, y_stem in (
        ("pattern",      y_pattern      if upper_repet_ok else None),
        ("bass_pattern", y_bass_pattern if bass_repet_ok else None),
    ):
        if y_stem is None or len(downbeat_times) < 2:
            cluster_results[stem_name] = {"n_clusters": 0, "bars": [],
                                          "granularity": "bar",
                                          "cluster_bar_counts": {}}
            continue
        try:
            cmeta = cluster_pattern_stem(y_stem, SR, downbeat_times)
            paths = write_cluster_audio(y_stem, SR, cmeta, comp_dir, stem_name=stem_name)
            cmeta["audio_paths"] = {str(k): str(v) for k, v in paths.items()}
            cluster_results[stem_name] = cmeta
            print(f"[separation] {stem_name}: {cmeta['n_clusters']} clusters "
                  f"({cmeta['granularity']} granularity, "
                  f"{len(cmeta['bars'])} windows)")
        except Exception as e:
            print(f"[separation] clustering failed for {stem_name}: {e}")
            cluster_results[stem_name] = {"n_clusters": 0, "bars": [],
                                          "granularity": "bar",
                                          "cluster_bar_counts": {}}

    with open(session_dir / "pattern_clusters.json", "w") as f:
        json.dump(cluster_results, f)

    method_parts = ["demucs", "madmom"]
    if upper_repet_ok or bass_repet_ok:
        method_parts.append("repet-sim")

    metadata: Dict[str, Any] = {
        "method":           "+".join(method_parts),
        "tempo_bpm":        round(tempo, 1),
        "hop_length":       HOP,
        "sr":               SR,
        "beat_frames":      [int(f) for f in beat_frames],
        "downbeat_frames":  [int(f) for f in downbeat_frames],
        "beats_per_bar":    int(beats_per_bar),
        "band_split_hz":    BAND_SPLIT_HZ,
        "groups":           groups,
        "has_bass_repet":   bool(bass_repet_ok),
        "has_upper_repet":  bool(upper_repet_ok),
        # Legacy fields kept as empty/zero so old transcription.py code that
        # peeks at them doesn't KeyError.
        "n_components":   0,
        "components":     [],
    }

    with open(session_dir / "components.json", "w") as f:
        json.dump(metadata, f)

    return metadata
