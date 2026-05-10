"""
Chroma-based chord detection and audio segmentation.

Chord information is extracted directly from the raw audio using CQT chroma
features — not from transcription notes.  This is the most reliable source
of harmonic information for piano-only recordings because:
  - Chroma captures the full harmonic energy regardless of note accuracy
  - It is robust to transcription octave errors and missed notes
  - librosa's chroma_cqt uses a constant-Q basis tuned to piano frequencies

Pipeline:
  1. Tempo detection (librosa beat tracker)
  2. Chroma CQT at sr=22050, bins_per_octave=36 (3× over-sampled for accuracy)
  3. Per-measure chroma aggregation (mean over frames in each measure window)
  4. Chord template matching on each measure's chroma vector
  5. Section detection via kernel-based novelty curve on chroma measures
  6. Period voting within each section (2, 4, 8 measure periods) for stability
  7. Measure-level segment metadata (JS divergence star kept for score display)

The chord_map is serialised into segments.json so all downstream steps load
it directly instead of re-inferring chords from imperfect note data.
"""
from __future__ import annotations

import numpy as np
import librosa
from pathlib import Path
from typing import List, Dict, Any, Tuple

from app.services.audio_utils import js_divergence, compute_mel_from_array

# ── chord templates (interval sets from root) ─────────────────────────────────
_CHORD_TEMPLATES = [
    [0, 4, 7],          # major
    [0, 3, 7],          # minor
    [0, 4, 7, 10],      # dom7
    [0, 3, 7, 10],      # min7
    [0, 4, 7, 11],      # maj7
    [0, 3, 6],          # dim
    [0, 5, 7],          # sus4
    [0, 2, 7],          # sus2
    [0, 3, 6, 9],       # dim7
    [0, 4, 8],          # aug
]

_NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


# ── chroma extraction ─────────────────────────────────────────────────────────

def _chroma_from_audio(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Compute CQT chroma with 3× over-sampling (bins_per_octave=36).
    Returns (12, T) normalised chroma matrix.
    """
    chroma = librosa.feature.chroma_cqt(
        y=y, sr=sr,
        bins_per_octave=36,
        norm=2,           # L2 normalise each frame
        hop_length=512,
    )
    return chroma.astype(np.float32)


def _chroma_per_measure(
    chroma: np.ndarray,
    sr: int,
    tempo: float,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Aggregate (12, T) chroma into (n_measures, 12) by mean-pooling each
    4-beat window.  Normalises each row so measures are comparable.
    """
    measure_dur  = (60.0 / tempo) * 4.0
    frames_per_m = max(1, int(round(measure_dur * sr / hop_length)))
    T            = chroma.shape[1]
    n_measures   = max(1, int(np.ceil(T / frames_per_m)))

    out = np.zeros((n_measures, 12), dtype=np.float32)
    for m in range(n_measures):
        f0 = m * frames_per_m
        f1 = min(f0 + frames_per_m, T)
        if f1 > f0:
            out[m] = chroma[:, f0:f1].mean(axis=1)

    # L1 normalise rows
    row_sums = out.sum(axis=1, keepdims=True)
    out /= np.where(row_sums > 0, row_sums, 1.0)
    return out


# ── chord fitting ─────────────────────────────────────────────────────────────

def _fit_chord(chroma_vec: np.ndarray) -> Tuple[int, set]:
    """
    Fit chord templates to a 12-dim chroma vector.
    Returns (root_pc, chord_pcs_set).
    """
    total      = float(chroma_vec.sum()) + 1e-8
    best_score = -1.0
    best_root, best_pcs = 0, {0, 4, 7}

    for intervals in _CHORD_TEMPLATES:
        for root in range(12):
            hits  = sum(float(chroma_vec[(root + i) % 12]) for i in intervals)
            score = hits / total
            if score > best_score:
                best_score = score
                best_root  = root
                best_pcs   = {(root + i) % 12 for i in intervals}

    return best_root, best_pcs


# ── section detection ─────────────────────────────────────────────────────────

def _detect_sections(
    chroma_measures: np.ndarray,
    min_section: int = 4,
) -> List[Tuple[int, int]]:
    """
    Kernel-based novelty curve on per-measure chroma → section boundaries.
    Returns list of (start_measure, end_measure) pairs.
    """
    n = len(chroma_measures)
    if n <= min_section * 2:
        return [(0, n)]

    k = min(4, n // 4)
    novelty = np.zeros(n, dtype=np.float32)
    for i in range(k, n - k):
        before = chroma_measures[max(0, i - k):i].mean(axis=0)
        after  = chroma_measures[i:min(n, i + k)].mean(axis=0)
        nb, na = np.linalg.norm(before), np.linalg.norm(after)
        if nb > 0 and na > 0:
            novelty[i] = 1.0 - float(np.dot(before, after) / (nb * na))

    threshold  = max(0.12, float(np.percentile(novelty[k:n - k], 65)))
    boundaries = [0]
    i = k
    while i < n - k:
        window  = novelty[max(0, i - 2):min(n, i + 3)]
        is_peak = novelty[i] >= threshold and novelty[i] == window.max()
        if is_peak and (i - boundaries[-1]) >= min_section:
            boundaries.append(i)
            i += min_section
        else:
            i += 1
    boundaries.append(n)

    return [(boundaries[j], boundaries[j + 1]) for j in range(len(boundaries) - 1)]


# ── period voting within a section ───────────────────────────────────────────

def _vote_section_chords(
    sec_chroma: np.ndarray,
    sec_start: int,
) -> List[Tuple[int, set]]:
    """
    Try repeating periods 2, 4, 8 within a section, pick the one with best
    average cosine similarity across repetitions, then fit chords to the
    averaged (voted) chroma per position.
    """
    M = len(sec_chroma)
    if M == 0:
        return []
    if M == 1:
        r, p = _fit_chord(sec_chroma[0])
        return [(r, p)]

    best_score, best_period = -1.0, 1
    best_voted = [sec_chroma.mean(axis=0)]

    for period in [2, 4, 8]:
        if period > M:
            continue
        n_full  = (M // period) * period
        stacked = sec_chroma[:n_full].reshape(-1, period, 12)
        voted   = stacked.mean(axis=0)   # (period, 12)

        sims = []
        for r in range(stacked.shape[0]):
            for p in range(period):
                a, b = stacked[r, p], voted[p]
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                if na > 0 and nb > 0:
                    sims.append(float(np.dot(a, b) / (na * nb)))

        score = float(np.mean(sims)) if sims else 0.0
        score += 0.005 * float(np.log2(max(period, 1)))

        if score > best_score:
            best_score, best_period = score, period
            best_voted = [voted[p] for p in range(period)]

    base = [_fit_chord(cv) for cv in best_voted]
    return [base[i % best_period] for i in range(M)]


# ── full chord map from audio ─────────────────────────────────────────────────

def build_chord_map_from_audio(
    y: np.ndarray,
    sr: int,
    tempo: float,
) -> Dict[int, Tuple[int, set]]:
    """
    Extract a per-measure chord map directly from raw audio chroma.
    Returns {measure_idx: (root_pc, chord_pcs_set)}.
    """
    chroma         = _chroma_from_audio(y, sr)
    chroma_measures = _chroma_per_measure(chroma, sr, tempo)
    sections       = _detect_sections(chroma_measures)

    chord_map: Dict[int, Tuple[int, set]] = {}
    for sec_start, sec_end in sections:
        sec_chroma  = chroma_measures[sec_start:sec_end]
        progression = _vote_section_chords(sec_chroma, sec_start)
        for offset, chord in enumerate(progression):
            chord_map[sec_start + offset] = chord

    return chord_map


def _chord_map_to_json(chord_map: Dict[int, Tuple[int, set]]) -> Dict[str, Any]:
    """Serialise chord map to JSON-safe dict."""
    return {str(k): [v[0], sorted(v[1])] for k, v in chord_map.items()}


def _chord_map_from_json(data: Dict[str, Any]) -> Dict[int, Tuple[int, set]]:
    """Deserialise chord map from JSON."""
    return {int(k): (v[0], set(v[1])) for k, v in data.items()}


# ── measure segmentation ──────────────────────────────────────────────────────

def _segment_measures(
    y_full: np.ndarray,
    y_bass: np.ndarray,
    y_upper: np.ndarray,
    sr: int,
    tempo: float,
    time_sig_num: int = 4,
) -> List[Dict[str, Any]]:
    """
    Split audio into measure-length segments.
    Compute bass vs melody JS divergence for each measure (used by score display).
    """
    measure_dur  = (60.0 / tempo) * time_sig_num
    total_dur    = librosa.get_duration(y=y_full, sr=sr)
    segments: List[Dict] = []
    t = 0.0
    m = 0

    while t < total_dur:
        t_end = min(t + measure_dur, total_dur)
        s0, s1 = int(t * sr), int(t_end * sr)
        seg_full = y_full[s0:s1]

        if len(seg_full) < 512:
            t = t_end
            continue

        mel_full = compute_mel_from_array(seg_full, sr)
        js_bass  = 1.0
        js_mel   = 1.0

        seg_bass  = y_bass[s0:min(s1, len(y_bass))]
        seg_upper = y_upper[s0:min(s1, len(y_upper))]

        if len(seg_bass) >= 512:
            js_bass = js_divergence(mel_full, compute_mel_from_array(seg_bass, sr))
        if len(seg_upper) >= 512:
            js_mel  = js_divergence(mel_full, compute_mel_from_array(seg_upper, sr))

        js_bass = float(np.nan_to_num(js_bass, nan=1.0, posinf=1.0, neginf=0.0))
        js_mel  = float(np.nan_to_num(js_mel,  nan=1.0, posinf=1.0, neginf=0.0))
        segments.append({
            'measure_index':  m,
            'start_time':     round(t, 3),
            'end_time':       round(t_end, 3),
            'duration':       round(t_end - t, 3),
            'time_signature': f'{time_sig_num}/4',
            'star':           'bass' if js_bass <= js_mel else 'melody',
            'js_scores':      {'bass': round(js_bass, 4), 'melody': round(js_mel, 4)},
        })

        t = t_end
        m += 1

    return segments


# ── main entry point ──────────────────────────────────────────────────────────

def analyse_audio(wav_path: Path, session_dir: Path) -> Dict[str, Any]:
    """
    Full segmentation + chord-map pipeline.

    Reads:
      wav_path                        — raw audio
      session_dir/components/bass.wav  — reconstructed bass group
      session_dir/components/upper.wav — reconstructed upper group

    Returns and writes segments.json containing:
      tempo_bpm, time_signature, chord_map, segment_count, segments
    """
    SR_SEG = 22050
    y,    sr = librosa.load(str(wav_path), sr=SR_SEG, mono=True)

    comp_dir   = session_dir / 'components'
    bass_path  = comp_dir / 'bass.wav'
    upper_path = comp_dir / 'upper.wav'

    y_bass  = librosa.load(str(bass_path),  sr=SR_SEG, mono=True)[0] if bass_path.exists()  else np.zeros(1)
    y_upper = librosa.load(str(upper_path), sr=SR_SEG, mono=True)[0] if upper_path.exists() else np.zeros(1)

    # Tempo
    tempo_arr, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo_arr)[0])
    if tempo <= 0:
        tempo = 120.0

    # Chord map from audio chroma
    chord_map = build_chord_map_from_audio(y, sr, tempo)

    # Measure segments
    segments = _segment_measures(y, y_bass, y_upper, sr, tempo)

    result: Dict[str, Any] = {
        'session_id':    wav_path.parent.name,
        'tempo_bpm':     round(tempo, 1),
        'time_signature': '4/4',
        'chord_map':     _chord_map_to_json(chord_map),
        'segment_count': len(segments),
        'segments':      segments,
    }

    with open(session_dir / 'segments.json', 'w') as f:
        import json
        json.dump(result, f)

    return result
