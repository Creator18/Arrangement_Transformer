"""
Bar-level clustering of a pattern stem into Pattern 1 … N.

After REPET-SIM has produced `pattern.wav` (or `bass_pattern.wav`), this
module chunks that stem at downbeat boundaries, features each bar, and
runs agglomerative clustering to group bars into distinct recurring
figures.  The result lets the UI present each pattern variant as its own
piano-roll row for review.

Features per bar
----------------
- Chroma (12 dims, summed over the bar): captures harmonic identity.
- Onset envelope, downsampled to 8 dims: captures rhythmic shape.
Total = 20-dim L2-normalised descriptor per bar.

Clustering
----------
Agglomerative with a cosine-distance threshold rather than a fixed K — a
piece with two figures gets N=2, a piece with five gets N=5.  Capped at
MAX_CLUSTERS to keep the UI sane.

Fallback
--------
- If there are fewer than 2 bars: single cluster (idx=0).
- If the bar-level clustering returns N=1, automatically retries at
  half-bar granularity (catches Alberti-style figures shorter than a bar).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import librosa


MAX_CLUSTERS    = 6
DISTANCE_THRESH = 0.25      # cosine distance; lower → more clusters


def _bar_windows(
    downbeat_times: np.ndarray,
    total_dur:      float,
    half_bar:       bool = False,
) -> List[Tuple[float, float]]:
    """Return list of (start_s, end_s) windows from downbeat times."""
    if downbeat_times is None or len(downbeat_times) < 2:
        return []
    times = list(downbeat_times)
    if times[-1] < total_dur:
        times.append(total_dur)
    windows: List[Tuple[float, float]] = []
    for i in range(len(times) - 1):
        a, b = float(times[i]), float(times[i + 1])
        if b - a < 0.2:
            continue
        if half_bar:
            mid = 0.5 * (a + b)
            windows.append((a, mid))
            windows.append((mid, b))
        else:
            windows.append((a, b))
    return windows


def _featurise_bar(
    y:        np.ndarray,
    sr:       int,
    onset_env: np.ndarray,
    times:    np.ndarray,
    a:        float,
    b:        float,
) -> np.ndarray:
    """Compute a 20-dim descriptor for the bar [a, b)."""
    i0 = int(a * sr)
    i1 = max(i0 + 1, int(b * sr))
    seg = y[i0:i1]
    if seg.size < 256:
        return np.zeros(20, dtype=np.float32)

    chroma = librosa.feature.chroma_stft(y=seg, sr=sr, n_fft=2048, hop_length=512)
    chroma_v = chroma.mean(axis=1)                          # (12,)

    mask = (times >= a) & (times < b)
    env_slice = onset_env[mask]
    if env_slice.size == 0:
        env_v = np.zeros(8, dtype=np.float32)
    else:
        # Resample to 8 dims by binning
        idx = np.linspace(0, env_slice.size, 9).astype(int)
        env_v = np.array(
            [env_slice[idx[k]:idx[k + 1]].mean() if idx[k + 1] > idx[k] else 0.0
             for k in range(8)],
            dtype=np.float32,
        )
        m = env_v.max()
        if m > 1e-6:
            env_v = env_v / m

    feat = np.concatenate([chroma_v, env_v]).astype(np.float32)
    n = float(np.linalg.norm(feat))
    return feat / n if n > 1e-6 else feat


def _cluster(features: np.ndarray) -> np.ndarray:
    """
    Agglomerative cosine clustering with a distance threshold.  Returns an
    array of cluster indices (0-based, ordered by descending bar count so
    cluster 0 is always the most frequent figure).
    """
    if features.shape[0] == 0:
        return np.zeros(0, dtype=int)
    if features.shape[0] == 1:
        return np.zeros(1, dtype=int)

    try:
        from sklearn.cluster import AgglomerativeClustering
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=DISTANCE_THRESH,
            metric="cosine",
            linkage="average",
        )
        labels = clusterer.fit_predict(features)
    except Exception:
        # Fallback: greedy nearest-cluster with cosine distance
        labels = np.full(features.shape[0], -1, dtype=int)
        centroids: List[np.ndarray] = []
        for i, f in enumerate(features):
            if not centroids:
                centroids.append(f); labels[i] = 0; continue
            dists = [1.0 - float(np.dot(f, c)) for c in centroids]
            j = int(np.argmin(dists))
            if dists[j] <= DISTANCE_THRESH:
                labels[i] = j
            else:
                labels[i] = len(centroids)
                centroids.append(f)

    # Cap clusters at MAX_CLUSTERS: merge least-frequent extras into the
    # nearest remaining centroid.
    unique, counts = np.unique(labels, return_counts=True)
    if unique.size > MAX_CLUSTERS:
        keep = unique[np.argsort(-counts)[:MAX_CLUSTERS]]
        keep_set = set(keep.tolist())
        # Compute centroids for kept clusters
        kept_centroids = {k: features[labels == k].mean(axis=0) for k in keep}
        for i, lab in enumerate(labels):
            if lab not in keep_set:
                dists = [1.0 - float(np.dot(features[i], c)) for c in kept_centroids.values()]
                labels[i] = list(kept_centroids.keys())[int(np.argmin(dists))]

    # Renumber by descending count → cluster 0 is the most frequent
    unique, counts = np.unique(labels, return_counts=True)
    order = unique[np.argsort(-counts)]
    remap = {old: new for new, old in enumerate(order)}
    return np.array([remap[l] for l in labels], dtype=int)


def cluster_pattern_stem(
    y:              np.ndarray,
    sr:             int,
    downbeat_times: np.ndarray,
) -> Dict:
    """
    Group bars of a pattern stem into recurring-figure clusters.

    Returns:
        {
          'n_clusters': int,
          'granularity': 'bar' | 'half-bar',
          'bars': [
            {'start': float, 'end': float, 'cluster': int},
            …
          ],
          'cluster_bar_counts': {0: int, 1: int, …},
        }
    """
    total_dur = float(len(y)) / float(sr)
    windows = _bar_windows(downbeat_times, total_dur, half_bar=False)
    granularity = "bar"

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=512)

    def _features_for(windows):
        return np.array(
            [_featurise_bar(y, sr, onset_env, times, a, b) for a, b in windows],
            dtype=np.float32,
        ) if windows else np.zeros((0, 20), dtype=np.float32)

    feats = _features_for(windows)
    labels = _cluster(feats)

    # If we got just one cluster (or no bars), try half-bar granularity
    if (labels.size and labels.max() == 0) or labels.size == 0:
        hb_windows = _bar_windows(downbeat_times, total_dur, half_bar=True)
        if hb_windows:
            hb_feats = _features_for(hb_windows)
            hb_labels = _cluster(hb_feats)
            if hb_labels.size and hb_labels.max() >= 1:
                windows, labels, granularity = hb_windows, hb_labels, "half-bar"

    bars = [
        {"start": float(a), "end": float(b), "cluster": int(c)}
        for (a, b), c in zip(windows, labels)
    ]
    counts: Dict[int, int] = {}
    for c in labels.tolist():
        counts[int(c)] = counts.get(int(c), 0) + 1

    return {
        "n_clusters":         int(labels.max() + 1) if labels.size else 0,
        "granularity":        granularity,
        "bars":               bars,
        "cluster_bar_counts": counts,
    }


def write_cluster_audio(
    y:        np.ndarray,
    sr:       int,
    cluster_meta: Dict,
    out_dir:  Path,
    stem_name: str = "pattern",
) -> Dict[int, Path]:
    """
    Concatenate the audio slices belonging to each cluster and write one
    WAV per cluster: <stem_name>_cluster_<i>.wav.  Returns idx → path map.
    """
    import soundfile as sf
    out_dir.mkdir(parents=True, exist_ok=True)

    by_cluster: Dict[int, List[np.ndarray]] = {}
    for bar in cluster_meta.get("bars", []):
        i0 = int(float(bar["start"]) * sr)
        i1 = max(i0 + 1, int(float(bar["end"]) * sr))
        slice_ = y[i0:i1]
        by_cluster.setdefault(int(bar["cluster"]), []).append(slice_)

    paths: Dict[int, Path] = {}
    for cidx, slices in by_cluster.items():
        if not slices:
            continue
        concat = np.concatenate(slices).astype(np.float32)
        path = out_dir / f"{stem_name}_cluster_{cidx}.wav"
        sf.write(str(path), concat, sr, subtype="PCM_16")
        paths[cidx] = path
    return paths
