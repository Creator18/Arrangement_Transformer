"""
featurizer.py — the shared embedding for figuration-cycle retrieval.

ONE featurizer, used verbatim on both sides:
  - corpus side: units extracted from clean MusicXML scores
  - query side:  units extracted from our (noisy) transcription

That symmetry is load-bearing.  If the two sides ever compute features
differently, retrieval quietly degrades with no error message.  Never
fork this logic per side.

Unit dict format (produced by corpus_build.py and, later, the query
segmenter):
    {
      "id":         str,                # provenance, e.g. "asap:Chopin/...:bar12"
      "meter":      "4/4",              # hard filter key, not embedded
      "cycle_bars": 1.0,                # 0.5 | 1.0 | 2.0 — hard filter key
      "beats_per_bar": 4,
      "style":      "Chopin",           # optional filter tag
      "notes": [                        # LH only, beat-relative to cycle start
        {"beat": 0.0, "dur": 0.5, "pitch": 48, "vel": 70},
        ...
      ],
      "chords": [                       # per-beat chord context
        {"beat": 0, "root_pc": 0, "pcs": [0, 4, 7]},
        ...
      ],
    }

Design notes
------------
- Invariances live in the REPRESENTATION: pitches are folded to chord
  degrees (key-invariant), times are beat-relative (tempo-invariant).
- Every feature is a histogram / ratio / statistic — never an exact
  sequence — so one phantom note or ±40 ms of onset jitter moves the
  vector slightly instead of breaking it.  This is the noise-robustness
  property the corruption self-retrieval test (validate_embedding.py)
  checks.
- Block weights: rhythm > harmony > motion > texture.  Rhythm is the
  perceptually dominant "same kind of accompaniment" factor.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


# ── grid + block configuration ────────────────────────────────────────────────
RHYTHM_SLOTS = 16          # one bar at 16th-note resolution
N_DIMS_RHYTHM  = RHYTHM_SLOTS
N_DIMS_HARMONY = 9
N_DIMS_MOTION  = 6
N_DIMS_TEXTURE = 8
N_DIMS = N_DIMS_RHYTHM + N_DIMS_HARMONY + N_DIMS_MOTION + N_DIMS_TEXTURE   # 39

# Per-block weights applied AFTER z-scoring (see normalize_matrix)
BLOCK_WEIGHTS = np.concatenate([
    np.full(N_DIMS_RHYTHM,  1.6),
    np.full(N_DIMS_HARMONY, 1.2),
    np.full(N_DIMS_MOTION,  1.0),
    np.full(N_DIMS_TEXTURE, 0.8),
]).astype(np.float32)


def _chord_at(chords: List[Dict], beat: float) -> Optional[Dict]:
    """Latest chord whose beat <= the note's beat (chords sorted by beat)."""
    best = None
    for c in chords:
        if float(c["beat"]) <= beat + 1e-6:
            best = c
        else:
            break
    return best


def _degree_class(pitch: int, chord: Optional[Dict]) -> int:
    """
    Map a pitch onto its role in the local chord:
      0 = root, 1 = third, 2 = fifth, 3 = seventh, 4 = other-chord-tone,
      5 = non-chord tone.  Falls back to 5 when no chord context exists.
    """
    if not chord:
        return 5
    pc = pitch % 12
    root = int(chord["root_pc"]) % 12
    interval = (pc - root) % 12
    if interval == 0:
        return 0
    if interval in (3, 4):
        return 1
    if interval == 7:
        return 2
    if interval in (10, 11):
        return 3
    if pc in set(int(p) % 12 for p in chord.get("pcs", [])):
        return 4
    return 5


def featurize_unit(unit: Dict) -> np.ndarray:
    """
    Unit dict → raw (un-normalized) feature vector of length N_DIMS.
    Returns zeros for an empty unit (caller should filter those out).
    """
    notes  = unit.get("notes", [])
    chords = sorted(unit.get("chords", []), key=lambda c: float(c["beat"]))
    beats_per_bar = int(unit.get("beats_per_bar", 4))
    cycle_bars    = float(unit.get("cycle_bars", 1.0))
    cycle_beats   = beats_per_bar * cycle_bars

    v = np.zeros(N_DIMS, dtype=np.float32)
    if not notes or cycle_beats <= 0:
        return v

    onsets = np.array([float(n["beat"]) for n in notes], dtype=np.float64)
    pitches = np.array([int(n["pitch"]) for n in notes], dtype=np.int64)
    durs   = np.array([max(1e-3, float(n.get("dur", 0.25))) for n in notes])
    vels   = np.array([float(n.get("vel", 80)) for n in notes])

    # ── rhythm block: velocity-weighted onset histogram over 16 slots ────────
    # Cycle time is folded onto a single bar's grid so a 2-bar cycle's two
    # halves overlay — the histogram captures the *recurring* placement.
    #
    # SOFT binning: each onset's mass is split linearly between its two
    # nearest slots by fractional position.  With hard binning, ±40 ms of
    # transcription jitter flips boundary notes into the adjacent slot and
    # the sparse histogram jumps — measured as the dominant cause of
    # self-retrieval failure (recall@3 0.78 under jitter-only corruption).
    # Soft binning makes the histogram continuous under small onset shifts.
    frac_slot = (onsets % beats_per_bar) / beats_per_bar * RHYTHM_SLOTS
    rhythm = np.zeros(RHYTHM_SLOTS, dtype=np.float32)
    for fs, w in zip(frac_slot, vels):
        lo = int(np.floor(fs)) % RHYTHM_SLOTS
        hi = (lo + 1) % RHYTHM_SLOTS          # wraps: end of bar ≈ start
        t = fs - np.floor(fs)
        rhythm[lo] += w * (1.0 - t)
        rhythm[hi] += w * t
    # Max-normalize.  (Sum-normalization was tried and measured worse: a
    # 15% note-drop shrinks the sum and inflates every remaining bin, while
    # the max only moves when the single loudest note happens to be
    # dropped.)
    if rhythm.max() > 0:
        rhythm /= rhythm.max()
    v[:N_DIMS_RHYTHM] = rhythm
    o = N_DIMS_RHYTHM

    # ── harmony block ─────────────────────────────────────────────────────────
    # Degree-class histogram (6 dims) + bass-on-beat-1 root ratio +
    # bass-on-strong-beat fifth ratio + chord-change rate.
    deg_hist = np.zeros(6, dtype=np.float32)
    for n in notes:
        deg_hist[_degree_class(int(n["pitch"]), _chord_at(chords, float(n["beat"])))] += 1
    deg_hist /= max(1, len(notes))

    # Lowest note at each integer beat — is it the root? the fifth?
    root_on_1 = 0.0
    fifth_on_strong = 0.0
    n_downbeats = 0
    n_strong = 0
    for b in range(int(cycle_beats)):
        at_beat = [n for n in notes if abs(float(n["beat"]) - b) < 0.26]
        if not at_beat:
            continue
        lowest = min(at_beat, key=lambda n: int(n["pitch"]))
        d = _degree_class(int(lowest["pitch"]), _chord_at(chords, float(b)))
        if b % beats_per_bar == 0:
            n_downbeats += 1
            root_on_1 += 1.0 if d == 0 else 0.0
        else:
            n_strong += 1
            fifth_on_strong += 1.0 if d == 2 else 0.0
    chord_rate = len(chords) / max(1.0, cycle_beats)

    v[o:o+6] = deg_hist
    v[o+6]   = root_on_1 / max(1, n_downbeats)
    v[o+7]   = fifth_on_strong / max(1, n_strong)
    v[o+8]   = min(1.0, chord_rate)
    o += N_DIMS_HARMONY

    # ── motion block ──────────────────────────────────────────────────────────
    # Simultaneity ratio, mean |interval| between successive onsets,
    # direction-change rate, off-beat ratio, syncopation weight, legato ratio.
    order = np.argsort(onsets)
    sorted_on = onsets[order]
    sorted_pi = pitches[order]

    # Group by ~simultaneous onsets.  Window = 0.18 beats (~90 ms at 120 BPM):
    # wide enough that two chord notes jittered ±40 ms in opposite directions
    # still group together, narrow enough that 16th-note runs (0.25 beats
    # apart) stay separate.
    SIMUL_WINDOW_BEATS = 0.18
    groups: List[List[int]] = []
    for i in order:
        if groups and abs(onsets[i] - onsets[groups[-1][0]]) < SIMUL_WINDOW_BEATS:
            groups[-1].append(i)
        else:
            groups.append([i])
    simultaneity = len(notes) / max(1, len(groups))

    steps = np.abs(np.diff(sorted_pi)) if len(sorted_pi) > 1 else np.array([0.0])
    dirs  = np.sign(np.diff(sorted_pi)) if len(sorted_pi) > 1 else np.array([0.0])
    dir_changes = float(np.mean(np.abs(np.diff(dirs)) > 0)) if dirs.size > 1 else 0.0
    offbeat = float(np.mean((onsets % 1.0) > 0.26))
    sync_w  = float(np.mean(((onsets % 1.0) > 0.4) & ((onsets % 1.0) < 0.6)))
    legato  = float(np.clip(np.mean(durs) / (cycle_beats / max(1, len(groups))), 0, 2)) / 2

    v[o]   = min(1.0, (simultaneity - 1.0) / 4.0)
    v[o+1] = min(1.0, float(np.mean(steps)) / 12.0)
    v[o+2] = dir_changes
    v[o+3] = offbeat
    v[o+4] = sync_w
    v[o+5] = legato
    o += N_DIMS_MOTION

    # ── texture block ─────────────────────────────────────────────────────────
    density   = len(notes) / cycle_beats
    centroid  = (float(np.mean(pitches)) - 60.0) / 24.0       # rel. middle C
    span      = float(pitches.max() - pitches.min()) / 24.0
    max_simul = max(len(g) for g in groups)
    stretch   = max(
        (max(pitches[g_] for g_ in g) - min(pitches[g_] for g_ in g))
        for g in groups
    ) / 14.0 if groups else 0.0
    sustain   = float(np.clip(np.sum(durs) / cycle_beats, 0, 3)) / 3.0
    vel_mean  = float(np.mean(vels)) / 127.0
    vel_var   = float(np.std(vels)) / 40.0

    v[o]   = min(1.0, density / 8.0)
    v[o+1] = np.clip(centroid, -1.5, 1.5)
    v[o+2] = min(1.5, span)
    v[o+3] = min(1.0, (max_simul - 1) / 4.0)
    v[o+4] = min(1.0, stretch)
    v[o+5] = sustain
    v[o+6] = vel_mean
    v[o+7] = min(1.0, vel_var)

    return v


# ── corpus-level normalization + similarity ──────────────────────────────────

def normalize_matrix(raw: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Z-score each dim over the corpus, apply block weights, L2-normalize rows.
    Returns (matrix, stats) where stats carries mean/std needed to embed
    queries into the same space later.
    """
    mean = raw.mean(axis=0)
    std  = raw.std(axis=0)
    std[std < 1e-6] = 1.0
    z = (raw - mean) / std
    z = z * BLOCK_WEIGHTS
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    return (z / norms).astype(np.float32), {
        "mean": mean.tolist(), "std": std.tolist(),
        "block_weights": BLOCK_WEIGHTS.tolist(),
    }


def embed_query(raw_vec: np.ndarray, stats: Dict) -> np.ndarray:
    """Embed one raw query vector using the corpus' normalization stats."""
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std  = np.asarray(stats["std"],  dtype=np.float32)
    z = (raw_vec - mean) / std * np.asarray(stats["block_weights"], dtype=np.float32)
    n = np.linalg.norm(z)
    return (z / n).astype(np.float32) if n > 1e-9 else z.astype(np.float32)


def top_k(query: np.ndarray, matrix: np.ndarray, k: int = 5,
          mask: Optional[np.ndarray] = None) -> List[Tuple[int, float]]:
    """
    Cosine top-k (matrix rows are L2-normalized).  `mask` is a boolean
    array marking rows that survived the hard filters (meter, cycle
    length, playability); rows outside the mask never rank.
    """
    sims = matrix @ query
    if mask is not None:
        sims = np.where(mask, sims, -np.inf)
    idx = np.argpartition(-sims, min(k, len(sims) - 1))[:k]
    idx = idx[np.argsort(-sims[idx])]
    return [(int(i), float(sims[i])) for i in idx if np.isfinite(sims[i])]
