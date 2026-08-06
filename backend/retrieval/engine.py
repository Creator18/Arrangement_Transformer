"""
engine.py — the generating half of retrieval-augmented arrangement.

The corpus side (corpus_build / featurizer / pattern_taxonomy) answers
"what accompaniment patterns exist in real scores?".  This module answers
"which one belongs here, and what notes does it become?".

Pipeline for one song
---------------------
    1. split      — melody vs accompaniment, from the layer labels the
                    transcriber already assigns
    2. segment    — variable-length spans of homogeneous texture, cut at
                    downbeats where harmony or density shifts.  NOT a fixed
                    4-bar grid: accompaniment figures live at their own
                    cycle length and phrases are not uniformly long
    3. query      — build the SAME feature vector the corpus uses, from our
                    own (noisy) notes
    4. retrieve   — hard-filter by meter + cycle length, then cosine top-k
    5. select     — re-rank the k by musical fit, not just vector distance:
                    harmonic agreement, density match, melody clearance,
                    mud avoidance, and continuity with the previous span
    6. instantiate— re-pitch the chosen pattern's chord-degree template onto
                    THIS span's actual chords, at THIS song's actual beat
                    times, clamped to the target keyboard

What retrieval does and does not replace
----------------------------------------
It borrows FIGURATION, not harmony.  The source's chord progression, its
bass roots and its melody all survive; what changes is how the
accompaniment is played.  Otherwise it stops being an arrangement of the
user's song and becomes a cover version of whatever the corpus contained.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from retrieval.featurizer import featurize_unit, embed_query, top_k
from retrieval.pattern_taxonomy import (
    MIN_MELODY_CLEARANCE_ST, mud_penalty, melody_clearance,
)


CORPUS_DIR = Path(__file__).resolve().parent / "corpus"

MELODY_LAYERS = ("melody",)
ACCOMP_LAYERS = ("pattern", "bass_pattern", "bass_melody", "bass")

# Selection cost weights.  Similarity is the base; the rest are musical
# corrections on top of it.
W_SIM         = 1.00     # vector similarity (higher is better)
W_DENSITY     = 0.45     # penalty for density mismatch vs the source span
W_CLEARANCE   = 0.60     # penalty for crowding the melody
W_MUD         = 0.50     # penalty for muddy low voicings
W_CONTINUITY  = 0.35     # bonus for keeping the previous span's pattern type


# ── 1. split ──────────────────────────────────────────────────────────────────

def split_melody_accompaniment(
    notes: Sequence[Dict],
    use_skyline: bool = True,
    window: float = 0.09,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Partition into (melody, accompaniment).

    Default is the SKYLINE rule — at each moment the highest sounding note
    is the melody, everything below it is accompaniment — not the
    transcriber's layer labels.

    Why not the labels: in of_fullmix mode 'melody' is the catch-all for
    any upper note the phase-7 classifier does not confidently tag, so it
    absorbs most of the accompaniment.  On the first real test piece that
    was 567 'melody' notes against 111 'accompaniment' — for a two-minute
    piece whose actual melody is a few hundred notes at most.  Retrieval
    is unusually sensitive to this: if source accompaniment hides inside
    the melody layer we keep it AND add a retrieved pattern on top, so the
    arrangement ends up with two accompaniments.

    Skyline is crude — it hands ornaments and inner voices that poke above
    the tune to the melody — but it fails in a bounded, predictable way,
    whereas a mislabelled catch-all fails silently and totally.
    """
    if not use_skyline:
        return ([n for n in notes if n.get("layer") in MELODY_LAYERS],
                [n for n in notes if n.get("layer") in ACCOMP_LAYERS])

    ns = sorted(notes, key=lambda n: (float(n["onset"]), -int(n["pitch"])))
    mel: List[Dict] = []
    acc: List[Dict] = []
    i = 0
    while i < len(ns):
        j = i
        while (j + 1 < len(ns)
               and float(ns[j + 1]["onset"]) - float(ns[i]["onset"]) < window):
            j += 1
        group = ns[i:j + 1]
        top = max(group, key=lambda n: int(n["pitch"]))
        for n in group:
            (mel if n is top else acc).append(n)
        i = j + 1
    return mel, acc


# ── 2. segment into homogeneous spans ────────────────────────────────────────

def _bar_times(downbeat_times: Sequence[float], total_dur: float) -> List[float]:
    bt = [float(t) for t in downbeat_times]
    if not bt:
        return []
    if bt[-1] < total_dur:
        bt.append(total_dur)
    return bt


def segment_spans(
    accomp:          Sequence[Dict],
    downbeat_times:  Sequence[float],
    chord_map:       Dict[int, Tuple[int, set]],
    beats_per_bar:   int,
    measure_dur:     float,
    min_bars:        int = 2,
    max_bars:        int = 8,
) -> List[Dict]:
    """
    Cut the piece into spans of homogeneous accompaniment texture.

    A boundary is placed at a downbeat when either the accompaniment
    density or the harmonic rhythm changes materially from the running
    average, and spans are bounded to [min_bars, max_bars] so one
    unchanging texture does not become a single 60-bar span (which would
    retrieve one pattern for the whole piece) nor fragment into single
    bars (which invites the ransom-note effect).
    """
    bars = _bar_times(downbeat_times, max((float(n["offset"]) for n in accomp),
                                          default=0.0))
    if len(bars) < 2:
        return []

    # Per-bar density
    dens: List[float] = []
    for i in range(len(bars) - 1):
        a, b = bars[i], bars[i + 1]
        k = sum(1 for n in accomp if a <= float(n["onset"]) < b)
        dens.append(k / max(1e-6, (b - a)))

    spans: List[Dict] = []
    start = 0
    for i in range(1, len(dens) + 1):
        run = i - start
        at_end = (i == len(dens))
        changed = False
        if not at_end and run >= min_bars:
            prev = np.mean(dens[start:i]) if i > start else 0.0
            cur  = dens[i]
            # 60% swing in note density reads as a texture change
            changed = abs(cur - prev) > 0.6 * max(prev, 1e-6)
        if at_end or run >= max_bars or changed:
            if run >= 1:
                spans.append({
                    "bar_start": start,
                    "bar_end":   i,
                    "t_start":   bars[start],
                    "t_end":     bars[min(i, len(bars) - 1)],
                })
            start = i
    return spans


# ── 3. query ──────────────────────────────────────────────────────────────────

def _chords_for_span(chord_map: Dict[int, Tuple[int, set]],
                     span: Dict, measure_dur: float,
                     beats_per_bar: int) -> List[Dict]:
    """Chord context for a span, expressed in beats from the span's start."""
    out: List[Dict] = []
    for bar in range(span["bar_start"], span["bar_end"]):
        info = (chord_map.get(bar) or chord_map.get(bar - 1)
                or chord_map.get(bar + 1))
        if info is None:
            continue
        root, pcs = info
        out.append({
            "beat":    (bar - span["bar_start"]) * beats_per_bar,
            "root_pc": int(root),
            "pcs":     sorted(int(p) for p in pcs),
        })
    return out


def build_query_unit(
    accomp:        Sequence[Dict],
    span:          Dict,
    beat_times:    Sequence[float],
    chord_map:     Dict[int, Tuple[int, set]],
    beats_per_bar: int,
    meter:         str,
    measure_dur:   float,
) -> Optional[Dict]:
    """
    Convert one span of OUR notes into the same unit dict shape the corpus
    uses, so the identical featurizer can embed it.  Times are converted
    from seconds to beats relative to the span start; the span is folded
    down to a single cycle so the query is comparable to corpus cycles.
    """
    a, b = span["t_start"], span["t_end"]
    notes = [n for n in accomp if a <= float(n["onset"]) < b]
    if len(notes) < 2:
        return None

    bt = np.asarray(beat_times, dtype=float)
    if bt.size < 2:
        return None

    def to_beats(t: float) -> float:
        i = int(np.searchsorted(bt, t, side="right")) - 1
        i = min(max(i, 0), bt.size - 2)
        period = bt[i + 1] - bt[i]
        if period <= 1e-6:
            return float(i)
        return float(i + (t - bt[i]) / period)

    base = to_beats(a)
    span_bars = max(1, span["bar_end"] - span["bar_start"])

    # Cycle length: does the span's first bar repeat?  Reuse the corpus'
    # convention of half / one / two bar cycles.
    cycle_bars = 1.0 if span_bars < 2 else 1.0     # refined below
    rel = [(to_beats(float(n["onset"])) - base, n) for n in notes]
    if span_bars >= 2:
        h1 = [r % beats_per_bar for r, _ in rel if r < beats_per_bar]
        h2 = [r % beats_per_bar for r, _ in rel
              if beats_per_bar <= r < 2 * beats_per_bar]
        if h1 and h2:
            p1, _ = np.histogram(h1, bins=8, range=(0, beats_per_bar))
            p2, _ = np.histogram(h2, bins=8, range=(0, beats_per_bar))
            n1, n2 = np.linalg.norm(p1), np.linalg.norm(p2)
            if n1 > 0 and n2 > 0 and float(p1 @ p2 / (n1 * n2)) < 0.7:
                cycle_bars = 2.0

    cycle_beats = beats_per_bar * cycle_bars
    q_notes = [
        {"beat": float(r % cycle_beats),
         "dur":  max(0.05, to_beats(float(n["offset"])) - to_beats(float(n["onset"]))),
         "pitch": int(n["pitch"]),
         "vel":   int(n.get("velocity", 80))}
        for r, n in rel
    ]
    return {
        "id":            f"query:bar{span['bar_start']}",
        "meter":         meter,
        "beats_per_bar": beats_per_bar,
        "cycle_bars":    cycle_bars,
        "notes":         q_notes,
        "chords":        _chords_for_span(chord_map, span, measure_dur, beats_per_bar),
    }


# ── corpus loading ────────────────────────────────────────────────────────────

class Corpus:
    """Lazily-loaded pattern corpus + its normalization stats."""

    def __init__(self, corpus_dir: Path = CORPUS_DIR):
        self.units: List[Dict] = [
            json.loads(l) for l in open(corpus_dir / "units.jsonl", encoding="utf-8")
        ]
        self.matrix = np.load(corpus_dir / "vectors.npy")
        self.stats  = json.load(open(corpus_dir / "norm.json", encoding="utf-8"))
        self._meter = np.array([u["meter"] for u in self.units])
        self._cycle = np.array([float(u["cycle_bars"]) for u in self.units])

    def mask_for(self, meter: str, cycle_bars: float) -> np.ndarray:
        """Hard compatibility filter — meter and cycle length must match."""
        return (self._meter == meter) & (np.isclose(self._cycle, cycle_bars))


# ── 4 + 5. retrieve and select ───────────────────────────────────────────────

def select_pattern(
    corpus:        Corpus,
    query:         Dict,
    melody_low:    Optional[int],
    prev_type:     Optional[str],
    k:             int = 8,
) -> Optional[Tuple[Dict, float, Dict]]:
    """
    Retrieve top-k by vector similarity, then re-rank by musical fit.
    Returns (unit, score, diagnostics) for the winner, or None.
    """
    mask = corpus.mask_for(query["meter"], query["cycle_bars"])
    if not mask.any():
        return None
    qv = embed_query(featurize_unit(query), corpus.stats)
    cands = top_k(qv, corpus.matrix, k=k, mask=mask)
    if not cands:
        return None

    q_beats = query["beats_per_bar"] * query["cycle_bars"]
    q_density = len(query["notes"]) / max(1e-6, q_beats)

    best = None
    for idx, sim in cands:
        u = corpus.units[idx]
        u_beats = u["beats_per_bar"] * u["cycle_bars"]
        u_density = len(u["notes"]) / max(1e-6, u_beats)

        density_pen = abs(u_density - q_density) / max(1.0, q_density)
        mud_pen     = mud_penalty(u["notes"])
        if melody_low is not None:
            clear = melody_clearance(u["notes"], melody_low)
            clear_pen = max(0.0, (MIN_MELODY_CLEARANCE_ST - clear)) / 12.0
        else:
            clear_pen = 0.0
        cont_bonus = 1.0 if (prev_type and u.get("pattern_type") == prev_type) else 0.0

        score = (W_SIM * sim
                 - W_DENSITY   * min(density_pen, 2.0)
                 - W_CLEARANCE * min(clear_pen, 2.0)
                 - W_MUD       * mud_pen
                 + W_CONTINUITY * cont_bonus)
        diag = {"sim": round(float(sim), 4),
                "density_pen": round(float(density_pen), 3),
                "clear_pen": round(float(clear_pen), 3),
                "mud_pen": round(float(mud_pen), 3),
                "continuity": cont_bonus,
                "score": round(float(score), 4)}
        if best is None or score > best[1]:
            best = (u, float(score), diag)
    return best


# ── 6. instantiate ────────────────────────────────────────────────────────────

_DEGREE_ORDER = (0, 2, 1, 3)     # root, fifth, third, seventh — voicing priority


def _degree_of(pitch: int, chord: Optional[Dict]) -> int:
    """Which chord member a corpus pitch represents (0=root,1=3rd,2=5th,3=7th,4=other)."""
    if not chord:
        return 4
    iv = (pitch - int(chord["root_pc"])) % 12
    return {0: 0, 3: 1, 4: 1, 7: 2, 10: 3, 11: 3}.get(iv, 4)


def _pitch_for_degree(degree: int, chord: Dict, near: int,
                      lo: int, hi: int) -> int:
    """
    Realise a chord degree as an actual pitch close to `near`, inside
    [lo, hi].  Falls back to the chord root when the degree is absent.
    """
    root = int(chord["root_pc"])
    want = {0: 0, 1: None, 2: 7, 3: 10}.get(degree)
    pcs = [int(p) % 12 for p in chord.get("pcs", [])] or [root]
    if degree == 1:                                  # third: major or minor
        cand = [p for p in pcs if (p - root) % 12 in (3, 4)]
        target_pc = cand[0] if cand else root
    elif want is None:
        target_pc = root
    else:
        cand = [p for p in pcs if (p - root) % 12 == want]
        target_pc = cand[0] if cand else (root + want) % 12
    # nearest octave placement to `near`
    best, bestd = None, 1e9
    for octave in range(0, 11):
        p = target_pc + 12 * octave
        if p < lo or p > hi:
            continue
        d = abs(p - near)
        if d < bestd:
            best, bestd = p, d
    return best if best is not None else max(lo, min(hi, near))


def instantiate(
    unit:          Dict,
    span:          Dict,
    beat_times:    Sequence[float],
    chord_map:     Dict[int, Tuple[int, set]],
    beats_per_bar: int,
    register_lo:   int = 28,
    register_hi:   int = 60,
) -> List[Dict]:
    """
    Turn a retrieved pattern into real notes over THIS span: tile its cycle
    across the span's bars, re-pitch each note by chord degree onto the
    span's actual chords, and place onsets at the song's actual beat times
    so rubato and tempo drift are inherited rather than flattened.
    """
    bt = np.asarray(beat_times, dtype=float)
    if bt.size < 2:
        return []
    cycle_bars  = float(unit["cycle_bars"])
    cycle_beats = beats_per_bar * cycle_bars
    src_chords  = sorted(unit.get("chords", []), key=lambda c: float(c["beat"]))

    def beat_to_time(global_beat: float) -> float:
        i = int(np.floor(global_beat))
        frac = global_beat - i
        i = min(max(i, 0), bt.size - 2)
        return float(bt[i] + frac * (bt[i + 1] - bt[i]))

    out: List[Dict] = []
    n_bars = span["bar_end"] - span["bar_start"]
    reps = int(np.ceil(n_bars / max(cycle_bars, 0.5)))
    for r in range(reps):
        bar_off = r * cycle_bars
        if bar_off >= n_bars:
            break
        abs_bar = span["bar_start"] + bar_off
        chord = (chord_map.get(int(abs_bar)) or chord_map.get(int(abs_bar) - 1))
        if chord is None:
            continue
        tgt = {"root_pc": int(chord[0]), "pcs": sorted(int(p) for p in chord[1])}

        for n in unit["notes"]:
            nb = float(n["beat"])
            if nb >= cycle_beats:
                continue
            src_chord = None
            for c in src_chords:
                if float(c["beat"]) <= nb + 1e-6:
                    src_chord = c
            deg = _degree_of(int(n["pitch"]), src_chord)
            # keep the pattern's internal contour: place relative to where the
            # source note sat within its own chord's register
            near = register_lo + (int(n["pitch"]) - (src_chord["root_pc"] if src_chord else 0)) % 24
            pitch = _pitch_for_degree(deg, tgt, near, register_lo, register_hi)

            gb = (span["bar_start"] + bar_off) * beats_per_bar + nb
            on = beat_to_time(gb)
            off = beat_to_time(gb + max(0.1, float(n.get("dur", 0.5))))
            if off <= on:
                off = on + 0.08
            out.append({
                "pitch":    int(pitch),
                "onset":    round(on, 4),
                "offset":   round(off, 4),
                "velocity": int(n.get("vel", 70)),
                "layer":    "pattern",
                "source":   unit["id"],
                "pattern_type": unit.get("pattern_type", "mixed"),
            })
    return out
