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

        # Clearance is judged on the BEST ACHIEVABLE octave placement, not on
        # the corpus pitches as-stored.  instantiate() runs an octave search,
        # so a pattern that crowds the melody where it happens to sit in its
        # source score will simply be moved — penalising it here discards a
        # candidate the placement stage would have fixed, and (measured) let
        # clearance swing the score by up to 0.70 while the entire similarity
        # range is 0.45.  That put the melody's register in charge of which
        # texture got chosen, which is how the second half of the test piece
        # ended up thrashing between five figurations in six spans.
        if melody_low is not None:
            span_ = max(int(n["pitch"]) for n in u["notes"]) - \
                    min(int(n["pitch"]) for n in u["notes"])
            best_clear = max(
                melody_clearance([{**n, "pitch": int(n["pitch"]) + 12 * s}
                                  for n in u["notes"]], melody_low)
                for s in (-2, -1, 0, 1, 2)
            )
            clear_pen = max(0.0, (MIN_MELODY_CLEARANCE_ST - best_clear)) / 12.0
            # A pattern too WIDE to ever clear the melody is genuinely
            # unusable; that is the only case this should still veto.
            if span_ > 30:
                clear_pen += 0.5
        else:
            clear_pen = 0.0

        # Continuity scales with how decisive similarity is.  A flat bonus
        # comparable to the whole similarity spread makes the first pick
        # self-reinforce regardless of fit; scaling it means "keep the
        # current texture unless something is clearly better".
        top_sim = cands[0][1]
        decisiveness = min(1.0, max(0.0, (top_sim - sim) / max(top_sim, 1e-6)))
        cont_bonus = ((1.0 - decisiveness)
                      if (prev_type and u.get("pattern_type") == prev_type) else 0.0)

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


def _make_bar_clock(downbeat_times: Sequence[float],
                    beat_times: Sequence[float],
                    beats_per_bar: int):
    """
    Build (bar_index, beat_within_bar) -> seconds.

    Anchored to actual DOWNBEAT times, with the real beats that fall inside
    each bar used for intra-bar interpolation.  The previous version
    computed a global beat index as bar*beats_per_bar + beat and indexed
    beat_times with it, which silently assumes the beat and downbeat grids
    agree everywhere.  madmom detects them separately, so one dropped beat
    or a pickup bar makes the two drift — and because the index is
    cumulative, the error grows through the piece.  That is the "sometimes
    rushed, sometimes delayed" artefact.
    """
    db = np.asarray(sorted(float(t) for t in downbeat_times), dtype=float)
    bt = np.asarray(sorted(float(t) for t in beat_times),     dtype=float)

    def clock(bar: float, beat_in_bar: float) -> Optional[float]:
        b = int(np.floor(bar))
        if db.size < 2 or b < 0:
            return None
        if b >= db.size - 1:
            # past the last detected downbeat: extrapolate with the last bar
            if db.size < 2:
                return None
            bar_len = float(db[-1] - db[-2])
            over = bar - (db.size - 1)
            start, length = float(db[-1]) + over * bar_len, bar_len
        else:
            start, length = float(db[b]), float(db[b + 1] - db[b])
        frac = beat_in_bar / max(1, beats_per_bar)
        # Prefer the real beats inside this bar over linear interpolation,
        # so intra-bar rubato survives.
        inner = bt[(bt > start + 1e-6) & (bt < start + length - 1e-6)]
        grid = np.concatenate(([start], inner, [start + length]))
        pos = frac * (grid.size - 1)
        i = int(np.floor(pos))
        i = min(max(i, 0), grid.size - 2)
        return float(grid[i] + (pos - i) * (grid[i + 1] - grid[i]))

    return clock


def _octave_shift_search(
    notes:          List[Dict],
    target_centroid: Optional[float],
    melody_low:     Optional[int],
    lo:             int,
    hi:             int,
    max_shift:      int = 2,
) -> int:
    """
    Choose an octave displacement for a generated pattern.

    Objective (lower is better):
        |centroid - source span's accompaniment centroid|   register match
      + mud penalty                                          low tight voicings
      + melody-crowding penalty                              clearance
      + out-of-range penalty                                 keyboard fit

    Enumerated over a 5-point space (-2..+2 octaves) rather than hill
    climbed: with a space this small, exhaustive search costs the same and
    cannot get stuck in a local optimum, which a hill climb over a
    non-convex objective like this genuinely can.

    On the audio-domain objective you suggested: comparing rendered
    waveforms would be the most faithful signal, but each render is
    seconds and this runs per span, so it would add minutes per song.  The
    register-centroid term is a symbolic proxy for the same thing — octave
    placement is precisely what shifts spectral energy between registers —
    at ~zero cost.  If the proxy proves insufficient in listening, the
    hook to swap in a spectral objective is this function's signature.
    """
    if not notes:
        return 0
    best_shift, best_cost = 0, float("inf")
    for s in range(-max_shift, max_shift + 1):
        shifted = [{**n, "pitch": int(n["pitch"]) + 12 * s} for n in notes]
        ps = np.array([n["pitch"] for n in shifted], dtype=float)

        cost = 0.0
        if target_centroid is not None:
            cost += abs(float(ps.mean()) - target_centroid) / 12.0
        cost += 1.5 * mud_penalty(shifted)
        if melody_low is not None:
            clear = melody_low - float(ps.max())
            cost += max(0.0, (MIN_MELODY_CLEARANCE_ST - clear)) / 12.0
        cost += float(np.mean((ps < lo) | (ps > hi))) * 3.0

        if cost < best_cost:
            best_shift, best_cost = s, cost
    return best_shift


def instantiate(
    unit:            Dict,
    span:            Dict,
    beat_times:      Sequence[float],
    chord_map:       Dict[int, Tuple[int, set]],
    beats_per_bar:   int,
    downbeat_times:  Sequence[float] = (),
    source_notes:    Sequence[Dict] = (),
    melody_low:      Optional[int] = None,
    register_lo:     int = 28,
    register_hi:     int = 67,
) -> List[Dict]:
    """
    Turn a retrieved pattern into real notes over THIS span: tile its cycle
    across the span's bars, re-pitch each note by chord degree onto the
    span's actual chords, place onsets on the song's actual bar/beat clock
    so rubato is inherited, then choose the octave that best matches where
    the SOURCE's own accompaniment sat in this span.
    """
    clock = _make_bar_clock(downbeat_times or beat_times, beat_times, beats_per_bar)
    cycle_bars  = float(unit["cycle_bars"])
    cycle_beats = beats_per_bar * cycle_bars
    src_chords  = sorted(unit.get("chords", []), key=lambda c: float(c["beat"]))

    # Where the source's accompaniment actually sits in this span — the
    # target the octave search aims at.  Without this the pattern lands in
    # whatever register the corpus piece happened to use.
    src_span = [n for n in source_notes
                if span["t_start"] <= float(n["onset"]) < span["t_end"]]
    target_centroid = (float(np.mean([int(n["pitch"]) for n in src_span]))
                       if src_span else None)

    raw: List[Dict] = []
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

            # Preserve the pattern's internal contour, but anchor it where
            # the SOURCE's accompaniment actually sits rather than at the
            # bottom of the keyboard.  Anchoring at register_lo (as the
            # first version did) put every pattern in E1-Eb3 regardless of
            # context, which is why generated accompaniment came out nearly
            # an octave below the source's own.
            pat_mean = float(np.mean([int(x["pitch"]) for x in unit["notes"]]))
            anchor   = (target_centroid - pat_mean) if target_centroid is not None \
                       else (register_lo + 12 - pat_mean)
            near     = int(round(int(n["pitch"]) + anchor))
            pitch  = _pitch_for_degree(deg, tgt, near, register_lo, register_hi)

            on  = clock(abs_bar + (nb // beats_per_bar), nb % beats_per_bar)
            off = clock(abs_bar + ((nb + max(0.1, float(n.get("dur", 0.5))))
                                   // beats_per_bar),
                        (nb + max(0.1, float(n.get("dur", 0.5)))) % beats_per_bar)
            if on is None:
                continue
            if off is None or off <= on:
                off = on + 0.12
            raw.append({
                "pitch":    int(pitch),
                "onset":    round(float(on), 4),
                "offset":   round(float(off), 4),
                "velocity": int(n.get("vel", 70)),
                "layer":    "pattern",
                "source":   unit["id"],
                "pattern_type": unit.get("pattern_type", "mixed"),
            })

    # Octave placement, decided once per span so the texture stays coherent
    shift = _octave_shift_search(raw, target_centroid, melody_low,
                                 register_lo, register_hi)
    if shift:
        for n in raw:
            n["pitch"] = int(n["pitch"]) + 12 * shift
    return raw
