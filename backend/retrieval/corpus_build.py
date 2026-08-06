"""
corpus_build.py — extract figuration-cycle units from MusicXML scores.

Walks a score corpus (ASAP for v1 — its MusicXML is clean and already on
disk), and for each score:
    1. Parse with music21; take the LOWER staff as the LH part.  Scores
       encode the hand split for us — this is exactly the supervision the
       audio pipeline never gets.
    2. Chordify the full score per beat for harmonic context.
    3. Walk bar by bar.  For each bar, build a candidate unit; detect the
       cycle length by comparing rhythm profiles of neighbouring bars
       (bar ≈ next bar → 1-bar cycle; half-bars similar → 0.5;
       pairs similar → 2.0).
    4. Deduplicate near-identical units per piece (an Alberti bar repeated
       40 times contributes a handful of entries, not 40).

Outputs (backend/retrieval/corpus/):
    units.jsonl    one unit dict per line (featurizer.py format + provenance)
    vectors.npy    raw feature matrix, row i ↔ line i of units.jsonl
    norm.json      normalization stats (mean/std/block weights) for queries
    build_log.txt  per-score status

Usage:
    python backend/retrieval/corpus_build.py --limit 5     # smoke test
    python backend/retrieval/corpus_build.py               # full ASAP
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT  = Path(__file__).resolve().parent.parent.parent
ASAP_DIR   = REPO_ROOT / "data" / "asap-dataset"
OUT_DIR    = Path(__file__).resolve().parent / "corpus"

sys.path.insert(0, str(REPO_ROOT / "backend"))
from retrieval.featurizer import featurize_unit, normalize_matrix

# music21 is heavyweight to import — do it lazily in main() so --help is fast.


# ── chord inference from the score itself ────────────────────────────────────

_TRIADS = {
    (0, 4, 7): "maj", (0, 3, 7): "min", (0, 3, 6): "dim", (0, 4, 8): "aug",
}


def _chords_per_beat(chordified, beats_per_bar: int) -> Dict[int, Dict]:
    """
    music21 chordified stream → {global_beat_index: {"root_pc", "pcs"}}.
    Picks the most plausible root by testing each pc as root against
    triad templates; falls back to the lowest pc.
    """
    out: Dict[int, Dict] = {}
    for ch in chordified.recurse().getElementsByClass("Chord"):
        beat_idx = int(ch.getOffsetInHierarchy(chordified))
        pcs = sorted(set(p.midi % 12 for p in ch.pitches))
        if not pcs:
            continue
        root = None
        for cand in pcs:
            rel = tuple(sorted((p - cand) % 12 for p in pcs))
            for tpl in _TRIADS:
                if set(tpl).issubset(set(rel)):
                    root = cand
                    break
            if root is not None:
                break
        if root is None:
            root = min(pcs)
        # Keep the FIRST chord seen at each beat (chordify emits sub-beat
        # slices; the beat's opening harmony is the useful label).
        if beat_idx not in out:
            out[beat_idx] = {"root_pc": int(root), "pcs": [int(p) for p in pcs]}
    return out


# ── rhythm profile similarity (for cycle detection + dedupe) ─────────────────

def _rhythm_profile(notes: List[Dict], beats_per_bar: int, slots: int = 16) -> np.ndarray:
    prof = np.zeros(slots, dtype=np.float32)
    for n in notes:
        s = int((float(n["beat"]) % beats_per_bar) / beats_per_bar * slots)
        prof[min(s, slots - 1)] += 1.0
    m = prof.max()
    return prof / m if m > 0 else prof


def _similar(a: np.ndarray, b: np.ndarray, thresh: float = 0.80) -> bool:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return False
    return float(a @ b / (na * nb)) >= thresh


# ── per-score extraction ─────────────────────────────────────────────────────

def extract_units_from_score(xml_path: Path, style: str) -> List[Dict]:
    import music21

    score = music21.converter.parse(str(xml_path))
    parts = score.parts
    if len(parts) < 2:
        return []                                  # need two staves for a hand split
    lh_part = parts[-1]                            # lower staff = LH

    # Meter (first time signature; pieces with mid-score changes contribute
    # only their opening-meter bars in v1)
    ts = lh_part.recurse().getElementsByClass("TimeSignature").first()
    if ts is None:
        return []
    beats_per_bar = int(ts.numerator * (4 / ts.denominator))
    if beats_per_bar < 2 or beats_per_bar > 12:
        return []
    meter = f"{ts.numerator}/{ts.denominator}"

    chords_by_beat = _chords_per_beat(score.chordify(), beats_per_bar)

    # Collect LH notes as (global_beat, dur_beats, pitch, vel)
    lh_notes: List[Dict] = []
    for n in lh_part.recurse().notes:
        beat = float(n.getOffsetInHierarchy(lh_part))
        dur  = float(n.duration.quarterLength)
        vel  = int(n.volume.velocity or 72)
        pitches = n.pitches if hasattr(n, "pitches") else [n.pitch]
        for p in pitches:
            lh_notes.append({"beat": beat, "dur": dur, "pitch": int(p.midi), "vel": vel})
    if not lh_notes:
        return []

    n_bars = int(max(n["beat"] for n in lh_notes) // beats_per_bar) + 1

    # Bar-indexed views
    bars: List[List[Dict]] = [[] for _ in range(n_bars)]
    for n in lh_notes:
        bi = int(n["beat"] // beats_per_bar)
        if 0 <= bi < n_bars:
            bars[bi].append(n)
    profiles = [_rhythm_profile(b, beats_per_bar) for b in bars]

    units: List[Dict] = []
    kept_profiles: List[np.ndarray] = []

    for bi in range(n_bars):
        if len(bars[bi]) < 2:
            continue

        # Cycle length: compare this bar's rhythm with its neighbour(s)
        if bi + 1 < n_bars and _similar(profiles[bi], profiles[bi + 1]):
            cycle = 1.0
        elif (bi + 3 < n_bars
              and _similar(profiles[bi],     profiles[bi + 2])
              and _similar(profiles[bi + 1], profiles[bi + 3])):
            cycle = 2.0
        else:
            cycle = 1.0                             # default: treat as 1-bar

        # Per-piece dedupe: skip if a kept unit already has this rhythm
        if any(_similar(profiles[bi], kp, 0.92) for kp in kept_profiles):
            continue

        span_bars = int(cycle)
        span_notes: List[Dict] = []
        for off in range(span_bars):
            if bi + off < n_bars:
                span_notes.extend(bars[bi + off])
        if len(span_notes) < 2:
            continue

        base_beat = bi * beats_per_bar
        unit = {
            "id":            f"{style}:{xml_path.parent.name}:bar{bi}",
            "meter":         meter,
            "beats_per_bar": beats_per_bar,
            "cycle_bars":    cycle,
            "style":         style,
            "notes": [
                {"beat": round(n["beat"] - base_beat, 4),
                 "dur":  round(n["dur"], 4),
                 "pitch": n["pitch"],
                 "vel":  n["vel"]}
                for n in span_notes
            ],
            "chords": [
                {"beat": b - base_beat, **chords_by_beat[b]}
                for b in range(base_beat, base_beat + span_bars * beats_per_bar)
                if b in chords_by_beat
            ],
        }
        units.append(unit)
        kept_profiles.append(profiles[bi])

    return units


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Only the first N scores (smoke test)")
    args = parser.parse_args()

    if not ASAP_DIR.exists():
        raise SystemExit(f"ASAP not found at {ASAP_DIR}")

    xml_scores = sorted(set(ASAP_DIR.glob("*/*/*/xml_score.musicxml"))
                        | set(ASAP_DIR.glob("*/*/xml_score.musicxml")))
    if args.limit:
        xml_scores = xml_scores[: args.limit]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_units: List[Dict] = []
    log_lines: List[str] = []
    t0 = time.time()

    for i, xml in enumerate(xml_scores, 1):
        style = xml.relative_to(ASAP_DIR).parts[0]      # composer dir
        try:
            units = extract_units_from_score(xml, style)
            all_units.extend(units)
            msg = f"[{i}/{len(xml_scores)}] {xml.parent.name}: {len(units)} units"
        except Exception as e:
            msg = f"[{i}/{len(xml_scores)}] {xml.parent.name}: FAILED {type(e).__name__}: {e}"
        print(msg)
        log_lines.append(msg)

    # Featurize everything, drop degenerate rows
    raw = np.stack([featurize_unit(u) for u in all_units]) if all_units else np.zeros((0, 1))
    keep = np.linalg.norm(raw, axis=1) > 1e-6
    all_units = [u for u, k in zip(all_units, keep) if k]
    raw = raw[keep]

    matrix, stats = normalize_matrix(raw)

    with open(OUT_DIR / "units.jsonl", "w", encoding="utf-8") as f:
        for u in all_units:
            f.write(json.dumps(u) + "\n")
    np.save(OUT_DIR / "vectors.npy", matrix)
    np.save(OUT_DIR / "vectors_raw.npy", raw)
    with open(OUT_DIR / "norm.json", "w", encoding="utf-8") as f:
        json.dump(stats, f)
    (OUT_DIR / "build_log.txt").write_text("\n".join(log_lines), encoding="utf-8")

    from collections import Counter
    meters = Counter(u["meter"] for u in all_units)
    styles = Counter(u["style"] for u in all_units)
    print(f"\nDone in {time.time()-t0:.1f}s: {len(all_units)} units "
          f"from {len(xml_scores)} scores")
    print("meters:", dict(meters.most_common()))
    print("styles:", dict(styles.most_common(8)))


if __name__ == "__main__":
    main()
