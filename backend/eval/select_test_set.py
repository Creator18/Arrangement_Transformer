"""
Pick 50 held-out MAESTRO pieces for the evaluation harness.

Source: data/maestro-v3.0.0/maestro-v3.0.0.csv

Selection rules
---------------
1. Only pieces from MAESTRO's official 'test' split (never used for training)
2. Duration in [30 s, 8 min] — long enough to be meaningful, short enough that
   synthesising all 50 fits comfortably in one render session
3. Stratified by canonical_composer so the sample isn't dominated by whichever
   composer happens to have the most test recordings.  Each composer with
   at least 1 eligible piece gets at least 1 spot; the rest is allocated
   proportionally to that composer's share of the eligible pool
4. Fixed random seed (42) so the picks are deterministic — re-running this
   script always produces the same manifest.  Critical for the regression
   testing workflow downstream

Output: backend/eval/data/manifest.json
    {
      "seed": 42,
      "total_picked": 50,
      "criteria": {...},
      "pieces": [
        {
          "id": "<safe_filename_stem>",
          "composer": "...",
          "title": "...",
          "duration_s": float,
          "year": int,
          "midi_path": "<repo-rooted>",
          "audio_synth_path": "<eval/data/audio/<id>.wav>",
          "split": "test"
        },
        ...
      ]
    }

Run
---
    python backend/eval/select_test_set.py
"""
from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List


# ── config ────────────────────────────────────────────────────────────────────
REPO_ROOT       = Path(__file__).resolve().parent.parent.parent
MAESTRO_DIR     = REPO_ROOT / "data" / "maestro-v3.0.0"
MAESTRO_CSV     = MAESTRO_DIR / "maestro-v3.0.0.csv"
EVAL_DIR        = REPO_ROOT / "backend" / "eval"
MANIFEST_PATH   = EVAL_DIR / "data" / "manifest.json"

N_PICKS         = 50
DURATION_MIN_S  = 30.0
DURATION_MAX_S  = 8 * 60.0          # 8 minutes
SEED            = 42


def _safe_id(midi_filename: str) -> str:
    """
    Turn '2009/MIDI-Unprocessed_..._WAV.midi' into a filesystem-safe id like
    'maestro_2009_unprocessed_11_r1_2009'.  Truncated; uniqueness comes from
    keeping the full midi filename in the manifest separately.
    """
    stem = Path(midi_filename).stem
    safe = re.sub(r"[^a-zA-Z0-9]+", "_", stem).strip("_").lower()
    return safe[:80]


def _stratified_pick(
    eligible: List[Dict],
    n_picks:  int,
    seed:     int,
) -> List[Dict]:
    """
    Stratified sample across canonical_composer.  Each composer with eligible
    pieces gets at least 1 slot.  Remaining slots distributed proportionally
    using the largest-remainder method, then individual pieces drawn
    randomly within each composer.
    """
    rng = random.Random(seed)

    by_composer: Dict[str, List[Dict]] = defaultdict(list)
    for r in eligible:
        by_composer[r["canonical_composer"]].append(r)

    composers = sorted(by_composer.keys())
    if len(composers) >= n_picks:
        # More composers than picks — pick one piece from each of n_picks
        # randomly-chosen composers
        chosen_composers = rng.sample(composers, n_picks)
        picks = []
        for c in chosen_composers:
            picks.append(rng.choice(by_composer[c]))
        return picks

    # Floor: 1 piece per composer, then distribute remaining by share
    total_eligible = sum(len(by_composer[c]) for c in composers)
    quotas = {c: 1 for c in composers}
    remaining = n_picks - len(composers)

    if remaining > 0:
        # Allocate by composer's share, using largest-remainder
        shares = {c: (len(by_composer[c]) / total_eligible) * remaining
                  for c in composers}
        floors = {c: int(shares[c]) for c in composers}
        allocated = sum(floors.values())
        residuals = sorted(composers,
                           key=lambda c: shares[c] - floors[c],
                           reverse=True)
        for c in residuals[: remaining - allocated]:
            floors[c] += 1
        for c in composers:
            quotas[c] += floors[c]
            # Don't exceed the number of pieces this composer actually has
            quotas[c] = min(quotas[c], len(by_composer[c]))

    # If clamping shrank the total, top up from composers with spare capacity
    while sum(quotas.values()) < n_picks:
        spare = [c for c in composers if quotas[c] < len(by_composer[c])]
        if not spare:
            break
        rng.shuffle(spare)
        for c in spare:
            if sum(quotas.values()) >= n_picks:
                break
            quotas[c] += 1

    picks: List[Dict] = []
    for c in composers:
        pool = by_composer[c]
        rng.shuffle(pool)
        picks.extend(pool[: quotas[c]])

    # Trim to exactly n_picks if we somehow overshot
    rng.shuffle(picks)
    return picks[: n_picks]


def main() -> None:
    if not MAESTRO_CSV.exists():
        raise SystemExit(f"MAESTRO CSV not found at {MAESTRO_CSV}")

    with open(MAESTRO_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    test_rows = [r for r in rows if r["split"] == "test"]
    eligible = [
        r for r in test_rows
        if DURATION_MIN_S <= float(r["duration"]) <= DURATION_MAX_S
    ]

    print(f"MAESTRO: {len(rows)} rows total, {len(test_rows)} in test split, "
          f"{len(eligible)} after duration filter [{DURATION_MIN_S:g}s, "
          f"{DURATION_MAX_S:g}s]")

    if len(eligible) < N_PICKS:
        raise SystemExit(f"Only {len(eligible)} eligible pieces, need {N_PICKS}")

    picked = _stratified_pick(eligible, N_PICKS, SEED)

    pieces = []
    for r in picked:
        pid = _safe_id(r["midi_filename"])
        pieces.append({
            "id":               pid,
            "composer":         r["canonical_composer"],
            "title":            r["canonical_title"],
            "duration_s":       round(float(r["duration"]), 2),
            "year":             int(r["year"]),
            "midi_path":        f"data/maestro-v3.0.0/{r['midi_filename']}",
            "audio_synth_path": f"backend/eval/data/audio/{pid}.wav",
            "split":            r["split"],
        })

    # Sort by composer then duration for readability
    pieces.sort(key=lambda p: (p["composer"], p["duration_s"]))

    from collections import Counter
    composer_counts = Counter(p["composer"] for p in pieces)

    manifest = {
        "seed":           SEED,
        "total_picked":   len(pieces),
        "criteria": {
            "split":            "test",
            "duration_min_s":   DURATION_MIN_S,
            "duration_max_s":   DURATION_MAX_S,
            "stratify_by":      "canonical_composer",
        },
        "composer_distribution": dict(sorted(composer_counts.items())),
        "pieces":         pieces,
    }

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nManifest written to {MANIFEST_PATH.relative_to(REPO_ROOT)}")
    print(f"Picked {len(pieces)} pieces across {len(composer_counts)} composers:")
    for composer, count in sorted(composer_counts.items()):
        print(f"  {count:>2}  {composer}")


if __name__ == "__main__":
    main()
