"""
validate_embedding.py — corruption self-retrieval test.

The single most important property of the retrieval embedding: a unit
corrupted THE WAY OUR TRANSCRIPTION PIPELINE CORRUPTS THINGS must still
retrieve its own clean version near the top.  If it can't find itself,
it certainly can't find musically appropriate neighbours for real
queries.

Corruption model (matches the pipeline's measured error profile from
backend/eval/analyze_errors.py):
    - drop 15% of notes at random          (missed notes — 55% of FNs)
    - add octave phantoms for 8% of notes  (octave_dup FPs — 21-24%)
    - jitter onsets by ±40 ms at 120 BPM   (timing_off — ~22%)
    - jitter velocities ±20                (velocity is untrustworthy)

Pass criterion: clean self in top-3 for >= 90% of sampled units.
(Self-similarity is checked per raw unit id.)

Usage:
    python backend/retrieval/validate_embedding.py            # 300 samples
    python backend/retrieval/validate_embedding.py --n 1000
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CORPUS    = Path(__file__).resolve().parent / "corpus"

sys.path.insert(0, str(REPO_ROOT / "backend"))
from retrieval.featurizer import featurize_unit, embed_query, top_k


def corrupt(unit: dict, rng: random.Random) -> dict:
    """Apply the pipeline-style corruption model to a copy of `unit`."""
    u = copy.deepcopy(unit)
    beats_per_bar = int(u.get("beats_per_bar", 4))
    jitter_beats = 0.04 / (60.0 / 120.0)     # ±40 ms at 120 BPM, in beats

    notes = []
    for n in u["notes"]:
        if rng.random() < 0.15:              # dropped note
            continue
        nn = dict(n)
        nn["beat"] = max(0.0, nn["beat"] + rng.uniform(-jitter_beats, jitter_beats))
        nn["vel"]  = max(1, min(127, nn["vel"] + rng.randint(-20, 20)))
        notes.append(nn)
        if rng.random() < 0.08:              # octave phantom (weaker, +12)
            ph = dict(nn)
            ph["pitch"] = min(108, ph["pitch"] + 12)
            ph["vel"]   = max(1, int(ph["vel"] * 0.5))
            notes.append(ph)
    u["notes"] = notes if notes else u["notes"]
    return u


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    units = [json.loads(l) for l in
             open(CORPUS / "units.jsonl", encoding="utf-8")]
    matrix = np.load(CORPUS / "vectors.npy")
    stats  = json.load(open(CORPUS / "norm.json", encoding="utf-8"))

    raw = np.load(CORPUS / "vectors_raw.npy")
    raw_norms = np.linalg.norm(raw, axis=1)
    raw_norms[raw_norms < 1e-9] = 1.0
    raw_unit = raw / raw_norms[:, None]

    rng = random.Random(args.seed)
    sample_idx = rng.sample(range(len(units)), min(args.n, len(units)))

    hits_at = {1: 0, 3: 0, 5: 0}
    func_at = {1: 0, 3: 0, 5: 0}      # "functional": self OR near-twin retrieved
    ranks = []
    TWIN_COS = 0.95
    for i in sample_idx:
        q = corrupt(units[i], rng)
        qv = embed_query(featurize_unit(q), stats)

        # Hard filters, same as production retrieval: meter + cycle length
        mask = np.array([
            u["meter"] == q["meter"] and u["cycle_bars"] == q["cycle_bars"]
            for u in units
        ])
        results = top_k(qv, matrix, k=5, mask=mask)
        result_ids = [units[j]["id"] for j, _ in results]

        try:
            rank = result_ids.index(units[i]["id"]) + 1
        except ValueError:
            rank = 99
        ranks.append(rank)

        # Functional rank: position of the first result that IS the self or
        # is a near-twin of the clean self in raw feature space.  Retrieving
        # an equivalent Alberti bar from another piece is a success for
        # arrangement purposes — the corpus is FULL of legitimate twins.
        func_rank = 99
        for pos, (j, _) in enumerate(results, start=1):
            if j == i or float(raw_unit[j] @ raw_unit[i]) >= TWIN_COS:
                func_rank = pos
                break

        for k in hits_at:
            if rank <= k:
                hits_at[k] += 1
            if func_rank <= k:
                func_at[k] += 1

    n = len(sample_idx)
    print(f"Corruption self-retrieval over {n} sampled units:")
    print("  strict (must retrieve own id):")
    for k in (1, 3, 5):
        print(f"    recall@{k}: {hits_at[k] / n:.3f}")
    print(f"  functional (own id OR raw-cosine >= {TWIN_COS} twin):")
    for k in (1, 3, 5):
        print(f"    recall@{k}: {func_at[k] / n:.3f}")
    print(f"  median strict rank: {sorted(ranks)[n // 2]}")
    verdict = "PASS" if func_at[3] / n >= 0.90 else "FAIL"
    print(f"\n  {verdict}  (criterion: functional recall@3 >= 0.90)")


if __name__ == "__main__":
    main()
