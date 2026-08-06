"""
build_dataset.py — training data for the score-informed correction model.

The idea: our pipeline hears audio and guesses notes.  Real scores say what
those notes *should* have been.  Train on (our guess, the score) pairs and
the model learns the systematic shape of our mistakes — then at inference,
where no score exists, it applies those learned corrections.

This module builds the reference side and the manifest.  Two jobs:

  1. CONTAMINATION FILTER.  ASAP overlaps MAESTRO heavily, and our eval
     set is 50 MAESTRO test-split pieces.  Training on any of them —
     including a *different performance of the same composition* — leaks
     the test set and makes every subsequent COnPOff gain partly fake.
     Composer names differ between the two datasets ("Frédéric Chopin" vs
     "Chopin"), so matching is done on ASCII-folded surnames plus
     normalised title substrings, not raw strings.  A naive comparison
     silently reports zero overlap, which is how this nearly shipped.

  2. SCORE TARGETS.  Parse each MusicXML into the notation facts audio
     cannot supply:
       - hand: which staff the note is written on.  Our transcriber has
         to guess this from pitch; the score simply states it.
       - voice: inner-voice membership within a staff.
       - notated duration in beats, already quantised by definition.
         This is the big one — COnPOff is our worst metric, and whether a
         note is a staccato quarter or an eighth-plus-rest is a NOTATION
         convention, not an acoustic fact.  It cannot be learned from
         audio at all; it can only come from scores.
       - tie/grace flags, so ornaments and tied groups aren't treated as
         ordinary onsets.

Usage:
    python backend/correction/build_dataset.py --limit 5   # smoke test
    python backend/correction/build_dataset.py             # full
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ASAP_DIR  = REPO_ROOT / "data" / "asap-dataset"
EVAL_MAN  = REPO_ROOT / "backend" / "eval" / "data" / "manifest.json"
OUT_DIR   = Path(__file__).resolve().parent / "data"

sys.path.insert(0, str(REPO_ROOT / "backend"))


# ── name normalisation (the thing that silently broke the first attempt) ─────

def _ascii(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def norm_title(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", _ascii(s).lower())


def surname(composer: str) -> str:
    c = _ascii(composer).split("/")[0].strip()      # "Wagner / Liszt" -> "Wagner"
    parts = c.split()
    return parts[-1].lower() if parts else ""


# ── 1. contamination filter ──────────────────────────────────────────────────

def contaminated_folders() -> Tuple[Set[str], Dict[str, int]]:
    """
    ASAP score folders that must not be trained on, because they share a
    performance or a composition with the eval set.
    """
    man  = json.load(open(EVAL_MAN, encoding="utf-8"))
    rows = list(csv.DictReader(open(ASAP_DIR / "metadata.csv", encoding="utf-8")))

    eval_midis = {Path(p["midi_path"]).name for p in man["pieces"]}
    eval_titles: Dict[str, List[str]] = {}
    for p in man["pieces"]:
        eval_titles.setdefault(surname(p["composer"]), []).append(norm_title(p["title"]))

    bad: Set[str] = set()
    counts = {"by_performance": 0, "by_composition": 0}
    for r in rows:
        mm = r.get("maestro_midi_performance", "").strip()
        hit_perf = bool(mm) and Path(mm).name in eval_midis
        t = norm_title(r["title"])
        hit_comp = any(t and (t in e or e in t)
                       for e in eval_titles.get(surname(r["composer"]), []))
        if hit_perf:
            counts["by_performance"] += 1
        if hit_comp:
            counts["by_composition"] += 1
        if hit_perf or hit_comp:
            bad.add(r["folder"])
    return bad, counts


# ── 2. score targets ─────────────────────────────────────────────────────────

# Notated durations in quarter-note beats -> their names, for the model's
# duration vocabulary.
_DUR_NAMES = {
    0.125: "32nd", 0.1875: "dotted32", 0.25: "16th", 0.375: "dotted16",
    1/3: "8thTriplet", 0.5: "8th", 0.75: "dotted8", 2/3: "quarterTriplet",
    1.0: "quarter", 1.5: "dotted-quarter", 2.0: "half", 3.0: "dotted-half",
    4.0: "whole", 6.0: "dotted-whole",
}


def _nearest_dur_name(ql: float) -> str:
    if ql <= 0:
        return "unknown"
    best = min(_DUR_NAMES, key=lambda d: abs(d - ql))
    return _DUR_NAMES[best] if abs(best - ql) < 0.06 else "other"


def extract_score_targets(xml_path: Path) -> Optional[Dict]:
    """
    MusicXML -> the reference notation the correction model is trained
    toward.  Times are in quarter-note beats from the start of the score,
    which is the natural unit for notation (and tempo-independent).
    """
    import music21

    try:
        score = music21.converter.parse(str(xml_path))
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

    parts = list(score.parts)
    if not parts:
        return None

    ts = score.recurse().getElementsByClass("TimeSignature").first()
    meter = f"{ts.numerator}/{ts.denominator}" if ts else "4/4"
    ks = score.recurse().getElementsByClass("KeySignature").first()
    key_sharps = int(ks.sharps) if ks is not None else 0

    notes: List[Dict] = []
    for pi, part in enumerate(parts):
        # Convention: first staff is the right hand, last is the left.
        hand = "RH" if pi == 0 else "LH"
        for n in part.recurse().notes:
            ql = float(n.duration.quarterLength)
            if ql <= 0:
                continue
            onset = float(n.getOffsetInHierarchy(score))
            tie = None
            if getattr(n, "tie", None) is not None:
                tie = str(n.tie.type)
            pitches = n.pitches if hasattr(n, "pitches") else [n.pitch]
            for p in pitches:
                notes.append({
                    "pitch":     int(p.midi),
                    "onset_b":   round(onset, 4),
                    "dur_b":     round(ql, 4),
                    "dur_name":  _nearest_dur_name(ql),
                    "hand":      hand,
                    "voice":     int(getattr(n, "voice", 1) or 1)
                                 if isinstance(getattr(n, "voice", 1), int) else 1,
                    "tie":       tie,
                    "grace":     bool(n.duration.isGrace),
                    "spelling":  p.name,          # F# vs Gb — audio cannot tell
                })
    if not notes:
        return None
    notes.sort(key=lambda x: (x["onset_b"], x["pitch"]))
    return {
        "meter":       meter,
        "key_sharps":  key_sharps,
        "n_notes":     len(notes),
        "n_staves":    len(parts),
        "notes":       notes,
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    bad, counts = contaminated_folders()
    rows = list(csv.DictReader(open(ASAP_DIR / "metadata.csv", encoding="utf-8")))
    clean = [r for r in rows if r["folder"] not in bad]

    print(f"ASAP: {len(rows)} performances over "
          f"{len({r['folder'] for r in rows})} scores")
    print(f"contamination: {counts['by_performance']} by performance, "
          f"{counts['by_composition']} by composition -> "
          f"{len(bad)} score folders excluded")
    print(f"clean: {len(clean)} performances over "
          f"{len({r['folder'] for r in clean})} scores")

    folders = sorted({r["folder"] for r in clean})
    if args.limit:
        folders = folders[: args.limit]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    targets_path = OUT_DIR / "score_targets.jsonl"
    ok = failed = 0
    t0 = time.time()
    with open(targets_path, "w", encoding="utf-8") as out:
        for i, folder in enumerate(folders, 1):
            xml = ASAP_DIR / folder / "xml_score.musicxml"
            if not xml.exists():
                failed += 1
                continue
            res = extract_score_targets(xml)
            if not res or "error" in res:
                failed += 1
                print(f"  [{i}/{len(folders)}] {folder}: FAILED "
                      f"{res.get('error') if res else 'empty'}")
                continue
            res["folder"] = folder
            out.write(json.dumps(res) + "\n")
            ok += 1
            if i % 25 == 0 or i == len(folders):
                print(f"  [{i}/{len(folders)}] ok={ok} failed={failed}")

    manifest = {
        "contamination": {**counts, "excluded_folders": sorted(bad)},
        "n_clean_performances": len(clean),
        "n_scores_processed":   ok,
        "performances": [
            {"folder": r["folder"], "perf_midi": r["midi_performance"],
             "composer": r["composer"], "title": r["title"]}
            for r in clean if r["folder"] in set(folders)
        ],
    }
    json.dump(manifest, open(OUT_DIR / "train_manifest.json", "w", encoding="utf-8"))
    print(f"\ndone in {time.time()-t0:.0f}s: {ok} scores -> "
          f"{targets_path.relative_to(REPO_ROOT)}, {failed} failed")


if __name__ == "__main__":
    main()
