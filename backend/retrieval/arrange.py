"""
arrange.py — end-to-end retrieval-augmented arrangement of one session.

Reads a completed session directory (transcription.json, segments.json,
components.json), runs the retrieval engine over it, and returns the
arranged note list: the source melody, plus accompaniment borrowed from
the corpus and re-pitched onto the source's own harmony.

Run standalone against any eval run directory:

    python backend/retrieval/arrange.py backend/eval/runs/<piece_id>
    python backend/retrieval/arrange.py <dir> --out arranged.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

from retrieval.engine import (
    Corpus, split_melody_accompaniment, segment_spans, build_query_unit,
    select_pattern, instantiate,
)


def _chord_map_from_json(data: Dict) -> Dict[int, tuple]:
    return {int(k): (int(v[0]), set(v[1])) for k, v in data.items()}


def arrange_session(session_dir: Path, corpus: Optional[Corpus] = None,
                    verbose: bool = True) -> Dict:
    session_dir = Path(session_dir)
    trans = json.load(open(session_dir / "transcription.json", encoding="utf-8"))
    segs  = json.load(open(session_dir / "segments.json",      encoding="utf-8"))
    comps = json.load(open(session_dir / "components.json",    encoding="utf-8"))

    notes         = trans["notes"]
    tempo         = float(segs.get("tempo_bpm", 120.0))
    meter         = segs.get("time_signature", "4/4")
    chord_map     = _chord_map_from_json(segs.get("chord_map", {}))
    beats_per_bar = int(comps.get("beats_per_bar", 4))
    sr, hop       = int(comps.get("sr", 22050)), int(comps.get("hop_length", 512))
    beat_times     = [f * hop / sr for f in comps.get("beat_frames", [])]
    downbeat_times = [f * hop / sr for f in comps.get("downbeat_frames", [])]
    measure_dur    = (60.0 / tempo) * beats_per_bar

    melody, accomp = split_melody_accompaniment(notes)
    if verbose:
        print(f"melody {len(melody)} notes | accompaniment {len(accomp)} notes")

    spans = segment_spans(accomp, downbeat_times, chord_map,
                          beats_per_bar, measure_dur)
    if verbose:
        print(f"{len(spans)} spans "
              f"({', '.join(str(s['bar_end']-s['bar_start']) for s in spans[:12])}"
              f"{' ...' if len(spans) > 12 else ''} bars)")

    corpus = corpus or Corpus()
    out_accomp: List[Dict] = []
    provenance: List[Dict] = []
    prev_type: Optional[str] = None

    for span in spans:
        q = build_query_unit(accomp, span, beat_times, chord_map,
                             beats_per_bar, meter, measure_dur)
        if q is None:
            continue
        span_mel = [n for n in melody
                    if span["t_start"] <= float(n["onset"]) < span["t_end"]]
        # Robust melody floor: the 20th percentile, not the absolute minimum.
        # Clearance measured against the single lowest melody note (a dip, an
        # ornament, a skyline mis-assignment) pins the accompaniment far lower
        # than the tune actually sits.
        mel_low = (int(np.percentile([int(n["pitch"]) for n in span_mel], 20))
                   if span_mel else None)

        picked = select_pattern(corpus, q, mel_low, prev_type)
        if picked is None:
            continue
        unit, score, diag = picked
        prev_type = unit.get("pattern_type")

        gen = instantiate(unit, span, beat_times, chord_map, beats_per_bar,
                          downbeat_times=downbeat_times,
                          source_notes=accomp,
                          melody_low=mel_low)
        out_accomp.extend(gen)
        provenance.append({
            "bars":         [span["bar_start"], span["bar_end"]],
            "source":       unit["id"],
            "pattern_type": unit.get("pattern_type"),
            "style":        unit.get("style"),
            "notes_out":    len(gen),
            **diag,
        })

    arranged = sorted(melody + out_accomp,
                      key=lambda n: (float(n["onset"]), int(n["pitch"])))
    return {
        "session_id":  trans.get("session_id"),
        "notes":       arranged,
        "provenance":  provenance,
        "counts": {
            "melody_kept":        len(melody),
            "accomp_source":      len(accomp),
            "accomp_generated":   len(out_accomp),
            "total":              len(arranged),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    res = arrange_session(Path(args.session_dir))
    print()
    print("counts:", res["counts"])
    print()
    print("provenance (first 10 spans):")
    for p in res["provenance"][:10]:
        print(f"  bars {p['bars'][0]:>3}-{p['bars'][1]:<3} "
              f"{p['pattern_type']:<16} from {p['style']:<10} "
              f"sim={p['sim']:.3f} score={p['score']:.3f} -> {p['notes_out']} notes")
    from collections import Counter
    print()
    print("pattern types used:",
          dict(Counter(p["pattern_type"] for p in res["provenance"])))

    if args.out:
        json.dump(res, open(args.out, "w", encoding="utf-8"))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
