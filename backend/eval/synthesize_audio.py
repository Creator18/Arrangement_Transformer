"""
Synthesize WAV audio for every piece in the eval manifest.

Loads each ground-truth MIDI, converts to our pipeline's note-dict format,
and renders through the same sampler the rest of the system uses (sfizz +
Salamander when available, SF2 fallback otherwise).  Idempotent — pieces
whose audio already exists on disk are skipped.

Why synthesise instead of using MAESTRO's recorded audio
--------------------------------------------------------
The official MAESTRO V3 audio bundle is ~118 GB.  Synthesising from MIDI
through our own sampler is free, deterministic, and removes a download
step.  Trade-off: synthesised audio is cleaner than real recordings —
no mic preamp noise, no room reverb, no performance imperfections beyond
what's in the MIDI.  The eval F1 numbers on synthesised audio will be
somewhat optimistic relative to numbers on real audio.  We mitigate this
later by manually downloading 5–10 real MAESTRO recordings for a
reality-anchor check; the bulk of the eval stays on synthesised audio
because reproducibility matters more than absolute calibration.

Run
---
    python backend/eval/synthesize_audio.py             # all pieces
    python backend/eval/synthesize_audio.py --force     # re-render existing
    python backend/eval/synthesize_audio.py --limit 5   # first 5 only (smoke test)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pretty_midi
import soundfile as sf


REPO_ROOT     = Path(__file__).resolve().parent.parent.parent
MANIFEST_PATH = REPO_ROOT / "backend" / "eval" / "data" / "manifest.json"
AUDIO_DIR     = REPO_ROOT / "backend" / "eval" / "data" / "audio"
SAMPLE_RATE   = 22050


# Make `from app.services.sampler import render_notes` resolve when run from
# the repo root.  The eval scripts intentionally live outside the FastAPI
# package so they can be invoked without spinning up the server.
sys.path.insert(0, str(REPO_ROOT / "backend"))
from app.services.sampler import render_notes


def _midi_to_pedal(midi_path: Path) -> List[Dict]:
    """
    Damper-down intervals from the MIDI's CC64, as {"onset", "offset"}.

    This was missing, and its absence made pedal unmeasurable.  The eval audio
    was rendered from MAESTRO with the sustain pedal stripped, so the model's
    pedal head had nothing to detect and any pedal accuracy figure — or any
    change to pedal handling — was meaningless on this corpus.

    MAESTRO is Disklavier-captured, so its CC64 is a real pedal performance
    rather than a quantised approximation.  Half-pedalling is collapsed to
    a binary at 64, which is what our own pipeline models; capturing partial
    depths would need the sampler to model them too.
    """
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    downs: List[Dict] = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        start = None
        for cc in sorted((c for c in inst.control_changes if c.number == 64),
                         key=lambda c: c.time):
            if cc.value >= 64 and start is None:
                start = float(cc.time)
            elif cc.value < 64 and start is not None:
                downs.append({"onset": start, "offset": float(cc.time)})
                start = None
        if start is not None:
            downs.append({"onset": start, "offset": float(pm.get_end_time())})
    downs.sort(key=lambda d: d["onset"])
    return downs


def _midi_to_notes(midi_path: Path) -> List[Dict]:
    """
    Load a MIDI file and emit the pipeline's standard note-dict list.
    Drum tracks are skipped (MAESTRO has none, but defensive).
    """
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    out: List[Dict] = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            if n.end <= n.start:
                continue
            out.append({
                "pitch":    int(n.pitch),
                "onset":    float(n.start),
                "offset":   float(n.end),
                "velocity": max(1, min(127, int(n.velocity))),
            })
    out.sort(key=lambda n: (n["onset"], n["pitch"]))
    return out


def _render_one(piece: Dict, force: bool) -> Dict:
    midi_path  = REPO_ROOT / piece["midi_path"]
    audio_path = REPO_ROOT / piece["audio_synth_path"]
    audio_path.parent.mkdir(parents=True, exist_ok=True)

    if audio_path.exists() and not force:
        return {"id": piece["id"], "status": "skipped",
                "size_kb": audio_path.stat().st_size // 1024}

    if not midi_path.exists():
        return {"id": piece["id"], "status": "missing_midi",
                "midi_path": str(midi_path)}

    t0 = time.time()
    notes = _midi_to_notes(midi_path)
    if not notes:
        return {"id": piece["id"], "status": "empty_midi"}

    pedal = _midi_to_pedal(midi_path)
    audio = render_notes(notes, sr=SAMPLE_RATE, pedal=pedal)
    sf.write(str(audio_path), audio.astype(np.float32),
             SAMPLE_RATE, subtype="PCM_16")

    return {
        "id":         piece["id"],
        "status":     "rendered",
        "notes":      len(notes),
        "pedal":      len(pedal),
        "duration_s": round(audio.size / SAMPLE_RATE, 2),
        "render_s":   round(time.time() - t0, 2),
        "size_kb":    audio_path.stat().st_size // 1024,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Re-render even if the WAV already exists")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N manifest entries")
    args = parser.parse_args()

    if not MANIFEST_PATH.exists():
        raise SystemExit(f"Manifest not found at {MANIFEST_PATH.relative_to(REPO_ROOT)}\n"
                         f"Run select_test_set.py first.")

    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)

    pieces = manifest["pieces"]
    if args.limit is not None:
        pieces = pieces[: args.limit]

    print(f"Synthesising {len(pieces)} piece(s) into "
          f"{AUDIO_DIR.relative_to(REPO_ROOT)} (sr={SAMPLE_RATE})")
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    total_t0 = time.time()
    for i, piece in enumerate(pieces, 1):
        res = _render_one(piece, force=args.force)
        results.append(res)
        if res["status"] == "rendered":
            print(f"  [{i:>2}/{len(pieces)}] {res['id']}  "
                  f"{res['notes']} notes, {res['duration_s']:.1f}s audio, "
                  f"{res['render_s']:.1f}s render")
        elif res["status"] == "skipped":
            print(f"  [{i:>2}/{len(pieces)}] {res['id']}  (cached, "
                  f"{res['size_kb']} KB)")
        else:
            print(f"  [{i:>2}/{len(pieces)}] {res['id']}  ⚠ {res['status']}")

    elapsed = time.time() - total_t0
    rendered = sum(1 for r in results if r["status"] == "rendered")
    skipped  = sum(1 for r in results if r["status"] == "skipped")
    failed   = sum(1 for r in results if r["status"] not in ("rendered", "skipped"))

    print(f"\nDone in {elapsed:.1f}s — "
          f"rendered={rendered}, cached={skipped}, failed={failed}")


if __name__ == "__main__":
    main()
