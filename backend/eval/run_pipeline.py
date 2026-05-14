"""
Run the transcription half of the pipeline on a single (audio, midi) pair.

This is the "subject under test" for the eval harness.  It executes the
same code paths the FastAPI routers do — separate_components,
transcribe_components, analyse_audio — but without spinning up an HTTP
server.  Each invocation creates an isolated session directory under
backend/eval/runs/<piece_id>/ so multiple pieces can be evaluated in
parallel later without clobbering each other.

The ORIGINAL transcription is what we measure against ground truth.  That's
the audio → 88-key-MIDI step, which is what Klangio competes on.  The
TRANSFORMED side (88 → 61 with playability rules) is a different problem
and gets evaluated separately via blind A/B, not against MAESTRO MIDI.

What we run
-----------
    1. separate_components  → bass.wav, upper.wav, REPET stems, beat/downbeat
    2. transcribe_components → transcription.json (notes + timing_feel)
    3. analyse_audio        → segments.json (chord_map, tempo, downbeats)
                              + applies ORIGINAL-side cleanup
                              (scale-snap + hand-limits) in-place

We deliberately stop before /transform because transcription accuracy is
what's measured here.  The transform stage is a separate evaluation
problem.

Usage
-----
    python backend/eval/run_pipeline.py <piece_id>
    python backend/eval/run_pipeline.py --all              # every manifest entry
    python backend/eval/run_pipeline.py --all --force      # re-run cached
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional


REPO_ROOT     = Path(__file__).resolve().parent.parent.parent
MANIFEST_PATH = REPO_ROOT / "backend" / "eval" / "data" / "manifest.json"
RUNS_DIR      = REPO_ROOT / "backend" / "eval" / "runs"

sys.path.insert(0, str(REPO_ROOT / "backend"))

# The FastAPI app's main.py applies a couple of process-wide shims at startup:
#   1. scipy 1.8+ removed scipy.signal.hann; librosa-internal code still calls
#      the old name, so we re-attach it from scipy.signal.windows.hann
#   2. Bundled FFmpeg gets prepended to PATH so madmom's resampling fallback
#      works without a system install
# These eval scripts intentionally don't import main.py (we don't want the
# FastAPI server starting up), so we re-apply the same shims here.
import scipy.signal
import scipy.signal.windows
if not hasattr(scipy.signal, "hann"):
    scipy.signal.hann = scipy.signal.windows.hann
import os as _os
for _ff in [REPO_ROOT / "ffmpeg-8.1.1-essentials_build" / "ffmpeg-8.1.1-essentials_build" / "bin",
            REPO_ROOT / "ffmpeg-8.1.1-full_build-shared" / "ffmpeg-8.1.1-full_build-shared" / "bin"]:
    if _ff.exists():
        _os.environ["PATH"] = str(_ff) + _os.pathsep + _os.environ.get("PATH", "")
        break


def _run_one(piece: Dict, force: bool = False) -> Dict:
    """
    Execute separate → transcribe → segment on one piece.  Returns a
    summary dict with timing, status, and the path to the resulting
    transcription.json.

    On success the run directory contains everything the comparator needs:
        runs/<id>/audio.wav                  — symlink/copy of synth audio
        runs/<id>/components/{bass,upper,...}.wav
        runs/<id>/components.json
        runs/<id>/transcription.json         — what we evaluate
        runs/<id>/segments.json
        runs/<id>/pattern_clusters.json
        runs/<id>/run_summary.json           — timing breakdown
    """
    piece_id   = piece["id"]
    run_dir    = RUNS_DIR / piece_id
    audio_src  = REPO_ROOT / piece["audio_synth_path"]
    transcr    = run_dir / "transcription.json"

    if transcr.exists() and not force:
        return {"id": piece_id, "status": "cached",
                "transcription_path": str(transcr.relative_to(REPO_ROOT))}

    if not audio_src.exists():
        return {"id": piece_id, "status": "missing_audio",
                "audio_path": str(audio_src.relative_to(REPO_ROOT))}

    # Fresh run dir.  Note: this also wipes anything cached from a previous
    # partial run, which is correct — we don't want stale components.json
    # from before a separation tweak.
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    shutil.copy2(audio_src, run_dir / "audio.wav")

    # Import lazily so the help banner doesn't pay the model-load cost
    from app.services.separation   import separate_components
    from app.services.transcription import transcribe_components
    from app.services.segmentation import analyse_audio

    timings: Dict[str, float] = {}
    summary: Dict = {"id": piece_id, "status": "ok"}

    try:
        t = time.time()
        separate_components(run_dir / "audio.wav", run_dir)
        timings["separate_s"] = round(time.time() - t, 2)

        t = time.time()
        notes, timing_feel = transcribe_components(run_dir)
        timings["transcribe_s"] = round(time.time() - t, 2)

        # Persist the transcription in the same shape the FastAPI router does,
        # so the segment-side ORIGINAL cleanup pass can find and rewrite it.
        with open(run_dir / "transcription.json", "w", encoding="utf-8") as f:
            json.dump({
                "session_id":  piece_id,
                "notes":       notes,
                "timing_feel": timing_feel,
            }, f)

        t = time.time()
        seg_result = analyse_audio(run_dir / "audio.wav", run_dir)
        timings["segment_s"] = round(time.time() - t, 2)

        # Mirror what the FastAPI router does after analyse_audio: apply
        # ORIGINAL-side cleanup (scale snap + hand limits) to transcription.json.
        from app.routers.segment import _apply_original_cleanup
        try:
            cleanup_info = _apply_original_cleanup(run_dir, seg_result)
            summary["cleanup"] = cleanup_info
        except Exception as e:
            summary["cleanup_error"] = str(e)

        summary["notes_final"] = len(json.loads(
            (run_dir / "transcription.json").read_text(encoding="utf-8")
        )["notes"])
        summary["timings"] = timings
        summary["total_s"] = round(sum(timings.values()), 2)
        summary["transcription_path"] = str(
            (run_dir / "transcription.json").relative_to(REPO_ROOT)
        )

    except Exception as e:
        summary["status"] = "error"
        summary["error"]  = f"{type(e).__name__}: {e}"
        summary["timings"] = timings

    with open(run_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("piece_id", nargs="?", default=None,
                        help="Manifest piece id (run this one)")
    parser.add_argument("--all", action="store_true",
                        help="Run every manifest entry")
    parser.add_argument("--force", action="store_true",
                        help="Re-run pieces that have a cached transcription")
    parser.add_argument("--limit", type=int, default=None,
                        help="With --all, only the first N pieces")
    args = parser.parse_args()

    if not MANIFEST_PATH.exists():
        raise SystemExit("Manifest missing. Run select_test_set.py first.")

    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)

    pieces: List[Dict] = manifest["pieces"]

    if args.all:
        target = pieces[: args.limit] if args.limit else pieces
    elif args.piece_id:
        target = [p for p in pieces if p["id"] == args.piece_id]
        if not target:
            raise SystemExit(f"No piece with id={args.piece_id} in manifest")
    else:
        parser.print_help()
        return

    print(f"Running pipeline on {len(target)} piece(s) "
          f"into {RUNS_DIR.relative_to(REPO_ROOT)}")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict] = []
    total_t0 = time.time()
    for i, p in enumerate(target, 1):
        res = _run_one(p, force=args.force)
        summaries.append(res)
        if res["status"] == "ok":
            t = res.get("timings", {})
            print(f"  [{i:>2}/{len(target)}] {res['id']}  "
                  f"sep={t.get('separate_s')}s "
                  f"trans={t.get('transcribe_s')}s "
                  f"seg={t.get('segment_s')}s  "
                  f"notes={res.get('notes_final')}")
        elif res["status"] == "cached":
            print(f"  [{i:>2}/{len(target)}] {res['id']}  (cached)")
        else:
            print(f"  [{i:>2}/{len(target)}] {res['id']}  ! {res['status']}: "
                  f"{res.get('error', '')}")

    elapsed = time.time() - total_t0
    ok      = sum(1 for s in summaries if s["status"] == "ok")
    cached  = sum(1 for s in summaries if s["status"] == "cached")
    failed  = sum(1 for s in summaries if s["status"] not in ("ok", "cached"))

    print(f"\nDone in {elapsed:.1f}s — ok={ok}, cached={cached}, failed={failed}")


if __name__ == "__main__":
    main()
