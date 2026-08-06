"""
build_pairs.py — produce (our extraction, score truth) training pairs.

For each clean ASAP performance:
    1. synthesize audio from the performance MIDI (same sampler the eval
       harness uses, so the audio domain matches what we measure on)
    2. run OUR pipeline over it — separation, transcription, segment
    3. map our notes into score-beat space with the ASAP annotations
    4. pair them against the score targets

The result is the correction model's training set: for every note we
produced, what the score says it should have been; plus what we invented
and what we missed.

This is deliberately the same pipeline the eval harness runs, not a
shortcut — a correction model must learn the error profile of the
system it will actually be correcting.

    python backend/correction/build_pairs.py --limit 8
    python backend/correction/build_pairs.py --limit 8 --report-only
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ASAP_DIR  = REPO_ROOT / "data" / "asap-dataset"
OUT_DIR   = Path(__file__).resolve().parent / "data"
WORK_DIR  = OUT_DIR / "work"

sys.path.insert(0, str(REPO_ROOT / "backend"))

# Same startup shims the FastAPI app installs (see eval/run_pipeline.py)
import scipy.signal, scipy.signal.windows
if not hasattr(scipy.signal, "hann"):
    scipy.signal.hann = scipy.signal.windows.hann

from correction.align import BeatMap, align_notes, summarise_corrections


def _prepare_audio(perf_midi: Path, dest: Path) -> bool:
    import pretty_midi, soundfile as sf, warnings
    from app.services.sampler import render_notes
    if dest.exists():
        return True
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pm = pretty_midi.PrettyMIDI(str(perf_midi))
    except Exception:
        return False
    notes = [{"pitch": int(n.pitch), "onset": float(n.start),
              "offset": float(n.end), "velocity": int(n.velocity)}
             for i in pm.instruments if not i.is_drum for n in i.notes]
    if not notes:
        return False
    audio = render_notes(notes, sr=22050)
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dest), audio, 22050, subtype="PCM_16")
    return True


def _run_pipeline(session_dir: Path) -> Optional[List[Dict]]:
    from app.services.separation    import separate_components
    from app.services.transcription import transcribe_components
    from app.services.segmentation  import analyse_audio
    from app.routers.segment        import _apply_original_cleanup
    try:
        separate_components(session_dir / "audio.wav", session_dir)
        notes, feel = transcribe_components(session_dir)
        json.dump({"session_id": session_dir.name, "notes": notes,
                   "timing_feel": feel},
                  open(session_dir / "transcription.json", "w", encoding="utf-8"))
        seg = analyse_audio(session_dir / "audio.wav", session_dir)
        try:
            _apply_original_cleanup(session_dir, seg)
        except Exception:
            pass
        return json.load(open(session_dir / "transcription.json",
                              encoding="utf-8"))["notes"]
    except Exception as e:
        print(f"    pipeline failed: {type(e).__name__}: {e}")
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--report-only", action="store_true",
                    help="Re-report from pairs already on disk")
    args = ap.parse_args()

    targets = {r["folder"]: r for r in
               (json.loads(l) for l in
                open(OUT_DIR / "score_targets.jsonl", encoding="utf-8"))}
    manifest = json.load(open(OUT_DIR / "train_manifest.json", encoding="utf-8"))
    perfs = [p for p in manifest["performances"] if p["folder"] in targets]

    # One performance per score, for breadth over depth
    seen, chosen = set(), []
    for p in perfs:
        if p["folder"] in seen:
            continue
        seen.add(p["folder"])
        chosen.append(p)
        if len(chosen) >= args.limit:
            break

    pairs_path = OUT_DIR / "pairs.jsonl"
    if not args.report_only:
        WORK_DIR.mkdir(parents=True, exist_ok=True)
        rows = {r["folder"]: r for r in
                csv.DictReader(open(ASAP_DIR / "metadata.csv", encoding="utf-8"))}
        out = open(pairs_path, "w", encoding="utf-8")
        t0 = time.time()
        for i, p in enumerate(chosen, 1):
            folder = p["folder"]
            meta = rows.get(folder)
            if meta is None:
                continue
            sess = WORK_DIR / folder.replace("/", "_")
            sess.mkdir(parents=True, exist_ok=True)
            print(f"[{i}/{len(chosen)}] {folder}")
            if not _prepare_audio(ASAP_DIR / p["perf_midi"], sess / "audio.wav"):
                print("    no audio"); continue
            ours = _run_pipeline(sess)
            if not ours:
                continue
            try:
                bm = BeatMap.for_performance(
                    ASAP_DIR / folder,
                    Path(meta["performance_annotations"]).name)
            except Exception as e:
                print(f"    no beat map: {e}"); continue
            res = align_notes(ours, targets[folder]["notes"], bm)
            res["folder"] = folder
            res["corrections"] = summarise_corrections(res)
            out.write(json.dumps(res) + "\n")
            s = res["stats"]
            print(f"    matched {s['n_matched']}/{s['n_score']} "
                  f"P={s['precision']:.3f} R={s['recall']:.3f} "
                  f"| dur x{res['corrections'].get('duration_ratio_median')}")
            # Working audio/stems are large; keep only the transcription
            for junk in ("components", "audio.wav"):
                q = sess / junk
                if q.is_dir():
                    shutil.rmtree(q, ignore_errors=True)
                elif q.exists():
                    q.unlink()
        out.close()
        print(f"\nbuilt in {time.time()-t0:.0f}s -> "
              f"{pairs_path.relative_to(REPO_ROOT)}")

    # ── report ────────────────────────────────────────────────────────────
    rows = [json.loads(l) for l in open(pairs_path, encoding="utf-8")]
    if not rows:
        return
    tot = {k: sum(r["stats"][k] for r in rows)
           for k in ("n_ours", "n_score", "n_matched", "n_spurious", "n_missed")}
    print()
    print(f"=== correction training set: {len(rows)} pieces ===")
    print(f"our notes {tot['n_ours']:,} | score notes {tot['n_score']:,}")
    print(f"matched {tot['n_matched']:,} "
          f"(P={tot['n_matched']/max(1,tot['n_ours']):.3f} "
          f"R={tot['n_matched']/max(1,tot['n_score']):.3f})")
    print(f"spurious (model must DELETE): {tot['n_spurious']:,}")
    print(f"missed   (model must INSERT): {tot['n_missed']:,}")

    m = [x for r in rows for x in r["matched"]]
    if m:
        dr = np.array([x["dur_ratio"] for x in m])
        oe = np.array([x["onset_err_b"] for x in m])
        print()
        print("systematic corrections the model would learn:")
        print(f"  duration ratio  median {np.median(dr):.3f} "
              f"(1.0 = correct; <1 = we cut notes short)")
        print(f"  too short (<0.8x): {100*np.mean(dr<0.8):.1f}%   "
              f"too long (>1.25x): {100*np.mean(dr>1.25):.1f}%")
        print(f"  onset bias {np.median(oe):+.4f} beats, "
              f"spread {np.std(oe):.4f}")
        from collections import Counter
        hands = Counter(x["tgt_hand"] for x in m)
        print(f"  hand labels available: {dict(hands)}")


if __name__ == "__main__":
    main()
