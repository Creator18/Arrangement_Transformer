"""How much does the app's post-processing move note timing?

The user heard trills and close notes slightly off against the original, while
the model's own onsets are within 5.8 ms (median) of the truth.  After the
transcriber, the piano path runs refine_notes -> dedupe_octave_partials ->
_beat_adaptive_quantize (16th-note grid from detected beats, snaps up to
45 ms) — and 45 ms stays inside the 50 ms F1 tolerance, so F1 never sees it.

Scores each stage's onsets against the unfiltered MIDI: F1 at 50 ms and at a
tight 20 ms, onset error, and how often the grid moved a note audibly.
Beats from madmom when installed (as the app does for solo piano), else librosa.

    python eval/post_timing_check.py --dir E:/arrangement_transformer/maestro_real
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
from real_audio_eval import midi_truth, arrays       # noqa: E402


def beats_for(wav):
    try:
        from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
        return np.array(DBNBeatTrackingProcessor(fps=100)(RNNBeatProcessor()(str(wav)))), "madmom"
    except Exception:                                       # noqa: BLE001
        import librosa
        y, sr = librosa.load(str(wav), sr=22050, mono=True)
        t, b = librosa.beat.beat_track(y=y, sr=sr)
        return librosa.frames_to_time(b, sr=sr), "librosa"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    from mir_eval.transcription import precision_recall_f1_overlap as prf, match_notes
    from app.services import spectral_transcription as ST
    from app.services import transcription as TR

    stats = {k: {"f50": [], "f20": [], "err": []} for k in ("model", "refine+dedupe", "grid snap")}
    moved = []
    src = set()
    for wav in sorted(pathlib.Path(args.dir).glob("*.wav")):
        ref = midi_truth(wav.with_suffix(".midi"))
        notes = ST.transcribe(wav)
        beats, s = beats_for(wav); src.add(s)
        tempo = 60.0 / np.median(np.diff(beats)) if len(beats) > 1 else 120.0
        r1 = TR.dedupe_octave_partials(TR.refine_notes(sorted(notes, key=lambda n: (n["onset"], n["pitch"])), tempo))
        r2 = TR._beat_adaptive_quantize([dict(n) for n in r1], beats, subdivisions=4)
        moved += [abs(b["onset"] - a["onset"]) for a, b in zip(r1, r2)]
        ri, rp = arrays(ref)
        for k, est in (("model", notes), ("refine+dedupe", r1), ("grid snap", r2)):
            ei, ep = arrays(est)
            stats[k]["f50"].append(prf(ri, rp, ei, ep, onset_tolerance=0.05, offset_ratio=None)[2])
            stats[k]["f20"].append(prf(ri, rp, ei, ep, onset_tolerance=0.02, offset_ratio=None)[2])
            stats[k]["err"] += [abs(ei[j, 0] - ri[i, 0]) for i, j in
                                match_notes(ri, rp, ei, ep, onset_tolerance=0.05, offset_ratio=None)]
        print(f"  {wav.stem[:50]}  tempo {tempo:.0f}", flush=True)

    print(f"\nbeats from {', '.join(src)}")
    print(f"{'stage':<16}{'F1@50ms':>9}{'F1@20ms':>9}{'median err':>12}{'p90 err':>9}{'>20ms':>8}")
    for k, s in stats.items():
        e = np.array(s["err"]) * 1000
        print(f"{k:<16}{np.mean(s['f50']):>9.3f}{np.mean(s['f20']):>9.3f}{np.median(e):>10.1f}ms"
              f"{np.percentile(e, 90):>7.1f}ms{np.mean(e > 20):>8.1%}")
    m = np.array(moved) * 1000
    print(f"\ngrid snap moved: {np.mean(m > 1):.1%} of notes;  >10 ms {np.mean(m > 10):.1%}  "
          f">20 ms {np.mean(m > 20):.1%}  >30 ms {np.mean(m > 30):.1%}  (median move of moved "
          f"{np.median(m[m > 1]):.1f} ms)")


if __name__ == "__main__":
    main()
