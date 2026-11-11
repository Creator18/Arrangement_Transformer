"""The real-song test at the HOSTED APP's settings (user, 2026-11-04: re-run eval/s2p_evaluate.py the way the app runs).

s2p_evaluate.py tunes each system's onset threshold on validation and skips the clean-up, so it does not show what the
app plays (it picked 0.90 for base20m and 0.95 for run 3; the app uses 0.8 for both).  Here every test song goes
through the app's own note path, app/services/spectral_transcription.transcribe(): cond 2, the fixed onset / frame
thresholds, and the clean-up model from ARRANGEMENT_CLEANUP* (read from the environment, as in the app).  Each
--ckpt gives two rows: "<label> app" (with the clean-up) and "<label> raw" (the same thresholds, no clean-up).
Sustain only adds pedal, not notes, so it changes nothing here.

    ARRANGEMENT_CLEANUP=.../cleanup4_nobeat/best.pt ARRANGEMENT_CLEANUP_KEEP=0.3 ARRANGEMENT_CLEANUP_MERGE=0.5 \\
    ARRANGEMENT_CLEANUP_NK=0.8 python eval/s2p_app_eval.py --m2m /scratch/.../m2m --units .../m2m_units.json \\
        --exclude-old-train --ckpt base=<base20m> --ckpt r3=<run 3> --thr 0.8 --frame-thr 0.5 --out results.json
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
from s2p_evaluate import CARD, LEVELS, _test_job    # noqa: E402


def main():
    import multiprocessing as mp
    pool = mp.Pool(7)                       # before CUDA is touched: workers fork clean
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2m", required=True)
    ap.add_argument("--ckpt", required=True, action="append", help="label=path; repeat to compare")
    ap.add_argument("--thr", type=float, default=0.8, help="onset threshold (ARRANGEMENT_SONG_ARRANGER_THR)")
    ap.add_argument("--frame-thr", type=float, default=0.5, help="ARRANGEMENT_SONG_ARRANGER_FRAME_THR")
    ap.add_argument("--units", default="")
    ap.add_argument("--exclude-old-train", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    from app.services import cleanup
    from app.services import spectral_transcription as ST
    if not cleanup.enabled():
        sys.exit("ARRANGEMENT_CLEANUP is not set: the app rows need the clean-up model")
    clean_ckpt = cleanup.CKPT
    print(f"thr {args.thr}, frame thr {args.frame_thr}, clean-up {clean_ckpt} "
          f"(keep {cleanup.KEEP}, merge {cleanup.MERGE}, never-kept {cleanup.NK})", flush=True)

    m2m = pathlib.Path(args.m2m)
    npy = m2m / "midi_numpy" if (m2m / "midi_numpy").is_dir() else m2m / "music2midi-dataset" / "midi_numpy"
    sp = json.loads((m2m / "m2m_splits.json").read_text())
    if args.units:
        U = json.loads(pathlib.Path(args.units).read_text())
        old_train = set(sp["train"])
        sp = dict(U["splits"], difficulty=U["difficulty"])
        if args.exclude_old_train:
            sp["test"] = [i for i in sp["test"] if i not in old_train]
    ranges = json.loads((ROOT / "app" / "ml" / "scorecard_ranges.json").read_text())
    level = sp["difficulty"]

    def notes_of(wav, path, with_cleanup):
        cleanup.CKPT = clean_ckpt if with_cleanup else ""
        ns = ST.transcribe(wav, onset_thr=args.thr, ckpt=pathlib.Path(path), cond=2, frame_thr=args.frame_thr)
        return np.array([[n["onset"], n["offset"], n["pitch"], n["velocity"]] for n in ns]) if ns else np.zeros((0, 4))

    te = sp["test"][: args.limit or None]
    res = collections.defaultdict(list)
    pend = []
    for sid in te:
        wav = m2m / "audio" / f"{sid}.wav"
        ref = np.load(npy / f"{sid}.npy")
        song = dict(np.load(m2m / "scorecard" / f"song_{sid}.npz"))
        lv = level[sid]
        for spec in args.ckpt:
            label, path = spec.split("=", 1)
            for tag, wc in (("app", True), ("raw", False)):
                est = notes_of(wav, path, wc)
                pend.append((f"{label} {tag}", pool.apply_async(_score, ((est, ref, song, lv, ranges),))))
        pend.append(("human (reference)", pool.apply_async(_test_job,
                     (("human (reference)", "human", None, 0, ref, song, lv, lv, ranges),))))
        print(f"  test {sid} {lv}", flush=True)
    for name, job in pend:
        out = job.get()
        res[name].append(out[1] if isinstance(out, tuple) else out)

    m = lambda rows, k: float(np.nanmean([r[k] for r in rows])) if rows else float("nan")
    print(f"\nTEST at the app's settings: {len(te)} songs\n")
    for lv in LEVELS + ["all"]:
        print(f"-- {lv} --")
        print(f"   {'system':<18}{'strict':>8}{'P':>7}{'R':>7}{'tol':>7}{'pc':>7}{'in-range':>10}"
              + "".join(f"{k[:9]:>10}" for k in CARD))
        for n in list(res):
            rows = [r for r in res[n] if lv == "all" or r["level"] == lv]
            print(f"   {n:<18}{m(rows, 'f1'):>8.3f}{m(rows, 'p'):>7.3f}{m(rows, 'r'):>7.3f}"
                  f"{m(rows, 'tol'):>7.3f}{m(rows, 'pc'):>7.3f}{m(rows, 'inside'):>10.2f}"
                  + "".join(f"{m(rows, k):>10.2f}" for k in CARD))
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(dict(thr=args.thr, frame_thr=args.frame_thr,
                                                          cleanup=str(clean_ckpt), results=res), indent=1))
        print(f"\nwrote {args.out}")


def _score(a):
    """Worker: similarity scores and the scorecard (target: advanced, what the app asks for) for given notes."""
    from s2p_evaluate import f1s
    from scorecard import score_arrangement
    est, ref, song, lv, ranges = a
    f, p, r, ft, fp = f1s(ref, est)
    card = score_arrangement(est, song) if len(est) > 50 else {k: np.nan for k in CARD}
    inside = [ranges[k]["advanced"]["p10"] <= card[k] <= ranges[k]["advanced"]["p90"] for k in CARD
              if np.isfinite(card.get(k, np.nan))]
    return dict(level=lv, f1=f, p=p, r=r, tol=ft, pc=fp, inside=float(np.mean(inside)) if inside else 0.0,
                **{k: card.get(k, np.nan) for k in CARD})


if __name__ == "__main__":
    main()
