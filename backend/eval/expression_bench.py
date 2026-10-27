"""Dynamics and sustain (app/services/expression.py) measured on renders with real ground truth.

MAESTRO MIDI carries what a pop song never gives us: pianists' own velocities and their sustain pedal (CC64).  Each
clip is a 30 s window rendered through the app's piano (sfizz + Salamander) WITH that pedal, so the audio rings the
way the pianist pedalled it; then the app's song setup reads it (base at the 0.8 bar + clean-up v2 keep 0.4, the
hosted configuration) and expression.py adds velocities and a pedal track.

  fit   MAESTRO validation pieces: the dynamics coefficients (least squares on notes matched to the truth) and the
        sustain thresholds (grid, best mean frame F1)
  test  MAESTRO test pieces, never used to fit; scored with eval/velocity_metrics.py (spread ratio, Pearson, MAE,
        soft-note bias) and eval/pedal_metrics.py (frame F1, IoU, down rates, the always-down F1)

    python eval/expression_bench.py --ckpt <base best.pt> --cleanup <cleanup2 best.pt> --out <dir> [--n-fit 30 --n-test 40]
"""
import argparse
import json
import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))


def window(pm, t0, secs):
    """Notes (key presses / releases, velocities) and pedal intervals of [t0, t0 + secs), times from 0."""
    notes = [{"pitch": n.pitch, "onset": n.start - t0, "offset": min(secs, n.end - t0), "velocity": n.velocity}
             for inst in pm.instruments for n in inst.notes if t0 <= n.start < t0 + secs]
    cc = sorted((c.time, c.value) for inst in pm.instruments for c in inst.control_changes if c.number == 64)
    ped, down = [], None
    for t, v in cc:
        if v >= 64 and down is None:
            down = t
        elif v < 64 and down is not None:
            ped.append((down, t))
            down = None
    if down is not None:
        ped.append((down, t0 + secs + 5))
    pedal = [{"onset": max(0.0, a - t0), "offset": min(secs, b - t0)} for a, b in ped if b > t0 and a < t0 + secs]
    return notes, pedal


def matched(ours, truth):
    """(truth velocities, row index in ours) of notes matched by onset (50 ms) and pitch."""
    import librosa
    import mir_eval
    if not ours or not truth:
        return np.zeros(0), np.zeros(0, int)
    m = mir_eval.transcription.match_notes(
        np.array([[n["onset"], max(n["offset"], n["onset"] + 1e-3)] for n in truth]),
        np.array([librosa.midi_to_hz(n["pitch"]) for n in truth]),
        np.array([[n["onset"], max(n["offset"], n["onset"] + 1e-3)] for n in ours]),
        np.array([librosa.midi_to_hz(n["pitch"]) for n in ours]),
        onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None)
    return np.array([float(truth[r]["velocity"]) for r, _ in m]), np.array([e for _, e in m], int)


RING = [9.0, 12.0, 16.0, 20.0, 25.0, 30.0, 40.0]
JAC = [0.0, 0.3, 0.5, 0.7]                   # 0 = the pedal never changes inside a region


def lifts(pred, truth, secs, tol=0.1):
    """The pedal LIFTS (re-pedalling at a change), which frame F1 barely sees: recall = share of the pianist's lifts
    inside the clip that ours also makes within tol; precision = share of our lifts the pianist also makes."""
    up = lambda ps: [p["offset"] for p in ps if 0.2 < p["offset"] < secs - 0.2]
    t, o = up(truth), up(pred)
    rec = np.mean([any(abs(x - y) <= tol for y in o) for x in t]) if t else np.nan
    pre = np.mean([any(abs(x - y) <= tol for y in t) for x in o]) if o else np.nan
    return rec, pre, len(t), len(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cleanup", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-fit", type=int, default=30)
    ap.add_argument("--n-test", type=int, default=40)
    ap.add_argument("--secs", type=float, default=30.0)
    ap.add_argument("--thr", type=float, default=0.8)
    a = ap.parse_args()
    os.environ["ARRANGEMENT_CLEANUP"] = a.cleanup
    os.environ["ARRANGEMENT_CLEANUP_KEEP"] = "0.4"
    import pretty_midi
    import soundfile as sf
    from app.ml.spectral import render as R
    from app.services import expression as X
    from app.services import sfz_sampler as Z
    from app.services import spectral_transcription as ST
    from velocity_metrics import velocity_scores
    from pedal_metrics import pedal_scores
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    clips = []
    for split, n in (("validation", a.n_fit), ("test", a.n_test)):
        rows = R.pieces(split)
        for row in [rows[i] for i in rng.choice(len(rows), min(n, len(rows)), replace=False)]:
            pm = pretty_midi.PrettyMIDI(str(R.MAESTRO / row["midi_filename"]))
            end = pm.get_end_time()
            if end < a.secs + 20:
                continue
            t0 = float(rng.uniform(10, end - a.secs - 5))
            truth, pedal = window(pm, t0, a.secs)
            if len(truth) < 40:
                continue
            y = Z.render_notes_sfz(truth, sr=X.SR, pedal=pedal).astype(np.float32)[: int(a.secs * X.SR)]
            wav = out / "clip.wav"
            sf.write(wav, y, X.SR)
            est = ST.transcribe(wav, onset_thr=a.thr, ckpt=pathlib.Path(a.ckpt), frame_thr=0.5)
            C = X.cqt_db(y)
            F = X.features(est, y, C) if est else np.zeros((0, 3))
            tv, ix = matched(est, truth)
            peds = {f"{r}/{j}": X.sustain_pedal(est, y, C, ring_db=r, change_jaccard=j) for r in RING for j in JAC}
            clips.append(dict(split=split, piece=row["canonical_title"], t0=t0, truth=truth, pedal=pedal, est=est,
                              F=F.tolist(), tv=tv.tolist(), ix=ix.tolist(), peds=peds))
            print(f"{split:10s} {row['canonical_title'][:40]:40s} truth {len(truth):4d} est {len(est):4d} "
                  f"matched {len(ix):4d}  pedal down {sum(p['offset'] - p['onset'] for p in pedal):5.1f} s", flush=True)
    (out / "clips.json").write_text(json.dumps(clips))

    fit = [c for c in clips if c["split"] == "validation"]
    test = [c for c in clips if c["split"] == "test"]

    # dynamics: least squares on the fit clips' matched notes
    def design(c, cols):
        F = np.array(c["F"])[c["ix"]] if c["ix"] else np.zeros((0, 3))
        return np.c_[np.ones(len(F)), F[:, cols]]
    fits = {}
    for name, cols in (("spread (base only)", [0]), ("full (base + key + song)", [0, 1, 2])):
        Xm = np.concatenate([design(c, cols) for c in fit])
        yv = np.concatenate([np.array(c["tv"]) for c in fit])
        w, *_ = np.linalg.lstsq(Xm, yv, rcond=None)
        coef = {"C0": float(w[0]), "C_BASE": 0.0, "C_KEY": 0.0, "C_SONG": 0.0}
        for i, k in enumerate(cols):
            coef[("C_BASE", "C_KEY", "C_SONG")[k]] = float(w[1 + i])
        fits[name] = coef
    print("\nfitted on", len(fit), "validation clips:")
    for name, coef in fits.items():
        print(f"   {name:26s} " + "  ".join(f"{k} {v:+.3f}" for k, v in coef.items()))

    def vel_rows(transform):
        rows = []
        for c in test:
            ours = transform(c)
            s = velocity_scores(ours, c["truth"])
            if s:
                rows.append(s)
        return {k: float(np.mean([r[k] for r in rows])) for k in ("spread_ratio", "pearson", "mae", "soft_bias",
                                                                    "mean_ours", "mean_truth", "std_ours", "std_truth")}
    def with_coef(coef):
        def f(c):
            F = np.array(c["F"])
            v = coef["C0"] + coef["C_BASE"] * F[:, 0] + coef["C_KEY"] * F[:, 1] + coef["C_SONG"] * F[:, 2] if len(F) else []
            return [dict(g, velocity=float(np.clip(x, X.VEL_LO, X.VEL_HI))) for g, x in zip(c["est"], v)]
        return f
    vres = {"base as it is": vel_rows(lambda c: c["est"])}
    for name, coef in fits.items():
        vres[name] = vel_rows(with_coef(coef))
    print(f"\nVELOCITY on {len(test)} test clips (notes matched to the pianist's, 50 ms)")
    print(f"   {'':28s}{'spread':>8s}{'pearson':>9s}{'MAE':>7s}{'soft bias':>11s}{'mean':>7s}{'std':>6s}")
    for name, r in vres.items():
        print(f"   {name:28s}{r['spread_ratio']:>8.2f}{r['pearson']:>9.3f}{r['mae']:>7.1f}{r['soft_bias']:>+11.1f}"
              f"{r['mean_ours']:>7.1f}{r['std_ours']:>6.1f}")
    print(f"   {'pianist (truth)':28s}{1.0:>8.2f}{1.0:>9.3f}{0.0:>7.1f}{0.0:>+11.1f}{vres['base as it is']['mean_truth']:>7.1f}"
          f"{vres['base as it is']['std_truth']:>6.1f}")

    # sustain: thresholds by mean F1 on the fit clips, scored on test
    def ped_mean(cs, key):
        rows = [pedal_scores(c["peds"][key] if key else [], c["pedal"], a.secs) for c in cs]
        r = {k: float(np.mean([x[k] for x in rows])) for k in ("precision", "recall", "f1", "iou", "down_rate_pred",
                                                                 "down_rate_truth", "f1_always_down")}
        L = [lifts(c["peds"][key] if key else [], c["pedal"], a.secs) for c in cs]
        r["lift_recall"] = float(np.nanmean([x[0] for x in L])) if L else float("nan")
        r["lift_precision"] = float(np.nanmean([x[1] for x in L])) if any(x[3] for x in L) else float("nan")
        r["lifts_truth"], r["lifts_ours"] = int(sum(x[2] for x in L)), int(sum(x[3] for x in L))
        return r
    keys = list(fit[0]["peds"]) if fit else []
    best = max(keys, key=lambda k: ped_mean(fit, k)["f1"]) if keys else None
    best_nochange = max([k for k in keys if k.endswith("/0.0")], key=lambda k: ped_mean(fit, k)["f1"]) if keys else None
    pres = {"no pedal (today)": ped_mean(test, None),
            f"ringing only, no changes ({best_nochange})": ped_mean(test, best_nochange),
            f"ringing + harmony changes ({best})": ped_mean(test, best)}
    print(f"\nSUSTAIN on {len(test)} test clips (10 ms frames vs the pianist's pedal); ring dB / change Jaccard "
          f"chosen on validation")
    print(f"   {'':44s}{'F1':>6s}{'IoU':>6s}{'P':>6s}{'R':>6s}{'down ours':>11s}{'down truth':>12s}"
          f"{'lift R':>8s}{'lift P':>8s}{'lifts ours/truth':>18s}")
    for name, r in pres.items():
        print(f"   {name:44s}{r['f1']:>6.3f}{r['iou']:>6.3f}{r['precision']:>6.3f}{r['recall']:>6.3f}"
              f"{r['down_rate_pred']:>11.2f}{r['down_rate_truth']:>12.2f}{r['lift_recall']:>8.2f}{r['lift_precision']:>8.2f}"
              f"{r['lifts_ours']:>9d}/{r['lifts_truth']}")
    hc = {k: ped_mean(fit, k) for k in keys if not k.endswith("/0.0")}
    if hc:                                   # the best harmony-changing setting by lift F1 on validation, for reference
        lf = lambda r: 2 * r["lift_recall"] * r["lift_precision"] / max(1e-9, r["lift_recall"] + r["lift_precision"])
        bl = max(hc, key=lambda k: lf(hc[k]) + hc[k]["f1"])
        r = ped_mean(test, bl)
        pres[f"ringing + changes, best F1 + lift F1 ({bl})"] = r
        print(f"   {'ringing + changes, best F1+lift F1 (' + bl + ')':44s}{r['f1']:>6.3f}{r['iou']:>6.3f}{r['precision']:>6.3f}"
              f"{r['recall']:>6.3f}{r['down_rate_pred']:>11.2f}{r['down_rate_truth']:>12.2f}{r['lift_recall']:>8.2f}"
              f"{r['lift_precision']:>8.2f}{r['lifts_ours']:>9d}/{r['lifts_truth']}")
    print(f"   {'always down (reference)':44s}{pres['no pedal (today)']['f1_always_down']:>6.3f}")
    (out / "summary.json").write_text(json.dumps(dict(fits=fits, velocity=vres, sustain=pres, best=best,
                                                      best_nochange=best_nochange), indent=1))
    print(f"\nwrote {out / 'summary.json'}")


if __name__ == "__main__":
    main()
