"""Onset bar 0.8 vs 0.85 vs 0.9 for the base, and what the notes a lower bar ADDS are (the user, 2026-10-27: "0.8
has all the notes I want, including vocals; clean it up instead of the 0.9 gate").

The app's decoding (spectral_transcription.transcribe: short notes and re-strikes kept, 60 ms fragment merge, centroid
starts), run once per file and decoded at every bar.  Two kinds of test:

  renders  (--clips: held-out npz renders, exact written labels): recall / precision / F1, double presses (a strike on
           a key that is already held, no written start there), notes per second, and recall of SUNG notes (mod_combo
           renders mark them).  Then every note the low bar adds over 0.9 (no 0.9 note on that key within 48 ms) is
           sorted: right (a written start), double press, octave / harmonic of a sounding note (+-12, +19, +24, +28
           above it, or 12 below), semitone / tone neighbour, else other; and the share right per peak onset
           probability, length, loudness and "the key is already held in the 0.9 output" -- what a cleanup could use.
  real     (--real: the 106 held-out Music2MIDI songs): the human arrangement is only one way to play the song, so
           precision is a floor; strict (pitch, 50 ms) and pitch-class (150 ms) F1, notes per second, recall of the
           SUNG line (the arrangement's top line where Demucs' vocal stem sounds), and the same sorting of the added
           notes against the human arrangement and the 0.9 output.

    python eval/thr_review.py --ckpt <base best.pt> --clips <dir> --n 300 --out <json>
    python eval/thr_review.py --ckpt <base best.pt> --real --out <json> [--shard k --nshards n]  |  --merge <jsons>
"""
import argparse
import json
import pathlib
import random
import sys

import numpy as np
import soundfile as sf

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.services import spectral_transcription as ST  # noqa: E402

THRS = (0.8, 0.85, 0.9)
LOW = (0.8, 0.85)
SR = 24000
KINDS = ("right", "double", "harmonic", "neighbour", "other")
PEAK = [0.8, 0.85, 0.9, 0.95, 1.01]
DUR = [0, 0.05, 0.1, 0.2, 0.5, 99]
VEL = [0, 40, 60, 80, 128]


def transcribe_all(wav):
    """notes at every bar from ONE model run, plus the onset roll (peak probability per added note)."""
    cache = {}
    real = ST.rolls

    def rolls(path, cond=None, with_offset=False):
        if "r" not in cache:
            cache["r"] = real(path, cond=cond, with_offset=with_offset)
        return cache["r"]
    ST.rolls = rolls
    try:
        out = {t: ST.transcribe(wav, onset_thr=t, ckpt=pathlib.Path(A.ckpt), frame_thr=0.5) for t in THRS}
    finally:
        ST.rolls = real
    return out, cache["r"][0]


def added(lo, hi):
    """notes in `lo` with no `hi` note on the same key within 48 ms."""
    by = {}
    for g in hi:
        by.setdefault(g["pitch"], []).append(g["onset"])
    return [g for g in lo if not any(abs(o - g["onset"]) <= 0.048 for o in by.get(g["pitch"], []))]


def held(notes, p, t):
    return any(g["pitch"] == p and g["onset"] < t - 0.03 and g["offset"] > t for g in notes)


def kind(g, starts, sounding):
    """starts(p, t, tol) -> bool written start;  sounding(p, t) -> bool a written note sounds."""
    p, t = g["pitch"], g["onset"]
    if starts(p, t, 0.05):
        return "right"
    if sounding(p, t):
        return "double"
    if any(sounding(p - d, t) for d in (12, 19, 24, 28)) or sounding(p + 12, t):
        return "harmonic"
    if any(sounding(p + d, t) for d in (-2, -1, 1, 2)):
        return "neighbour"
    return "other"


def features(g, on, hi_notes):
    k, f = g["pitch"] - 21, int(round(g["onset"] * 62.5))
    pk = float(on[k, max(0, f - 2):f + 3].max()) if 0 <= k < 88 else 0.0
    return [pk, g["offset"] - g["onset"], g["velocity"], int(held(hi_notes, g["pitch"], g["onset"]))]


def prf(n_hit_r, n_true, n_hit_p, n_pred):
    r, p = n_hit_r / max(1, n_true), n_hit_p / max(1, n_pred)
    return r, p, 2 * r * p / max(1e-9, r + p)


def match(pred, gt, tol, pc=False):
    """(gt notes hit, pred notes hit): greedy by distance, one-to-one.  pred / gt: lists of (t, pitch)."""
    key = (lambda p: p % 12) if pc else (lambda p: p)
    by = {}
    for i, (t, p) in enumerate(pred):
        by.setdefault(key(p), []).append((t, i))
    used, hits = set(), 0
    for t, p in gt:
        c = [(abs(tt - t), i) for tt, i in by.get(key(p), []) if abs(tt - t) <= tol and i not in used]
        if c:
            used.add(min(c)[1])
            hits += 1
    return hits, len(used)


def renders():
    from app.ml.spectral import frontend as S
    files = sorted(p for p in pathlib.Path(A.clips).glob("*.npz") if ".tmp" not in p.name)
    files = random.Random(0).sample(files, min(A.n, len(files)))
    R = {t: dict(hit=0, true=0, pred=0, dbl=0, secs=0.0, sung_hit=0, sung=0) for t in THRS}
    ADD = {t: [] for t in LOW}
    tmp = pathlib.Path(A.out).with_suffix(".tmp.wav")
    for f in files:
        z = np.load(f)
        y = z["y"].astype(np.float32) / 32767
        sf.write(tmp, y, S.SR)
        notes, on = transcribe_all(tmp)
        won, wfr = z["on"] > 0, z["fr"] > 0
        T = won.shape[1]
        gt = [(fr_ / S.FPS, k + 21) for k, fr_ in zip(*np.nonzero(won))]
        meta = z["note_meta"]
        sung = [(float(m[0]), int(m[1])) for m in meta if meta.shape[1] >= 9 and m[8] > 0.5] if str(z["piano"]) == "multi" else []

        def starts(p, t, tol):
            k, f0 = p - 21, int(round(t * S.FPS))
            return 0 <= k < 88 and won[k, max(0, f0 - int(tol * S.FPS)):f0 + int(tol * S.FPS) + 1].any()

        def sounding(p, t):
            k, f0 = p - 21, int(round(t * S.FPS))
            return 0 <= k < 88 and 0 <= f0 < T and bool(wfr[k, f0] or won[k, f0])
        for t in THRS:
            pred = [(g["onset"], g["pitch"]) for g in notes[t] if g["onset"] < T / S.FPS]
            h, hp = match(pred, gt, 0.05)
            R[t]["hit"] += h; R[t]["true"] += len(gt); R[t]["pred"] += len(pred); R[t]["secs"] += T / S.FPS
            R[t]["dbl"] += sum(1 for g in notes[t] if kind(g, starts, sounding) == "double")
            if sung:
                R[t]["sung_hit"] += match(pred, sung, 0.05)[0]; R[t]["sung"] += len(sung)
        for t in LOW:
            ADD[t] += [[kind(g, starts, sounding)] + features(g, on, notes[0.9]) for g in added(notes[t], notes[0.9])
                       if g["onset"] < T / S.FPS]
        print(f"{f.name}: {len(notes[0.8])} / {len(notes[0.85])} / {len(notes[0.9])} notes", flush=True)
    tmp.unlink(missing_ok=True)
    return dict(kind="renders", clips=str(A.clips), n=len(files), R=R, ADD=ADD)


def real():
    sys.path.insert(0, str(ROOT / "eval"))
    from rhythm_groups import top_voice
    m2m = pathlib.Path("/scratch/basu.ang/m2m")
    U = json.loads((m2m / "m2m_units.json").read_text())
    old = set(json.loads((m2m / "m2m_splits.json").read_text())["train"])
    test = [i for i in U["splits"]["test"] if i not in old][A.shard::A.nshards]
    R = {t: dict(s_hit=0, s_true=0, s_hp=0, pc_hit=0, pc_hp=0, pred=0, secs=0.0, sung_hit=0, sung=0) for t in THRS}
    ADD = {t: [] for t in LOW}
    tmp = pathlib.Path(A.out).with_suffix(".tmp.wav")
    for sid in test:
        y = np.load(f"/scratch/basu.ang/solo_det/m2m/{sid}.npy").astype(np.float32)
        sf.write(tmp, y, SR)
        notes, on = transcribe_all(tmp)
        hum = np.load(m2m / "midi_numpy" / f"{sid}.npy")
        hum = hum[np.lexsort((-hum[:, 2], hum[:, 0]))]
        gt = [(float(t), int(p)) for t, _, p, *_ in hum]
        sung = []
        zf = pathlib.Path(f"/scratch/basu.ang/solo_det/m2m_sim/{sid}.npz")
        if zf.exists():
            z = np.load(zf)
            hop, vdb = float(z["hop"]), z["vocals_db"].astype(float)
            voiced = (vdb > np.percentile(vdb, 99.5) - 35) & (vdb > -60)
            for t, _, p in hum[top_voice(hum)][:, :3]:
                i0, i1 = max(0, int((t - 0.1) / hop)), int((t + 0.1) / hop) + 1
                if i0 < len(voiced) and voiced[i0:i1].any():
                    sung.append((float(t), int(p)))
        ons = np.array([n[0] for n in gt]); offs = hum[:, 1]; pit = hum[:, 2].astype(int)

        def starts(p, t, tol):
            return bool(np.any((np.abs(ons - t) <= 0.15) & (pit % 12 == p % 12)))

        def sounding(p, t):
            return bool(np.any((ons < t - 0.03) & (offs > t) & (pit == p)))
        secs = len(y) / SR
        for t in THRS:
            pred = [(g["onset"], g["pitch"]) for g in notes[t]]
            h, hp = match(pred, gt, 0.05)
            hpc, hppc = match(pred, gt, 0.15, pc=True)
            r = R[t]
            r["s_hit"] += h; r["s_hp"] += hp; r["s_true"] += len(gt); r["pc_hit"] += hpc; r["pc_hp"] += hppc
            r["pred"] += len(pred); r["secs"] += secs
            r["sung_hit"] += match(pred, sung, 0.15, pc=True)[0]; r["sung"] += len(sung)
        for t in LOW:
            sung_a = np.array([s[0] for s in sung]); sung_p = np.array([s[1] for s in sung])
            for g in added(notes[t], notes[0.9]):
                k = kind(g, starts, sounding)
                is_sung = bool(len(sung_a) and np.any((np.abs(sung_a - g["onset"]) <= 0.15) & (sung_p % 12 == g["pitch"] % 12)))
                ADD[t].append([k, *features(g, on, notes[0.9]), int(is_sung),
                               int(any(abs(h_["onset"] - g["onset"]) < 0.5 and h_["pitch"] - g["pitch"] in (-12, -19, -24, 12)
                                       and h_["onset"] <= g["onset"] < h_["offset"] for h_ in notes[0.9]))])
        print(f"{sid}: {len(notes[0.8])} / {len(notes[0.85])} / {len(notes[0.9])} notes, {len(sung)} sung", flush=True)
    tmp.unlink(missing_ok=True)
    return dict(kind="real", n=len(test), R=R, ADD=ADD)


def bins(rows, col, edges):
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        s = [r for r in rows if a <= r[col] < b]
        out.append((f"{a:g}-{b:g}", len(s), sum(r[0] == "right" for r in s) / max(1, len(s))))
    return out


def report(D):
    R = {float(k): v for k, v in D["R"].items()}
    ADD = {float(k): v for k, v in D["ADD"].items()}
    print(f"\n=== {D['kind']} ({D['n']} {'clips' if D['kind'] == 'renders' else 'songs'}{', ' + D.get('clips', '') if D['kind'] == 'renders' else ''})")
    if D["kind"] == "renders":
        print(f"{'bar':<6}{'recall':>8}{'precision':>11}{'F1':>7}{'double/1000':>13}{'notes/s':>9}{'sung recall':>13}")
        for t in THRS:
            r = R[t]
            rc, pr, f1 = prf(r["hit"], r["true"], r["hit"], r["pred"])
            print(f"{t:<6}{rc:>8.3f}{pr:>11.3f}{f1:>7.3f}{1000 * r['dbl'] / max(1, r['true']):>13.1f}"
                  f"{r['pred'] / max(1e-9, r['secs']):>9.1f}{(r['sung_hit'] / r['sung'] if r['sung'] else float('nan')):>13.3f}")
    else:
        print(f"{'bar':<6}{'strict R / P / F1':>22}{'pitch-class R / P / F1':>27}{'notes/s':>9}{'sung line recall':>18}")
        for t in THRS:
            r = R[t]
            s = prf(r["s_hit"], r["s_true"], r["s_hp"], r["pred"])
            c = prf(r["pc_hit"], r["s_true"], r["pc_hp"], r["pred"])
            print(f"{t:<6}{s[0]:>8.3f} {s[1]:.3f} {s[2]:.3f}{c[0]:>13.3f} {c[1]:.3f} {c[2]:.3f}"
                  f"{r['pred'] / max(1e-9, r['secs']):>9.1f}{r['sung_hit'] / max(1, r['sung']):>18.3f}")
        print("(human: 8.0 notes/s; real songs: right = the pitch class is in the human arrangement within 150 ms)")
    for t in LOW:
        rows = ADD[t]
        n = len(rows)
        print(f"\nnotes {t} adds over 0.9: {n}  ({n / max(1e-9, R[0.9]['secs']):.2f} per second)")
        print("   " + "  ".join(f"{k} {sum(r[0] == k for r in rows) / max(1, n):.0%}" for k in KINDS))
        if D["kind"] == "real":
            print(f"   on the sung line {sum(r[5] for r in rows) / max(1, n):.0%};  an octave / 12th of a held 0.9 note "
                  f"{sum(r[6] for r in rows) / max(1, n):.0%} (right among those: "
                  f"{sum(r[0] == 'right' for r in rows if r[6]) / max(1, sum(r[6] for r in rows)):.0%})")
        for name, col, edges in (("peak onset prob", 1, PEAK), ("length s", 2, DUR), ("velocity", 3, VEL)):
            print(f"   share right by {name}: " + ",  ".join(f"{b} {s:.0%} (n {m})" for b, m, s in bins(rows, col, edges)))
        hk = [r for r in rows if r[4]]
        print(f"   key already held in the 0.9 output: {len(hk) / max(1, n):.0%} of the added, right "
              f"{sum(r[0] == 'right' for r in hk) / max(1, len(hk)):.0%}; not held: right "
              f"{sum(r[0] == 'right' for r in rows if not r[4]) / max(1, n - len(hk)):.0%}")


def main():
    global A
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--clips")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--merge", nargs="*")
    A = ap.parse_args()
    if A.merge:
        parts = [json.loads(pathlib.Path(p).read_text()) for p in A.merge]
        groups = {}
        for d in parts:
            groups.setdefault((d["kind"], d.get("clips", "")), []).append(d)
        for ds in groups.values():
            D = dict(ds[0])
            if len(ds) > 1:
                D["n"] = sum(d["n"] for d in ds)
                D["R"] = {t: {k: sum(d["R"][t][k] for d in ds) for k in ds[0]["R"][t]} for t in ds[0]["R"]}
                D["ADD"] = {t: [r for d in ds for r in d["ADD"][t]] for t in ds[0]["ADD"]}
            report(D)
        return
    D = real() if A.real else renders()
    pathlib.Path(A.out).write_text(json.dumps(D, default=float))
    report(D)


if __name__ == "__main__":
    main()
