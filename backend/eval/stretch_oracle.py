"""Does SLOWING DOWN dense passages help the transcriber catch quick notes?  (The user's idea, 2026-10-23:
a note-counting network would pick the passages to stretch; this tests the stretching itself first, with
the passages chosen by the TRUE note density -- the "oracle".  If even the oracle doesn't help, the
counter isn't worth building.)

For each held-out render clip (exact written notes known), at the app's 0.9 bar:

  as is            transcribe the clip normally
  whole xF         stretch the whole clip by F (pitch kept), transcribe, divide the note times by F
  oracle xF        the stretched notes only inside DENSE passages (true onsets per 0.5 s above --dense),
                   the as-is notes elsewhere
  unsure xF        the same, choosing passages by the transcriber's own doubt: many onset peaks between
                   0.5 and the bar in the as-is run (no ground truth needed -- a candidate counter-free signal)

Stretchers: "pv" (librosa phase vocoder, smears attacks) and "wsola" (overlap-add of whole stretches of the
waveform, attacks kept sharp; numpy, below).  Scores: note recall / precision / F1 (exact pitch, onset
within --tol frames), the same on QUICK notes (< 96 ms after the previous written onset), and double
presses per 1000 notes.

    python eval/stretch_oracle.py --ckpt plain_base=<pt> --ckpt attack=<pt> --clips <dir> [--clips <dir>] --n 200
"""
import argparse, pathlib, random, sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from app.ml.spectral import frontend as S                                   # noqa: E402
from app.ml.transcriber.decode import decode_notes                          # noqa: E402
from app.services import spectral_transcription as st                       # noqa: E402

KW = dict(min_frames=1, min_note_s=0.0, keep_restruck=True)
WIN = 0.5                                                                    # density window, s


def wsola(y, factor, sr=S.SR, frame=0.02, search=0.006):
    """Time-stretch by `factor` (>1 = slower), pitch kept: copy windows of the ORIGINAL waveform, each
    shifted within +-search to line up best with the previous one (so attacks stay sharp)."""
    N = int(frame * sr); H_out = N // 2; H_in = H_out / factor; tol = int(search * sr)
    win = np.hanning(N)
    n_out = int(len(y) * factor) + N
    out = np.zeros(n_out); norm = np.zeros(n_out)
    yp = np.pad(y, (tol, N + tol))
    prev = None; k = 0
    while True:
        pos = int(k * H_in) + tol
        if pos + N + tol > len(yp) or k * H_out + N > n_out:
            break
        if prev is None:
            best = pos
        else:                                                              # best-aligned candidate
            cand = yp[pos - tol:pos + tol + N]
            c = np.correlate(cand, prev, mode="valid")
            best = pos - tol + int(np.argmax(c))
        seg = yp[best:best + N]
        o = k * H_out; out[o:o + N] += seg * win; norm[o:o + N] += win
        prev = yp[best + H_out:best + H_out + N]                           # natural continuation
        if len(prev) < N:
            break
        k += 1
    return (out / np.maximum(norm, 1e-3))[:int(len(y) * factor)].astype(np.float32)


def stretch(y, factor, how):
    if factor == 1.0:
        return y
    if how == "pv":
        import librosa
        return librosa.effects.time_stretch(y, rate=1.0 / factor).astype(np.float32)
    return wsola(y, factor)


def probs(y):
    """Onset / frame / velocity probabilities (88, T) for a clip, in the app's windows."""
    import tempfile, soundfile as sf
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, y, S.SR)
        p = pathlib.Path(f.name)
    try:
        return st.rolls(p)
    finally:
        p.unlink(missing_ok=True)


def match(est, ref, tol):
    """Greedy one-to-one onset matching per pitch; returns matched flags for ref and est."""
    hr, he = np.zeros(len(ref), bool), np.zeros(len(est), bool)
    by = {}
    for i, (t, p) in enumerate(est):
        by.setdefault(p, []).append(i)
    for j, (t, p) in enumerate(ref):
        best, bd = None, tol + 1e-9
        for i in by.get(p, []):
            d = abs(est[i][0] - t)
            if not he[i] and d <= bd:
                best, bd = i, d
        if best is not None:
            hr[j] = he[best] = True
    return hr, he


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, help="name=path")
    ap.add_argument("--clips", action="append", required=True)
    ap.add_argument("--n", type=int, default=200, help="clips per --clips dir")
    ap.add_argument("--factors", default="1.5,2")
    ap.add_argument("--stretchers", default="pv,wsola")
    ap.add_argument("--dense", type=float, default=6.0, help="dense passage: true onsets per 0.5 s at least this")
    ap.add_argument("--unsure", type=int, default=3, help="unsure passage: onset peaks in [0.5, 0.9) per 0.5 s at least this")
    ap.add_argument("--tol", type=int, default=3, help="onset match window, frames (+-)")
    a = ap.parse_args()
    factors = [float(f) for f in a.factors.split(",")]; hows = a.stretchers.split(",")
    tol = a.tol / S.FPS
    for spec in a.ckpt:
        name, path = spec.split("=", 1)
        st._cache.clear(); st.CKPT = pathlib.Path(path)
        for cdir in a.clips:
            files = sorted(p for p in pathlib.Path(cdir).glob("*.npz") if ".tmp" not in p.name)
            files = random.Random(0).sample(files, min(a.n, len(files)))
            acc = {}                                                       # condition -> counts

            def add(cond, ref_quick, hr, he, n_dbl):
                c = acc.setdefault(cond, np.zeros(7))
                c += [hr.sum(), len(hr), he.sum(), len(he), hr[ref_quick].sum(), ref_quick.sum(), n_dbl]
            dense_share = []
            for f in files:
                z = np.load(f); y = z["y"].astype(np.float32) / 32767
                meta = z["note_meta"] if "note_meta" in z.files else None
                if meta is None or not len(meta):
                    continue
                L = len(y) / S.SR
                ref = sorted((float(o), int(p)) for o, p, *_ in meta if 0 <= o < L)
                if len(ref) < 5:
                    continue
                ons = np.array([r[0] for r in ref])
                prev = np.r_[-1.0, ons[:-1]]
                quick = (ons - prev) < 0.096
                nwin = int(np.ceil(L / WIN)); dens = np.bincount((ons // WIN).astype(int), minlength=nwin)[:nwin]
                dense = dens >= a.dense; dense_share.append(dense.mean())
                on, fr, vl = probs(y)
                base = [(n["onset"], n["pitch"]) for n in decode_notes(on, fr, vl, 0.9, 0.5, **KW)]
                pk = ((on >= 0.5) & (on < 0.9)).sum(0)                     # doubtful onset evidence per frame
                fw = int(WIN * S.FPS)
                unsure = np.array([pk[i * fw:(i + 1) * fw].sum() for i in range(nwin)]) >= a.unsure

                def score(cond, est):
                    est = sorted(est)
                    hr, he = match(est, ref, tol)
                    # double presses: two estimated onsets of the same key within 2 tol matched to ONE written note
                    by = {}
                    for t, p in est:
                        by.setdefault(p, []).append(t)
                    rset = {}
                    for t, p in ref:
                        rset.setdefault(p, []).append(t)
                    dbl = 0
                    for p, ts in by.items():
                        rt = np.array(rset.get(p, []))
                        for t1, t2 in zip(ts, ts[1:]):
                            if t2 - t1 < 0.12 and len(rt) and not ((rt > t1 + tol) & (rt <= t2 + tol)).any():
                                dbl += 1
                    add(cond, quick, hr, he, dbl)
                score("as is", base)
                for how in hows:
                    for F in factors:
                        ys = stretch(y, F, how)
                        o2, f2, v2 = probs(ys)
                        sn = [(n["onset"] / F, n["pitch"]) for n in decode_notes(o2, f2, v2, 0.9, 0.5, **KW)]
                        score(f"whole {how} x{F:g}", sn)
                        for cname, mask in (("oracle", dense), ("unsure", unsure)):
                            inside = lambda t: mask[min(nwin - 1, int(t // WIN))]
                            mix = [x for x in base if not inside(x[0])] + [x for x in sn if inside(x[0])]
                            score(f"{cname} {how} x{F:g}", mix)
            print(f"\n=== {name} on {cdir}  ({len(files)} clips; dense passages {100 * np.mean(dense_share):.1f}% of time)")
            print(f"   {'condition':22s} {'recall':>7s} {'prec':>7s} {'F1':>7s}   {'quick recall':>12s}   {'double /1000':>12s}")
            for cond, c in acc.items():
                R, P = c[0] / max(1, c[1]), c[2] / max(1, c[3]); F1 = 2 * R * P / max(1e-9, R + P)
                print(f"   {cond:22s} {R:7.3f} {P:7.3f} {F1:7.3f}   {c[4] / max(1, c[5]):12.3f}   {1000 * c[6] / max(1, c[1]):12.1f}")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
