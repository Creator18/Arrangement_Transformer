"""
Re-tune the activation threshold on REAL audio.

THRESH_Q was chosen on synthetic renders against the piano dictionary, and it
is wrong twice over: real recordings carry noise, room and bleed that clean
renders do not, so far more activations clear the bar — one real upload emitted
4638 notes — and the harmonic dictionary has since changed the activation
distribution again.

The obvious problem is that real audio has no ground truth, so F1 cannot be
computed on it.  This uses NOTE DENSITY as a stand-in, and does not simply
assume that is valid — step 1 checks it.

  1. On synthetic mixtures WHERE ground truth exists, sweep the threshold and
     record both F1 and density.  If the threshold that maximises F1 is close
     to the one that matches real-performance density, then density is a usable
     proxy.  If they diverge, this whole approach is unsound and says so.
  2. Take the target density from MAESTRO itself — what real pianists actually
     play, in notes per second and mean simultaneous notes.
  3. Sweep on the real uploads sitting in backend/temp and pick the threshold
     whose output lands in that band.

Step 1 is the part that matters.  Tuning a parameter against a proxy nobody
checked is how a number ends up looking principled while measuring nothing.

RESULT (4 synthetic mixtures + 3 real uploads, 60 s each, 2026-08-11)
---------------------------------------------------------------------
      q     COnP   prec    rec   synth n/s   real n/s
    0.90   0.243  0.206  0.307      10.98       8.06
    0.92   0.336  0.311  0.374       8.90       6.66
    0.94   0.457  0.475  0.445       6.98       5.07
    0.95   0.515  0.578  0.471       6.07       4.18
    0.96   0.528  0.664  0.443       4.98       3.25
    0.975  0.465  0.778  0.336       3.22       1.93

THE PREMISE OF THIS RE-TUNE WAS WRONG.  The claim was that real recordings
carry noise and room so far more activations clear the bar.  Measured, real
audio is LESS dense than synthetic at every threshold — 8.06 notes/s against
10.98 at q=0.90.  And 4638 notes over that ~7 minute upload is about 10
notes/s, which is near what a pianist actually plays.  The output was never
drowning in surplus notes.

What density IS good for is transferring an operating point.  It is monotonic
in q in both domains, so the F1 peak found on synthetic audio (q=0.96, 4.98
notes/s) can be located on real audio by density: 4.98 notes/s arrives at
q=0.94.  That is the value now shipping, raised from 0.90.

Worth noting the peak sits BELOW human density (MAESTRO: 7.64 notes/s).  Past
q=0.96 precision keeps climbing (0.778 at 0.975) while F1 falls, so the notes
being added at lower thresholds are mostly wrong ones.

The real conclusion is about PRECISION, not density.  Even at the best
operating point available, precision is 0.664 — a third of the emitted notes
are wrong.  That is what "noise" sounds like, and no threshold fixes it,
because it is not a question of how many notes are emitted but of which.
"""
from __future__ import annotations

import argparse, json, pathlib, sys, warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "backend/eval"))
warnings.filterwarnings("ignore")
import scipy.signal, scipy.signal.windows                       # noqa: E402
if not hasattr(scipy.signal, "hann"):
    scipy.signal.hann = scipy.signal.windows.hann

from octave_ghost_test import SR, stft_mag, nmf_h                # noqa: E402
from note_extraction_test import score, ref_notes                # noqa: E402
from app.services.piano_projection import (                      # noqa: E402
    extract_notes, get_dictionary, FPS)

QS = [0.80, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.975, 0.99]
TEMP = ROOT / "backend/temp"


def density(notes, seconds):
    """(notes per second, mean simultaneous notes while anything sounds)."""
    if not notes or seconds <= 0:
        return 0.0, 0.0
    n_fr = int(seconds * 50)
    occ = np.zeros(n_fr, dtype=np.int16)
    for on, off, _p in notes:
        occ[int(on * 50):min(int(off * 50), n_fr)] += 1
    live = occ[occ > 0]
    return len(notes) / seconds, float(live.mean()) if live.size else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--pieces", type=int, default=4)
    args = ap.parse_args()

    import librosa
    from mixture_test import (build_parts, render_part, mix, demucs_strip_drums,
                              BASS, GUITAR, DRUMS, RENDER_SR)

    W = get_dictionary()
    man = json.loads((ROOT / "backend/eval/data/manifest.json").read_text())
    pieces = []
    for p in man["pieces"]:
        midi = ROOT / p["midi_path"]
        if midi.exists():
            pieces.append((p, midi))
        if len(pieces) >= args.pieces:
            break

    # ── 2. what real pianists actually play ──────────────────────────────────
    dens = [density(ref_notes(m, args.seconds), args.seconds) for _, m in pieces]
    tgt_nps = float(np.mean([d[0] for d in dens]))
    tgt_poly = float(np.mean([d[1] for d in dens]))
    print(f"MAESTRO reference density: {tgt_nps:.2f} notes/s, "
          f"{tgt_poly:.2f} simultaneous\n")

    # ── 1. is density a valid proxy for F1? ──────────────────────────────────
    print("synthetic mixtures, where F1 IS computable:")
    print(f"{'q':>7}{'COnP':>8}{'prec':>8}{'rec':>8}{'notes/s':>10}{'poly':>7}")
    print("-" * 48)
    cache = []
    for p, midi in pieces:
        low, high, drums = build_parts(midi, args.seconds)
        m = mix((render_part(low, *BASS[:2]), BASS[2]),
                (render_part(high, *GUITAR[:2]), GUITAR[2]),
                (render_part(drums, DRUMS[0], DRUMS[1]), DRUMS[2]))
        y = librosa.resample(demucs_strip_drums(m), orig_sr=RENDER_SR, target_sr=SR)
        cache.append((nmf_h(stft_mag(y), W), ref_notes(midi, args.seconds)))
    best_f1_q, best_f1 = None, -1.0
    rows = {}
    for q in QS:
        f1s, prs, rcs, nps, poly = [], [], [], [], []
        for H, ref in cache:
            est = extract_notes(H, q)
            a, _b, pr, rc = score(ref, est)
            d = density(est, args.seconds)
            f1s.append(a); prs.append(pr); rcs.append(rc)
            nps.append(d[0]); poly.append(d[1])
        rows[q] = (np.mean(f1s), np.mean(nps), np.mean(poly))
        print(f"{q:>7.3f}{np.mean(f1s):>8.3f}{np.mean(prs):>8.3f}"
              f"{np.mean(rcs):>8.3f}{np.mean(nps):>10.2f}{np.mean(poly):>7.2f}")
        if np.mean(f1s) > best_f1:
            best_f1_q, best_f1 = q, float(np.mean(f1s))
    dens_q = min(QS, key=lambda q: abs(rows[q][1] - tgt_nps))
    print(f"\n  best F1 at q={best_f1_q} ({best_f1:.3f}), emitting {rows[best_f1_q][1]:.2f} notes/s")
    print(f"  matching a HUMAN's {tgt_nps:.2f} notes/s would instead pick q={dens_q}")
    print("  Density is monotonic in q, so it transfers a chosen OPERATING POINT")
    print("  across domains.  It is not a target in itself: this extractor peaks")
    print("  emitting fewer notes than a pianist plays, because past that point")
    print("  the notes it adds are mostly wrong.")

    # ── 3. the real uploads ──────────────────────────────────────────────────
    songs = []
    for d in sorted(TEMP.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        meta = d / "components.json"
        if not meta.exists() or not (d / "components" / "bass.wav").exists():
            continue
        try:
            if json.loads(meta.read_text()).get("method", "").startswith("demucs"):
                songs.append(d)
        except Exception:                                       # noqa: BLE001
            continue
        if len(songs) >= 3:
            break

    if not songs:
        print("\nno real song sessions found in backend/temp")
        return
    print(f"\nreal uploads ({len(songs)} sessions, first {args.seconds:.0f}s):")
    print(f"{'q':>7}" + "".join(f"{s.name[:8]:>12}" for s in songs) + f"{'mean n/s':>11}")
    print("-" * (7 + 12 * len(songs) + 11))

    Hs = []
    for d in songs:
        parts = []
        for name in ("bass.wav", "upper.wav"):
            y, _ = librosa.load(str(d / "components" / name), sr=SR, mono=True,
                                duration=args.seconds)
            parts.append(y)
        n = max(len(p) for p in parts)
        sig = np.zeros(n, dtype=np.float32)
        for p in parts:
            sig[: len(p)] += p
        pk = float(np.abs(sig).max())
        Hs.append(nmf_h(stft_mag(sig / pk * 0.95 if pk > 1e-9 else sig), W))

    for q in QS:
        ds = [density(extract_notes(H, q), args.seconds)[0] for H in Hs]
        print(f"{q:>7.3f}" + "".join(f"{v:>12.2f}" for v in ds)
              + f"{np.mean(ds):>11.2f}")
    # The target is NOT the MAESTRO density — it is the density the extractor
    # produces at its own F1 optimum.  Matching a human performance's note rate
    # would ignore that this extractor trades precision for recall at a
    # particular operating point, and that point is what we actually want to
    # reproduce on real audio.
    f1_opt_nps = rows[best_f1_q][1]
    real = {q: float(np.mean([density(extract_notes(H, q), args.seconds)[0]
                              for H in Hs])) for q in QS}
    pick = min(QS, key=lambda q: abs(real[q] - f1_opt_nps))
    print(f"\nF1-optimal density on synthetic: {f1_opt_nps:.2f} notes/s (q={best_f1_q})")
    print(f"MAESTRO human reference:          {tgt_nps:.2f} notes/s")
    print(f"-> on real audio that density arrives at q={pick} "
          f"({real[pick]:.2f} notes/s)")


if __name__ == "__main__":
    main()
