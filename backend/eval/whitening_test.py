"""
Fix the register collapse: how much spectral whitening, and what does it cost?

Measured on real uploads, 94% of every note the projection emitted landed below
G#3 — at every threshold, before and after the stem-balance fix.  Synthetic
mixtures ran roughly 50/50, which is why no earlier test caught it.  The cause
is structural: a low template packs forty harmonics into the bottom couple of
kHz, which is exactly where a produced record keeps its energy, so low
templates are the cheapest explanation for almost anything.

Whitening divides out a log-frequency-smoothed envelope, so templates compete
on how well they explain structure rather than on where the energy sits.

Two things have to be measured together, because whitening can trivially
"fix" the ratio by making everything equally wrong:

    on REAL audio        does the bass share fall towards the ~50% that
                         synthetic mixtures show
    on SYNTHETIC audio   does note F1 hold up, where ground truth exists

A setting that balances the registers while F1 collapses is not a fix.  The
right answer is the largest nu that leaves F1 intact.

RESULT (4 synthetic mixtures + 2 real uploads, 60 s, 2026-08-11) — REJECTED
---------------------------------------------------------------------------
      nu   synth F1   synth bass%   REAL bass%   real notes/s
    0.00      0.451           34%          95%          4.78
    0.25      0.352           27%          85%          5.96
    0.50      0.295           20%          62%          8.04
    0.75      0.263           12%          36%          9.45
    1.00      0.164            3%          18%         10.51

Whitening does move the ratio, and it wrecks accuracy doing it: by the time
real bass share reaches 62% the synthetic F1 has fallen from 0.451 to 0.295,
and synthetic material — which never had the imbalance — is pushed to 20% bass,
i.e. over-corrected in the opposite direction.  It is flattening everything
rather than fixing anything.

Limiting the number of partials per template does strictly better on both
axes at once (see N_HARMONICS in app/services/piano_projection.py): 8 partials
reaches 46% real bass share at F1 0.351, against whitening's 62% at 0.295.
WHITEN_NU therefore ships at 0.0 and the function is kept only so this
comparison can be re-run.
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
    extract_notes, get_dictionary, whiten, THRESH_Q)

SPLIT = 56
NUS = [0.0, 0.25, 0.5, 0.75, 1.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--pieces", type=int, default=4)
    args = ap.parse_args()

    import librosa, torch
    from mixture_test import (build_parts, render_part, mix, demucs_strip_drums,
                              BASS, GUITAR, DRUMS, RENDER_SR)
    from app.services.separation import _get_separator, _demucs_separate

    W = get_dictionary()
    man = json.loads((ROOT / "backend/eval/data/manifest.json").read_text())

    # ── synthetic, with ground truth ─────────────────────────────────────────
    synth = []
    for p in man["pieces"][: args.pieces]:
        midi = ROOT / p["midi_path"]
        if not midi.exists():
            continue
        low, high, drums = build_parts(midi, args.seconds)
        m = mix((render_part(low, *BASS[:2]), BASS[2]),
                (render_part(high, *GUITAR[:2]), GUITAR[2]),
                (render_part(drums, DRUMS[0], DRUMS[1]), DRUMS[2]))
        y = librosa.resample(demucs_strip_drums(m), orig_sr=RENDER_SR, target_sr=SR)
        synth.append((stft_mag(y), ref_notes(midi, args.seconds)))
    print(f"synthetic: {len(synth)} mixtures")

    # ── real uploads, drumless at original band balance ──────────────────────
    model = _get_separator()
    real = []
    TEMP = ROOT / "backend/temp"
    for d in sorted(TEMP.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        mp = d / "components.json"
        if not (mp.exists() and (d / "audio.wav").exists()):
            continue
        try:
            if not json.loads(mp.read_text()).get("method", "").startswith("demucs"):
                continue
        except Exception:                                       # noqa: BLE001
            continue
        y44, _ = librosa.load(str(d / "audio.wav"), sr=44100, mono=True,
                              duration=args.seconds)
        st = _demucs_separate(
            torch.from_numpy(np.stack([y44, y44]).astype(np.float32)), model)
        h = np.sum([v for k, v in st.items() if k != "drums"], axis=0).mean(axis=0)
        y = librosa.resample(h.astype(np.float32), orig_sr=44100, target_sr=SR)
        pk = float(np.abs(y).max())
        real.append(stft_mag(y / pk if pk > 1e-9 else y))
        if len(real) >= 2:
            break
    print(f"real: {len(real)} uploads\n")

    print(f"{'nu':>6}{'synth F1':>10}{'prec':>8}{'rec':>8}"
          f"{'synth bass%':>13}{'REAL bass%':>12}{'real n/s':>10}")
    print("-" * 67)
    for nu in NUS:
        f1s, prs, rcs, sb = [], [], [], []
        for V, ref in synth:
            est = extract_notes(nmf_h(whiten(V, nu), W), THRESH_Q)
            a, _o, pr, rc = score(ref, est)
            f1s.append(a); prs.append(pr); rcs.append(rc)
            lo = len([n for n in est if n[2] < SPLIT])
            sb.append(100.0 * lo / max(len(est), 1))
        rb, rn = [], []
        for V in real:
            est = extract_notes(nmf_h(whiten(V, nu), W), THRESH_Q)
            lo = len([n for n in est if n[2] < SPLIT])
            rb.append(100.0 * lo / max(len(est), 1))
            rn.append(len(est) / args.seconds)
        print(f"{nu:>6.2f}{np.mean(f1s):>10.3f}{np.mean(prs):>8.3f}"
              f"{np.mean(rcs):>8.3f}{np.mean(sb):>12.0f}%{np.mean(rb):>11.0f}%"
              f"{np.mean(rn):>10.2f}", flush=True)

    print("\nPick the largest nu whose synthetic F1 is intact.  A setting that")
    print("balances the registers while F1 collapses has only spread the errors")
    print("out more evenly.")


if __name__ == "__main__":
    main()
