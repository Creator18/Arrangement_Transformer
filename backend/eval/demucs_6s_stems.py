"""
Is a 6-source separation worth using, and is the guitar stem usable?

Everything downstream currently reads harmonic.wav — Demucs' bass + vocals +
other summed into one signal — and then splits it by FREQUENCY (240 Hz) or by
PITCH NUMBER (MIDI 56).  Neither of those is a source separation, so the melody
competes with a guitar for the same activations and any vocal note below G#3 is
handed to the accompaniment rebuild.

htdemucs_6s adds guitar and piano.  Two things decide whether it helps:

  the guitar stem   Plucked and decaying is the projection's best non-piano
                    case, measured at 56.5% against 48.2% for bowed strings and
                    34.6% for organ.  If guitar separates cleanly it is a much
                    better accompaniment source than `other`.
  the vocals stem   Same model family, so vocals should be comparable to the
                    4-source version — the question is whether pulling guitar
                    out of `other` leaves the vocal cleaner or dirtier.

The authors are explicit that "the piano source is not working great at the
moment", so this measures piano rather than assuming it.

Reported per stem: share of mix energy, and what the projection makes of it —
note count, density, and how the notes distribute across the melody split.  A
stem that yields notes spread over the whole keyboard is not a separated
source; one that concentrates in its own register is.
"""
from __future__ import annotations

import argparse, json, pathlib, sys, time, warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "backend/eval"))
warnings.filterwarnings("ignore")
import scipy.signal, scipy.signal.windows                       # noqa: E402
if not hasattr(scipy.signal, "hann"):
    scipy.signal.hann = scipy.signal.windows.hann

SPLIT = 56


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60.0)
    args = ap.parse_args()

    import librosa, torch
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    from octave_ghost_test import SR, stft_mag, nmf_h
    from app.services.piano_projection import (extract_notes, get_dictionary,
                                               velocities)

    W = get_dictionary()
    sess = None
    for d in sorted((ROOT / "backend/temp").iterdir(),
                    key=lambda x: x.stat().st_mtime, reverse=True):
        mp = d / "components.json"
        if mp.exists() and (d / "audio.wav").exists():
            c = json.loads(mp.read_text())
            if c.get("method", "").startswith("demucs"):
                sess = d
                break
    if sess is None:
        print("no song session found")
        return
    print(f"session {sess.name[:8]}, first {args.seconds:.0f}s\n")

    y, _ = librosa.load(str(sess / "audio.wav"), sr=44100, mono=True,
                        duration=args.seconds)
    mix = torch.from_numpy(np.stack([y, y]).astype(np.float32))
    mix_rms = float(np.sqrt((y ** 2).mean())) or 1.0

    def run(name):
        model = get_model(name)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = apply_model(model, mix.unsqueeze(0))[0].cpu().numpy()
        el = time.perf_counter() - t0
        return {s: out[i].mean(axis=0) for i, s in enumerate(model.sources)}, el

    def project(sig):
        r = librosa.resample(sig.astype(np.float32), orig_sr=44100, target_sr=SR)
        pk = float(np.abs(r).max())
        if pk < 1e-6:
            return 0, 0.0, 0.0
        H = nmf_h(stft_mag(r / pk), W)
        est = extract_notes(H)
        if not est:
            return 0, 0.0, 0.0
        hi = sum(1 for _a, _b, p in est if p >= SPLIT)
        return len(est), len(est) / args.seconds, 100.0 * hi / len(est)

    for name in ("htdemucs", "htdemucs_6s"):
        stems, el = run(name)
        print(f"=== {name} ===  {el:.1f}s for {args.seconds:.0f}s audio "
              f"({args.seconds/el:.2f}x realtime)")
        print(f"{'stem':<9}{'energy%':>9}{'notes':>8}{'notes/s':>9}{'>=G#3':>8}")
        for s, sig in stems.items():
            share = 100.0 * float(np.sqrt((sig ** 2).mean())) / mix_rms
            n, nps, hi = project(sig)
            print(f"{s:<9}{share:>8.1f}%{n:>8}{nps:>9.2f}{hi:>7.0f}%")

        # what the pipeline actually feeds the projection today
        keep = [v for k, v in stems.items() if k != "drums"]
        harm = np.sum(keep, axis=0)
        n, nps, hi = project(harm)
        print(f"{'ALL-DRUMS':<9}{'':>9}{n:>8}{nps:>9.2f}{hi:>7.0f}%   "
              f"<- what harmonic.wav is")
        print()


if __name__ == "__main__":
    main()
