"""Cascaded separation: pull one source out, then separate what is left.

The idea being tested is that Demucs has an easier job on a simpler mix.
Isolating bass from a full arrangement means competing against vocal harmonics
that occupy the same low-mid register; isolating it from a vocal-free mix does
not.  So: separate once, subtract the vocals from the ORIGINAL waveform, and
separate that remainder again.

What this is NOT.  Subtraction alone changes almost nothing, and that is
measured rather than assumed: Demucs is trained with a sum constraint, and on
this song its six stems reconstruct the original to within 4.4% RMS.  So
`original - vocals` differs from `bass + guitar + piano + other + drums` by
about 5%.  Any real difference here has to come from the SECOND SEPARATION
PASS seeing a cleaner input, not from the arithmetic.

The risk, stated up front because it is the likely failure: Demucs was trained
on complete mixes.  A vocal-free signal is out of distribution, and the model
may behave oddly on it — most plausibly by hallucinating vocal content into
other stems, because it expects vocals to be present and has capacity budgeted
for them.  That would show up as the remainder carrying melody it should not.

Writes rs_vocals.wav / rs_bass.wav / rs_rest.wav for `eartest.py --cascade`.
"""
from __future__ import annotations

import argparse, pathlib, sys, warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
warnings.filterwarnings("ignore")

OUT = ROOT / "backend" / "eartest"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    args = ap.parse_args()

    import torch, torchaudio, librosa, soundfile as sf
    from app.services.separation import _get_separator, _demucs_separate
    from app.services.piano_projection import SR

    sess = ROOT / "backend/temp" / args.session
    model = _get_separator()
    sr_m, ch = int(model.samplerate), int(model.audio_channels)

    audio, sr0 = torchaudio.load(str(sess / "audio.wav"))
    if sr0 != sr_m:
        audio = torchaudio.functional.resample(audio, sr0, sr_m)
    if audio.shape[0] == 1:
        audio = audio.repeat(ch, 1)
    elif audio.shape[0] > ch:
        audio = audio[:ch]
    print(f"input {audio.shape[1] / sr_m:.1f}s @ {sr_m} Hz, {audio.shape[0]}ch")

    rms = lambda x: float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))

    print("pass 1: separating the original")
    s1 = _demucs_separate(audio, model)          # numpy (2, samples) per stem
    a_np = audio.cpu().numpy()
    vocals = s1["vocals"]
    n1 = min(a_np.shape[1], vocals.shape[1])
    r1 = a_np[:, :n1] - vocals[:, :n1]
    print(f"  R1 = original - vocals   ({20 * np.log10(rms(r1) + 1e-12):.1f} dB)")

    print("pass 2: separating the vocal-free remainder")
    s2 = _demucs_separate(torch.from_numpy(r1).float(), model)
    bass2, drums2 = s2["bass"], s2["drums"]
    # drums have to come out too, or the remainder is not comparable to the
    # drumless mix everything else in this project is built on
    n = min(r1.shape[1], bass2.shape[1], drums2.shape[1])
    r2 = r1[:, :n] - bass2[:, :n] - drums2[:, :n]

    # How much did pass 2 actually change its mind about the bass?  If this is
    # near zero the cascade is a no-op and the experiment answers itself.
    b1 = s1["bass"][:, :n]
    print(f"  bass pass1 vs pass2: "
          f"{100 * rms(b1 - bass2[:, :n]) / max(rms(b1), 1e-12):.1f}% RMS difference")
    print(f"  vocals reappearing in pass 2: "
          f"{100 * rms(s2['vocals'][:, :n]) / max(rms(b1), 1e-12):.1f}% of bass RMS "
          f"(should be ~0; high means the model hallucinated them back)")

    OUT.mkdir(exist_ok=True)
    for name, arr in (("rs_vocals", vocals), ("rs_bass", bass2), ("rs_rest", r2)):
        y = np.asarray(arr, dtype=np.float32).mean(axis=0)
        y = librosa.resample(y, orig_sr=sr_m, target_sr=SR)
        pk = float(np.abs(y).max())
        sf.write(str(OUT / f"{name}.wav"), (y / pk * 0.89 if pk > 0 else y), SR)
        print(f"  wrote {name}.wav")


if __name__ == "__main__":
    main()
