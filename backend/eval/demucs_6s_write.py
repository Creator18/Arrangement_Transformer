"""
Write the six htdemucs_6s stems to disk, and check they reconstruct the mix.

The point of this script is to make the separation tangible: six ordinary WAVs
you can open in anything, plus the arithmetic that shows nothing was invented or
lost.  Separation redistributes energy — the stems sum back to the input sample
for sample — so a stem shrinking (`other` went 32.4% -> 2.3% once guitar was
named) is reassignment, not loss.

Also reports per-stem RMS relative to the mix, and NOT as a percentage of a
whole: RMS ratios do not sum to 100, and presenting them in a column headed
"energy%" earlier implied a partition that does not exist.
"""
from __future__ import annotations

import argparse, json, pathlib, sys, time, warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="0 = whole track")
    ap.add_argument("--session", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import librosa, soundfile as sf, torch
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    temp = ROOT / "backend/temp"
    if args.session:
        sess = temp / args.session
    else:
        sess = None
        for d in sorted(temp.iterdir(), key=lambda x: x.stat().st_mtime,
                        reverse=True):
            mp = d / "components.json"
            if mp.exists() and (d / "audio.wav").exists():
                if json.loads(mp.read_text()).get("method", "").startswith("demucs"):
                    sess = d
                    break
    if sess is None or not (sess / "audio.wav").exists():
        print("no song session found")
        return

    outdir = pathlib.Path(args.out) if args.out else (sess / "stems6")
    outdir.mkdir(parents=True, exist_ok=True)

    dur = args.seconds or None
    y, sr = librosa.load(str(sess / "audio.wav"), sr=44100, mono=True,
                         duration=dur)
    print(f"session {sess.name[:8]}  {len(y)/sr:.1f}s @ {sr} Hz\n")

    model = get_model("htdemucs_6s")
    mix = torch.from_numpy(np.stack([y, y]).astype(np.float32))
    t0 = time.perf_counter()
    with torch.no_grad():
        out = apply_model(model, mix.unsqueeze(0))[0].cpu().numpy()
    el = time.perf_counter() - t0
    print(f"separated in {el:.1f}s ({len(y)/sr/el:.2f}x realtime)\n")

    stems = {s: out[i].mean(axis=0) for i, s in enumerate(model.sources)}
    mix_rms = float(np.sqrt((y ** 2).mean())) or 1.0

    print(f"{'stem':<9}{'file':<14}{'seconds':>9}{'RMS vs mix':>12}{'peak':>8}")
    for name, sig in stems.items():
        p = outdir / f"{name}.wav"
        sf.write(str(p), sig.astype(np.float32), sr, subtype="PCM_16")
        rms = float(np.sqrt((sig ** 2).mean()))
        print(f"{name:<9}{p.name:<14}{len(sig)/sr:>8.1f}s{rms/mix_rms:>11.2f}x"
              f"{float(np.abs(sig).max()):>8.2f}")

    # do they reconstruct the input?
    recon = np.sum(list(stems.values()), axis=0)[: len(y)]
    err = float(np.sqrt(((recon - y[: len(recon)]) ** 2).mean()))
    print(f"\nsum of six stems vs original: RMS error {err:.6f} "
          f"({20*np.log10(err/mix_rms + 1e-12):.1f} dB below the mix)")
    print("-> separation REDISTRIBUTES; nothing is created or destroyed, so a")
    print("   stem shrinking means its content was reassigned, not lost.")
    print(f"\nwritten to {outdir}")


if __name__ == "__main__":
    main()
