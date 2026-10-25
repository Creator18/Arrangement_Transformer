"""How far do REAL songs sit from the render statistics the front end normalises with?

The front end standardises each (channel, pitch band) with a mean / spread measured on 64 RENDERED training
clips.  Real songs are mastered (compressed, bright, loud bass), so their features may land off-centre before
the model sees them.  This measures that shift and writes real-audio statistics for an inference-time test:

  render   raw features of rendered training clips (--renders)
  real     raw features of real Music2MIDI TRAINING songs (--m2m, by-song units; test songs untouched)

Prints, per channel and register, the real mean minus the render mean in render-spread units (0 = no
shift) and the spread ratio; writes <out>/real.npz and <out>/mixed.npz (mu, sd in the front end's shape)
for spectral_transcription's ARRANGEMENT_FE_STATS.

    python eval/fe_shift.py --renders /scratch/.../arr_render --m2m /scratch/.../m2m --out /scratch/.../fe_stats
"""
import argparse, json, pathlib, random, sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from app.ml.spectral import frontend as S                                   # noqa: E402

CH = ["h0.5", "h1", "h2", "h3", "h4", "h5", "onset"]
REG = [("bass <C3", 0, 3 * (48 - 21)), ("mid C3-C5", 3 * (48 - 21), 3 * (72 - 21)), ("high >=C5", 3 * (72 - 21), S.N_BINS)]


def stats(fe, wavs, dev):
    s = s2 = n = 0
    with torch.no_grad():
        for y in wavs:
            x = fe.raw(torch.from_numpy(y)[None].to(dev)).double()
            s = s + x.sum(dim=(0, 3), keepdim=True); s2 = s2 + (x ** 2).sum(dim=(0, 3), keepdim=True); n += x.shape[-1]
    mu = s / n
    return mu, torch.sqrt((s2 / n - mu ** 2).clamp(min=1e-6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", required=True); ap.add_argument("--m2m", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200, help="clips / songs per side"); ap.add_argument("--seconds", type=float, default=20.0)
    a = ap.parse_args()
    import librosa
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fe = S.SpectralFrontend().to(dev).eval()
    rng = random.Random(0)
    rpaths = sorted(pathlib.Path(a.renders, "train").glob("*.npz"))
    rwav = [np.load(p)["y"].astype(np.float32) / 32767 for p in rng.sample(rpaths, min(a.n, len(rpaths)))]
    units = json.loads(pathlib.Path(a.m2m, "m2m_units.json").read_text())
    keep = {str(x) for x in units["splits"]["train"]}                      # by-song split: training songs only
    audio = [p for p in sorted(pathlib.Path(a.m2m, "audio").glob("*.wav")) if p.stem in keep]
    assert audio, "no training-song audio found"
    mwav = []
    for p in rng.sample(audio, min(a.n, len(audio))):
        dur = librosa.get_duration(path=str(p))
        off = rng.uniform(0, max(0.0, dur - a.seconds))
        y, _ = librosa.load(str(p), sr=S.SR, mono=True, offset=off, duration=a.seconds)
        mwav.append(y.astype(np.float32))
    print(f"{len(rwav)} rendered clips, {len(mwav)} real training songs ({a.seconds:.0f} s each)")
    mu_r, sd_r = stats(fe, rwav, dev); mu_m, sd_m = stats(fe, mwav, dev)
    shift = ((mu_m - mu_r) / sd_r)[0, :, :, 0].cpu().numpy(); ratio = (sd_m / sd_r)[0, :, :, 0].cpu().numpy()
    print("\nreal minus render, in render-spread units (0 = same centre) | real spread / render spread")
    print(f"  {'channel':8s}" + "".join(f"{r[0]:>22s}" for r in REG))
    for c, nm in enumerate(CH):
        print(f"  {nm:8s}" + "".join(f"{shift[c, lo:hi].mean():+10.2f} | {ratio[c, lo:hi].mean():5.2f}x    " for _, lo, hi in REG))
    print(f"\n  overall: mean shift {shift.mean():+.2f}, mean |shift| {np.abs(shift).mean():.2f}, spread ratio {ratio.mean():.2f}x")
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "real.npz", mu=mu_m.float().cpu().numpy(), sd=sd_m.float().cpu().numpy())
    np.savez(out / "mixed.npz", mu=((mu_m + mu_r) / 2).float().cpu().numpy(), sd=((sd_m + sd_r) / 2).float().cpu().numpy())
    np.savez(out / "render.npz", mu=mu_r.float().cpu().numpy(), sd=sd_r.float().cpu().numpy())
    print(f"wrote {out}/real.npz, mixed.npz, render.npz")


if __name__ == "__main__":
    main()
