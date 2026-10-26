"""Demucs (htdemucs, weights already in the torch cache) on the Music2MIDI songs: per-20 ms loudness of the vocal
stem and of everything else (drums + bass + other).  Only the curves are kept (npz per song), not the stems; they
are the labels for the solo-voice detector (where the mix is basically the vocal stem).

    python eval/m2m_vocal_curves.py --audio ../data/music2midi/audio --out ../data/m2m_vocal_curves
"""
import argparse
import pathlib
import time

import numpy as np
import soundfile as sf
import torch

HOP = 0.02


def db(x, sr):
    h = int(HOP * sr)
    n = x.shape[-1] // h
    e = (x[..., : n * h].reshape(-1, n, h) ** 2).mean((0, 2))
    return (10 * np.log10(e + 1e-12)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    import julius
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model = get_model("htdemucs").cuda().eval()
    srcs = model.sources                                   # drums, bass, other, vocals
    files = sorted(pathlib.Path(a.audio).glob("*.wav"))[a.shard::a.nshards]
    t0 = time.time()
    for i, f in enumerate(files):
        dst = out / (f.stem + ".npz")
        if dst.exists():
            continue
        x, sr = sf.read(f, dtype="float32")
        x = torch.from_numpy(x.T if x.ndim > 1 else np.stack([x, x]))
        x = julius.resample_frac(x, sr, model.samplerate)
        ref = x.mean(0)
        m, s = ref.mean(), ref.std() + 1e-8
        with torch.no_grad():
            y = apply_model(model, ((x - m) / s)[None].cuda(), split=True, overlap=0.1, progress=False)[0]
        y = (y * s + m).cpu().numpy()
        voc = y[srcs.index("vocals")]
        rest = sum(y[srcs.index(k)] for k in srcs if k != "vocals")
        np.savez_compressed(dst, vocal_db=db(voc, model.samplerate), rest_db=db(rest, model.samplerate),
                            mix_db=db(x.numpy(), model.samplerate), hop=HOP)
        if i % 20 == 0:
            el = time.time() - t0
            print(f"{i + 1}/{len(files)} {f.name}  {el / 60:.1f} min", flush=True)
    print("done")


if __name__ == "__main__":
    main()
