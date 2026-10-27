"""Solo voice by SIMILARITY, not loudness (the user's idea, 2026-10-26): separate the vocal stem and compare it with the
full song at the same moment.  Where many instruments clutter the mix, the mix looks little like the vocal stem;
where the voice has at most one or two instruments with it, the two tracks look much alike.

Demucs (htdemucs) on the Music2MIDI songs, per 20 ms (the same frames as eval/m2m_vocal_curves.py):
    sim       cosine similarity of the mel AMPLITUDE spectra of the vocal stem and the mix, both summed over +-100 ms
              (1 = the mix is the voice; low = the mix is mostly something else)
    sim_log   the same on log-mel spectra floored 60 dB below the song's peak (gives quiet partials more say)
    vocal_db, mix_db, drums_db, bass_db, other_db   loudness per stem (to check the "two or fewer instruments" reading)
Only the curves are kept (npz per song), not the stems.

    python eval/m2m_stem_similarity.py --audio /scratch/basu.ang/m2m/audio --out /scratch/basu.ang/solo_det/m2m_sim
"""
import argparse
import pathlib
import time

import numpy as np
import soundfile as sf
import torch

HOP = 0.02
CTX = 5                       # frames each side: +-100 ms


def db(x, sr):
    h = int(HOP * sr)
    n = x.shape[-1] // h
    e = (x[..., : n * h].reshape(-1, n, h) ** 2).mean((0, 2))
    return (10 * np.log10(e + 1e-12)).astype(np.float32)


def similarity(voc, mix, sr, fb):
    """(sim, sim_log) per 20 ms frame: cosine of the two mel spectra (amplitude / floored log) over +-100 ms."""
    h = int(HOP * sr)
    win = torch.hann_window(2048, device=voc.device)
    spec = lambda x: torch.stft(x.mean(0), 2048, h, window=win, return_complex=True).abs()      # (freq, frames)
    mv, mm = fb @ spec(voc), fb @ spec(mix)                                                       # (mel, frames)
    k = torch.ones(1, 1, 2 * CTX + 1, device=voc.device)
    ctx = lambda m: torch.nn.functional.conv1d(m[:, None], k, padding=CTX)[:, 0]

    def cos(a, b):
        a, b = ctx(a), ctx(b)
        return ((a * b).sum(0) / (a.norm(dim=0) * b.norm(dim=0) + 1e-9)).cpu().numpy()
    floor = 20 * torch.log10(mm.max() + 1e-9) - 60
    lg = lambda m: (20 * torch.log10(m + 1e-9) - floor).clamp(min=0)
    n = voc.shape[-1] // h
    return cos(mv, mm)[:n].astype(np.float32), cos(lg(mv), lg(mm))[:n].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="", help="json list of song ids to do (default: every wav)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    import json
    import julius
    import librosa
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model = get_model("htdemucs").cuda().eval()
    srcs = model.sources                                   # drums, bass, other, vocals
    sr = model.samplerate
    fb = torch.from_numpy(librosa.filters.mel(sr=sr, n_fft=2048, n_mels=96, fmin=40, fmax=11000)).cuda()
    files = sorted(pathlib.Path(a.audio).glob("*.wav"))
    if a.only:
        keep = {str(s) for s in json.loads(pathlib.Path(a.only).read_text())}
        files = [f for f in files if f.stem in keep]
    files = files[a.shard::a.nshards]
    t0 = time.time()
    for i, f in enumerate(files):
        dst = out / (f.stem + ".npz")
        if dst.exists():
            continue
        x, fsr = sf.read(f, dtype="float32")
        x = torch.from_numpy(x.T if x.ndim > 1 else np.stack([x, x]))
        x = julius.resample_frac(x, fsr, sr)
        ref = x.mean(0)
        m, s = ref.mean(), ref.std() + 1e-8
        with torch.no_grad():
            y = apply_model(model, ((x - m) / s)[None].cuda(), split=True, overlap=0.1, progress=False)[0]
            y = y * s + m
            xc = x.cuda()
            sim, sim_log = similarity(y[srcs.index("vocals")], xc, sr, fb)
        y = y.cpu().numpy()
        np.savez_compressed(dst, sim=sim, sim_log=sim_log, hop=HOP, mix_db=db(x.numpy(), sr),
                            **{f"{k}_db": db(y[srcs.index(k)], sr) for k in ("vocals", "drums", "bass", "other")})
        if i % 20 == 0:
            print(f"{i + 1}/{len(files)} {f.name}  {(time.time() - t0) / 60:.1f} min", flush=True)
    print("done")


if __name__ == "__main__":
    main()
