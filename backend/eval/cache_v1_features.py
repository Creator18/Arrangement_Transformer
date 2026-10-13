"""Run v1 once over every Music2MIDI song; cache what the audio arranger listens to.

The audio arranger keeps v1 FROZEN, so its outputs never change during
training — computing them once makes training cheap (no spectral model in the
training loop).  Per song:

  feat    (N, 1792) float16 — v1's trunk features per 32 ms patch
          (SpectralTranscriber.features: 7 register bands x 256)
  notes   v1's own arrangement notes [on, off, pitch, vel], at the song's
          difficulty level, onset 0.60 / frame 0.3 (the tuned app settings)

Windowed like spectral_transcription.rolls: 20 s windows, 2 s discarded at
each inner edge, so every patch is computed with context on both sides.

    python eval/cache_v1_features.py --m2m /scratch/.../m2m --ckpt <v1 best.pt> --out <dir> [--shard i/n]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LEVELS = ["beginner", "intermediate", "advanced"]
WINDOW_S, EDGE_S = 20.0, 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2m", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()
    import torch, librosa
    from app.ml.spectral import frontend as S
    from app.ml.spectral.model import from_checkpoint
    from app.services import spectral_transcription as ST
    from app.ml.transcriber.decode import decode_notes

    dev = torch.device("cuda")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    net = from_checkpoint(ck).to(dev).eval()
    fe = S.SpectralFrontend().to(dev).eval(); fe.mu.copy_(ck["fe_mu"]); fe.sd.copy_(ck["fe_sd"])
    ST.CKPT = pathlib.Path(args.ckpt)
    m2m, out = pathlib.Path(args.m2m), pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sp = json.loads((m2m / "m2m_splits.json").read_text())
    ids = sorted(sp["train"] + sp["validation"] + sp["test"])
    si, sn = (int(v) for v in args.shard.split("/"))
    for sid in ids[si::sn]:
        dst = out / f"{sid}.npz"
        if dst.exists():
            continue
        wav = m2m / "audio" / f"{sid}.wav"
        cond = torch.tensor([LEVELS.index(sp["difficulty"][sid])], device=dev)
        y, _ = librosa.load(str(wav), sr=S.SR, mono=True)
        T = S.n_frames(len(y)); NP = (T + 1) // 2
        win, edge = int(WINDOW_S * S.FPS), int(EDGE_S * S.FPS)
        keep = win - 2 * edge
        feat = None
        f0 = 0
        with torch.no_grad():
            while f0 < T:
                a = max(0, f0 - edge)
                seg = y[a * S.HOP:(a + win - 1) * S.HOP]
                z = net.features(fe(torch.from_numpy(seg.astype(np.float32))[None].to(dev)), cond)[0]
                z = z.float().cpu().numpy().astype(np.float16)                 # (n_patches, 1792)
                if feat is None:
                    feat = np.zeros((NP, z.shape[1]), np.float16)
                lo = (f0 - a) // 2; hi = min(lo + keep // 2, z.shape[0], NP - a // 2)
                feat[a // 2 + lo:a // 2 + hi] = z[lo:hi]
                f0 = a + 2 * hi
        on, fr, vl = ST.rolls(wav, cond=int(cond))
        ns = decode_notes(on, fr, vl, 0.6, 0.3, min_frames=1, min_note_s=0.0, keep_restruck=True)
        notes = np.array([[n["onset"], n["offset"], n["pitch"], n["velocity"]] for n in ns], np.float32).reshape(-1, 4)
        np.savez(dst, feat=feat, notes=notes)
        print(f"  {sid}: {feat.shape[0]} patches, {len(notes)} v1 notes", flush=True)


if __name__ == "__main__":
    main()
