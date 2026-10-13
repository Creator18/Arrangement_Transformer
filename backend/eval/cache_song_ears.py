"""Stage 0 of the imaginer: v1's notes on each SONG window, computed once.

v1 is what we are correcting, so the remainders the imaginer learns are taken
against v1: kept = v1 & human, v1 - human (should drop), human - v1 (should
add).  They need A = v1's frame roll on the song, at the arrangement's own
difficulty, thresholded as the app does (frame 0.3).  The same every step, so
computed once and saved next to each pair as <id>_<w>.ears.npy (88 x T bits).
Training targets only — never an input (no human arrangement at test time).

    python eval/cache_song_ears.py --data /scratch/.../m2m_pairs --ears <s2p_v1/best.pt>
"""
from __future__ import annotations

import argparse, pathlib, sys
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--ears", required=True)
    ap.add_argument("--thr", type=float, default=0.3, help="frame threshold (the app uses 0.3 for v1)")
    a = ap.parse_args()
    from app.ml.spectral import frontend as S
    from app.ml.spectral.model import from_checkpoint
    ck = torch.load(a.ears, map_location="cuda", weights_only=False)
    ears = from_checkpoint(ck).cuda().eval()
    fe = S.SpectralFrontend().cuda().eval(); fe.mu.copy_(ck["fe_mu"]); fe.sd.copy_(ck["fe_sd"])
    paths = sorted(p for p in pathlib.Path(a.data).glob("*/*.npz") if not p.name.endswith(".tmp.npz"))
    n = 0
    with torch.no_grad():
        for p in paths:
            dst = p.with_suffix(".ears.npy")
            if dst.exists():
                continue
            z = np.load(p)
            y = torch.from_numpy(z["y"].astype(np.float32) / 32767)[None].cuda()
            cond = torch.tensor([int(z["difficulty"])]).cuda() if ears.cond_emb is not None else None
            fr = torch.sigmoid(ears(fe(y), cond)[1].float())[0].T.cpu().numpy() > a.thr   # (88, T)
            np.save(dst, np.packbits(fr, axis=1)); n += 1
            if n % 1000 == 0:
                print(f"  {n}/{len(paths)}", flush=True)
    print(f"{n} song-ears rolls cached", flush=True)


if __name__ == "__main__":
    main()
