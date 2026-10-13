"""Stage 0 of the imaginer: what the frozen piano ears hear in each SONG window, computed once.

The kept/dropped/added targets need A = the piano ears' frame roll on the song.
It is the same every step for a given window, so it is computed here once and
saved next to each pair as <id>_<w>.ears.npy (88 x T bits, np.packbits on time).

    python eval/cache_song_ears.py --data /scratch/.../m2m_pairs --ears <spectral_real/best.pt>
"""
from __future__ import annotations

import argparse, pathlib, sys
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--ears", required=True)
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
            y = torch.from_numpy(np.load(p)["y"].astype(np.float32) / 32767)[None].cuda()
            fr = torch.sigmoid(ears(fe(y))[1].float())[0].T.cpu().numpy() > 0.5      # (88, T)
            np.save(dst, np.packbits(fr, axis=1)); n += 1
            if n % 1000 == 0:
                print(f"  {n}/{len(paths)}", flush=True)
    print(f"{n} song-ears rolls cached", flush=True)


if __name__ == "__main__":
    main()
