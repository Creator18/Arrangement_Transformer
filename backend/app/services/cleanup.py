"""Clean-up of the song arranger's notes (app/ml/spectral/cleanup_train.py): the base at a low onset bar (0.8) hears
the notes the user wants, voice included, plus mistakes; this model keeps or drops each note.  It never adds one.

Trained on 20 s clips, so a song is read in 20 s windows every 10 s; each note's keep probability is the average over
the windows that hold it, weighted towards the window centre.

    ARRANGEMENT_CLEANUP        the clean-up checkpoint (off when unset)
    ARRANGEMENT_CLEANUP_KEEP   keep bar (default 0.4: the best F1 on held-out songs; 0.3 keeps more)
"""
import os
import pathlib

import numpy as np

CKPT = os.environ.get("ARRANGEMENT_CLEANUP", "")
KEEP = float(os.environ.get("ARRANGEMENT_CLEANUP_KEEP", "0.4"))
WIN_S, HOP_S = 20.0, 10.0
_net = {}


def enabled():
    return bool(CKPT)


def _load():
    if "net" not in _net:
        import torch
        from app.ml.spectral.cleanup_train import Cleanup
        s = torch.load(pathlib.Path(CKPT), map_location="cpu", weights_only=False)
        net = Cleanup(**s["cfg"])
        net.load_state_dict(s["model"])
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _net.update(net=net.to(dev).eval(), dev=dev)
        print(f"[cleanup] loaded {CKPT} (step {s['step']}, held-out F1 {s['score']:.3f}) on {dev}")
    return _net["net"], _net["dev"]


def keep_probs(notes, on, fr):
    """keep probability per note (notes: dicts with onset / offset / pitch / velocity; on / fr: the base's rolls)."""
    import torch
    from app.ml.spectral import cleanup_train as C
    net, dev = _load()
    if not notes:
        return np.zeros(0)
    T = on.shape[1]
    n = np.array([[g["onset"], g["offset"], g["pitch"], g["velocity"], 0.0] for g in notes], np.float32)
    k = n[:, 2].astype(int) - 21
    f = np.clip(np.round(n[:, 0] * C.FPS).astype(int), 0, T - 1)
    n[:, 4] = [on[kk, max(0, ff - 2):ff + 3].max() for kk, ff in zip(k, f)]      # peak onset probability, +-2 frames
    x = C.note_features(n, on, fr).astype(np.float32)
    acc, wsum = np.zeros(len(n)), np.zeros(len(n))
    end = float(n[:, 0].max())
    w0 = 0.0
    with torch.no_grad():
        while True:
            idx = np.nonzero((n[:, 0] >= w0) & (n[:, 0] < w0 + WIN_S))[0]
            for part in np.array_split(idx, max(1, int(np.ceil(len(idx) / C.MAXN)))) if len(idx) else []:
                P = torch.from_numpy(n[part, 2].astype(np.int64))[None].to(dev)
                Tt = torch.from_numpy(n[part, 0] - w0)[None].to(dev)
                X = torch.from_numpy(x[part])[None].to(dev)
                pad = torch.zeros(1, len(part), dtype=torch.bool, device=dev)
                p = torch.sigmoid(net(P, Tt, X, pad).float())[0].cpu().numpy()
                w = 1.0 - np.abs(n[part, 0] - (w0 + WIN_S / 2)) / (WIN_S / 2) + 1e-3     # centre-weighted
                acc[part] += w * p
                wsum[part] += w
            if w0 + WIN_S > end:
                break
            w0 += HOP_S
    return acc / np.maximum(wsum, 1e-9)


def apply(notes, on, fr):
    p = keep_probs(notes, on, fr)
    kept = [g for g, q in zip(notes, p) if q >= KEEP]
    print(f"[cleanup] kept {len(kept)} of {len(notes)} notes (keep bar {KEEP})")
    return kept
