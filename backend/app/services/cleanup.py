"""Clean-up of the song arranger's notes (app/ml/spectral/cleanup_train.py, cleanup2_train.py): the base at a low onset
bar (0.8) hears the notes the user wants, voice included, plus mistakes; this model keeps or drops each note.  It never
adds one.  v2 checkpoints also say where each note sits in the beat (learned quantizing): the beats come from madmom on
the whole song for now (eval/clip_beats.py; to be replaced by our own learned beat / tempo model).

Trained on 20 s clips, so a song is read in 20 s windows every 10 s; each note's keep (and position) probability is the
average over the windows that hold it, weighted towards the window centre.

    ARRANGEMENT_CLEANUP          the clean-up checkpoint (off when unset)
    ARRANGEMENT_CLEANUP_KEEP     keep bar (default 0.4: the best F1 on held-out songs; 0.3 keeps more)
    ARRANGEMENT_CLEANUP_QUANT    1 = move notes to the beat slot the position output picks (v2 only)
    ARRANGEMENT_CLEANUP_QUANT_P  ... only when that slot's probability is at least this (default 0.5)
"""
import os
import pathlib
import sys

import numpy as np

CKPT = os.environ.get("ARRANGEMENT_CLEANUP", "")
KEEP = float(os.environ.get("ARRANGEMENT_CLEANUP_KEEP", "0.4"))
QUANT = os.environ.get("ARRANGEMENT_CLEANUP_QUANT", "0") == "1"
QUANT_P = float(os.environ.get("ARRANGEMENT_CLEANUP_QUANT_P", "0.5"))
MAX_SHIFT = 0.06                    # s: training only ever matched starts within 48 ms, so never move a note further
WIN_S, HOP_S = 20.0, 10.0
_net = {}


def enabled():
    return bool(CKPT)


def _load():
    if "net" not in _net:
        import torch
        s = torch.load(pathlib.Path(CKPT), map_location="cpu", weights_only=False)
        cfg = dict(s["cfg"])
        v2 = cfg.pop("version", 1) == 2
        if v2:
            from app.ml.spectral.cleanup2_train import Cleanup2 as Net
        else:
            from app.ml.spectral.cleanup_train import Cleanup as Net
        net = Net(**cfg)
        net.load_state_dict(s["model"])
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _net.update(net=net.to(dev).eval(), dev=dev, v2=v2)
        print(f"[cleanup] loaded {CKPT} (v{2 if v2 else 1}, step {s['step']}, held-out F1 {s['score']:.3f}) on {dev}")
    return _net["net"], _net["dev"], _net["v2"]


def song_beats(audio_path):
    """madmom beats / downbeats / activations on the whole song (the record eval/clip_beats.py writes per clip)."""
    import librosa
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "eval"))
    import clip_beats
    y, _ = librosa.load(str(audio_path), sr=clip_beats.SR, mono=True)
    return clip_beats.track(y)


def probs(notes, on, fr, zb=None):
    """keep probability per note, and (v2) position probabilities (N, NPOS).  notes: dicts with onset / offset /
    pitch / velocity; on / fr: the base's rolls; zb: song_beats() for v2."""
    import torch
    from app.ml.spectral import cleanup_train as C
    net, dev, v2 = _load()
    if not notes:
        return np.zeros(0), None
    T = on.shape[1]
    n = np.array([[g["onset"], g["offset"], g["pitch"], g["velocity"], 0.0] for g in notes], np.float32)
    k = n[:, 2].astype(int) - 21
    f = np.clip(np.round(n[:, 0] * C.FPS).astype(int), 0, T - 1)
    n[:, 4] = [on[kk, max(0, ff - 2):ff + 3].max() for kk, ff in zip(k, f)]      # peak onset probability, +-2 frames
    x = C.note_features(n, on, fr)
    if v2:
        from app.ml.spectral import cleanup2_train as C2
        x = np.concatenate([x, C2.key_features(n), C2.beat_features(n[:, 0], zb)], 1)
        pos_acc = np.zeros((len(n), C2.NPOS))
    x = x.astype(np.float32)
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
                o = net(P, Tt, X, pad)
                lk = o[0] if v2 else o
                p = torch.sigmoid(lk.float())[0].cpu().numpy()
                w = 1.0 - np.abs(n[part, 0] - (w0 + WIN_S / 2)) / (WIN_S / 2) + 1e-3     # centre-weighted
                acc[part] += w * p
                wsum[part] += w
                if v2:
                    pos_acc[part] += w[:, None] * torch.softmax(o[1].float(), -1)[0].cpu().numpy()
            if w0 + WIN_S > end:
                break
            w0 += HOP_S
    ws = np.maximum(wsum, 1e-9)
    return acc / ws, (pos_acc / ws[:, None] if v2 else None)


def keep_probs(notes, on, fr, zb=None):
    return probs(notes, on, fr, zb)[0]


def quantize(notes, pos, zb):
    """move each note to its predicted beat slot (when confident, not OFF, and within MAX_SHIFT); then merge
    strikes of one key that land on the same start."""
    from app.ml.spectral import cleanup2_train as C2
    on = np.array([g["onset"] for g in notes], float)
    tgt = C2.snap_targets(on, zb)
    cls, conf = pos.argmax(1), pos.max(1)
    moved, out = 0, []
    for i, g in enumerate(notes):
        g = dict(g)
        if cls[i] != C2.OFF and conf[i] >= QUANT_P:
            d = float(tgt[i, cls[i]] - on[i])
            if 0 < abs(d) <= MAX_SHIFT:
                g["onset"] = max(0.0, g["onset"] + d)
                g["offset"] = max(g["onset"] + 0.03, g["offset"] + d)
                moved += 1
        out.append(g)
    out.sort(key=lambda g: (g["pitch"], g["onset"]))
    merged = []
    for g in out:
        h = merged[-1] if merged else None
        if h is not None and h["pitch"] == g["pitch"] and abs(h["onset"] - g["onset"]) < 0.005:
            h["offset"] = max(h["offset"], g["offset"])
            h["velocity"] = max(h["velocity"], g["velocity"])
            continue
        merged.append(g)
    merged.sort(key=lambda g: (g["onset"], g["pitch"]))
    print(f"[cleanup] quantized {moved} of {len(notes)} notes (slot prob >= {QUANT_P}), "
          f"{len(notes) - len(merged)} same-key strikes merged")
    return merged


def apply(notes, on, fr, audio_path=None):
    _, _, v2 = _load()
    zb = None
    if v2:
        if audio_path is None:
            raise ValueError("clean-up v2 needs the song's audio for its beats")
        zb = song_beats(audio_path)
        bpm = 60 / np.median(np.diff(zb["beats"])) if len(zb["beats"]) > 2 else 0
        print(f"[cleanup] madmom: {len(zb['beats'])} beats (~{bpm:.0f} bpm), {int((zb['dbnum'] == 1).sum())} downbeats")
    p, pos = probs(notes, on, fr, zb)
    keep = p >= KEEP
    kept = [g for g, q in zip(notes, keep) if q]
    print(f"[cleanup] kept {len(kept)} of {len(notes)} notes (keep bar {KEEP})")
    if v2 and QUANT and kept:
        kept = quantize(kept, pos[keep], zb)
    return kept
