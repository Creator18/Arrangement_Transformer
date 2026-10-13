"""Song -> piano with the structure-aware audio arranger (frozen v1 ears + grammar decoder).

Same steps as eval/audio_arranger_eval.py, for one uploaded song:
  1. v1 (frozen) hears the song: its trunk features per 32 ms patch, and its
     own arrangement notes at the requested level
  2. the bar grid is counted from v1's notes (structure.label)
  3. 8-bar windows go to the arranger together; each is written with
     bar-constrained greedy decoding (the bar skeleton from the grid, every
     musical choice the model's)
  4. tokens -> timed notes on the song's beat grid

Selected by ARRANGEMENT_SONG_ARRANGER_KIND=audio_arranger, with the checkpoint
in ARRANGEMENT_AUDIO_ARRANGER (and ARRANGEMENT_AUDIO_ARRANGER_GRID=1 for a
model trained with beat-grid input).
"""
from __future__ import annotations

import os
import pathlib
from typing import Any, Dict, List

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
CKPT = pathlib.Path(os.environ.get("ARRANGEMENT_AUDIO_ARRANGER",
                                   ROOT / "app" / "ml" / "grammar" / "checkpoints" / "audio_arranger_v1" / "best.pt"))
V1 = ROOT / "app" / "ml" / "spectral" / "checkpoints" / "s2p_v1" / "best.pt"
GRID_SRC = os.environ.get("ARRANGEMENT_AUDIO_ARRANGER_GRID") == "1"
BARS = 8
_cache: Dict[str, Any] = {}


def _load():
    if "net" in _cache:
        return _cache["net"], _cache["v1"], _cache["fe"], _cache["dev"]
    import torch
    from app.ml.grammar.audio_arranger import AudioArranger
    from app.ml.spectral import frontend as S
    from app.ml.spectral.model import from_checkpoint
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = AudioArranger()
    net.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=False)["model"])
    net = net.to(dev).eval()
    ck = torch.load(V1, map_location="cpu", weights_only=False)
    v1 = from_checkpoint(ck).to(dev).eval()
    fe = S.SpectralFrontend().to(dev).eval(); fe.mu.copy_(ck["fe_mu"]); fe.sd.copy_(ck["fe_sd"])
    _cache.update(net=net, v1=v1, fe=fe, dev=dev)
    print(f"[audio-arranger] loaded {CKPT.parent.name}/{CKPT.name} on {dev}")
    return net, v1, fe, dev


def _v1_features(y, v1, fe, dev, cond):
    """v1 trunk features over the whole song, windowed with context like spectral_transcription.rolls."""
    import torch
    from app.ml.spectral import frontend as S
    T = S.n_frames(len(y)); NP = (T + 1) // 2
    win, edge = int(20.0 * S.FPS), int(2.0 * S.FPS)
    keep = win - 2 * edge
    feat, f0 = None, 0
    with torch.no_grad():
        while f0 < T:
            a = max(0, f0 - edge)
            seg = y[a * S.HOP:(a + win - 1) * S.HOP]
            z = v1.features(fe(torch.from_numpy(seg.astype(np.float32))[None].to(dev)),
                            torch.tensor([cond], device=dev))[0].float().cpu().numpy()
            if feat is None:
                feat = np.zeros((NP, z.shape[1]), np.float32)
            lo = (f0 - a) // 2; hi = min(lo + keep // 2, z.shape[0], NP - a // 2)
            feat[a // 2 + lo:a // 2 + hi] = z[lo:hi]
            f0 = a + 2 * hi
    return feat


def transcribe(audio_path: pathlib.Path, level: int = 2) -> List[Dict[str, Any]]:
    import torch, librosa
    from app.ml.spectral import frontend as S
    from app.ml.transcriber.decode import decode_notes
    from app.ml.grammar import tokens as T, tokens2 as T2, structure as STc, constrained as CD
    from app.ml.grammar.audio_arranger import LEVEL0, PATCH_S, MAX_AUDIO, MAX_SRC, collate

    net, v1, fe, dev = _load()
    y, _ = librosa.load(str(audio_path), sr=S.SR, mono=True)
    feat = _v1_features(y, v1, fe, dev, level)
    on, fr, vl = _v1_rolls(audio_path, level)
    ns = decode_notes(on, fr, vl, 0.6, 0.3, min_frames=1, min_note_s=0.0, keep_restruck=True)
    vn = np.array([[n["onset"], n["offset"], n["pitch"], n["velocity"]] for n in ns], float).reshape(-1, 4)
    if len(vn) < 20:
        return []
    st = STc.label(vn)
    grid, beats = T2.bar_grid(st), st.beats
    jobs = []
    for b0 in range(0, len(grid) - 1, BARS):
        b1 = min(b0 + BARS, len(grid) - 1)
        t_a, t_b = beats[min(grid[b0], len(beats) - 1)], beats[min(grid[b1], len(beats) - 1)]
        t_b = min(t_b, t_a + MAX_AUDIO * PATCH_S)
        fw = feat[int(t_a / PATCH_S):int(t_b / PATCH_S) + 1][:MAX_AUDIO]
        if GRID_SRC:
            src = T2.encode_plain(vn, beats, list(grid[b0:b1 + 1]))[:MAX_SRC]
        else:
            n = vn[(vn[:, 0] >= t_a) & (vn[:, 0] < t_b)].copy(); n[:, :2] -= t_a
            src = T.encode(n)[:MAX_SRC] if len(n) else []
        if len(fw):
            jobs.append((src, fw, list(grid[b0:]), CD.bar_lengths(grid[b0:], b1 - b0)))
    notes = []
    for k in range(0, len(jobs), 16):
        chunk = jobs[k:k + 16]
        src, aud, alen, _, _ = collate([(s, f, []) for s, f, _, _ in chunk])
        prefix = torch.tensor([[T.BOS, LEVEL0 + level]] * len(chunk), device=dev)
        with torch.no_grad(), torch.autocast(dev.type, enabled=dev.type == "cuda", dtype=torch.float16):
            mem, pad = net.memory(src.to(dev), aud.to(dev), alen.to(dev))
            gen = CD.generate(lambda yy: net.grammar.decode(yy, mem, pad)[:, -1], prefix, [j[3] for j in chunk])
        for (s, f, gslice, _), toks in zip(chunk, gen):
            nt, _, _ = T2.decode(toks, beats, gslice)
            notes += [tuple(r) for r in nt]
        print(f"[audio-arranger] {min(k + 16, len(jobs))}/{len(jobs)} windows", flush=True)
    out = []
    for on_, off_, p, v in sorted(notes):
        p = int(p)
        if 21 <= p <= 108 and off_ > on_:
            out.append({"pitch": p, "pitch_name": librosa.midi_to_note(p), "onset": float(on_),
                        "offset": float(off_), "velocity": int(v)})
    print(f"[audio-arranger] {len(out)} notes, {len(grid) - 1} bars ({st.meter})")
    return out


def _v1_rolls(audio_path, level):
    from app.services import spectral_transcription as ST
    keep = ST.CKPT
    ST.CKPT = V1
    try:
        return ST.rolls(audio_path, cond=level)
    finally:
        ST.CKPT = keep
