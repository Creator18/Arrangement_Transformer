"""The user's stem split (2026-10-27): Demucs splits the song into the voice and the rest; the voice is read at a low
onset bar (0.8, where the base hears the sung line) and cleaned up with a stricter keep bar, the rest is read at the
0.9 bar; the two note sets are joined and quantized together (cleanup v2's position output, madmom beats on the mix).

    ARRANGEMENT_STEM_SPLIT          1 = on (needs a v2 ARRANGEMENT_CLEANUP checkpoint)
    ARRANGEMENT_STEM_VOCAL_THR      onset bar on the voice stem (0.8)
    ARRANGEMENT_STEM_VOCAL_KEEP     clean-up keep bar on the voice notes (0.5)
    ARRANGEMENT_STEM_REST_THR       onset bar on the rest (0.9)
"""
import os
import pathlib
import tempfile

import numpy as np

ON = os.environ.get("ARRANGEMENT_STEM_SPLIT", "0") == "1"
VOCAL_THR = float(os.environ.get("ARRANGEMENT_STEM_VOCAL_THR", "0.8"))
VOCAL_KEEP = float(os.environ.get("ARRANGEMENT_STEM_VOCAL_KEEP", "0.5"))
REST_THR = float(os.environ.get("ARRANGEMENT_STEM_REST_THR", "0.9"))
DUP_S = 0.05                        # a voice note on a key the rest already strikes within 50 ms is the same note
_busy = [False]


def enabled():
    return ON and not _busy[0]


def separate(audio_path, sr):
    """(vocals, rest) mono at `sr`, each scaled to the mix's loudness (the base was trained on loud masters)."""
    import librosa
    import torch
    from app.services.separation import _get_separator, _demucs_separate
    model = _get_separator()
    y, _ = librosa.load(str(audio_path), sr=model.samplerate, mono=False)
    y = np.stack([y, y]) if y.ndim == 1 else y[:2]
    st = _demucs_separate(torch.from_numpy(np.ascontiguousarray(y, np.float32)), model)
    voc = st["vocals"].mean(0)
    rest = sum(v for k, v in st.items() if k != "vocals").mean(0)
    mix_rms = float(np.sqrt(np.mean(y.mean(0) ** 2))) + 1e-9
    out = []
    for s in (voc, rest):
        s = librosa.resample(s.astype(np.float32), orig_sr=model.samplerate, target_sr=sr)
        g = mix_rms / (float(np.sqrt(np.mean(s ** 2))) + 1e-9)
        s = s * g
        out.append(s / max(1.0, float(np.abs(s).max()) / 0.99))
    return out


def _read(ST, wav, thr, ckpt, frame_thr):
    """the base's notes at `thr` and its rolls (on, fr), clean-up off."""
    cache = {}
    real = ST.rolls

    def rolls(path, cond=None, with_offset=False):
        if "r" not in cache:
            cache["r"] = real(path, cond=cond, with_offset=with_offset)
        return cache["r"]
    ST.rolls = rolls
    try:
        notes = ST.transcribe(wav, onset_thr=thr, ckpt=ckpt, frame_thr=frame_thr)
    finally:
        ST.rolls = real
    return notes, cache["r"][0], cache["r"][1]


def transcribe(audio_path, ckpt, frame_thr):
    import soundfile as sf
    from app.ml.spectral import frontend as S
    from app.services import cleanup
    from app.services import spectral_transcription as ST
    _busy[0] = True
    keep_cleanup = cleanup.CKPT
    try:
        voc, rest = separate(audio_path, S.SR)
        cleanup.CKPT = ""                                  # the base alone on each stem; the clean-up runs below
        with tempfile.TemporaryDirectory() as d:
            vw, rw = pathlib.Path(d) / "vocals.wav", pathlib.Path(d) / "rest.wav"
            sf.write(vw, voc, S.SR)
            sf.write(rw, rest, S.SR)
            vn, von, vfr = _read(ST, vw, VOCAL_THR, ckpt, frame_thr)
            rn, ron, rfr = _read(ST, rw, REST_THR, ckpt, frame_thr)
        cleanup.CKPT = keep_cleanup
        zb = cleanup.song_beats(audio_path)
        vp, vpos = cleanup.probs(vn, von, vfr, zb)
        _, rpos = cleanup.probs(rn, ron, rfr, zb)           # the rest is kept whole: only its beat positions
        keep = vp >= VOCAL_KEEP
        vk, vposk = [g for g, k in zip(vn, keep) if k], vpos[keep] if len(vn) else np.zeros((0, 8))
        by = {}
        for g in rn:
            by.setdefault(g["pitch"], []).append(g["onset"])
        new = [i for i, g in enumerate(vk) if not any(abs(o - g["onset"]) <= DUP_S for o in by.get(g["pitch"], []))]
        notes = list(rn) + [vk[i] for i in new]
        pos = np.concatenate([rpos if len(rn) else np.zeros((0, vposk.shape[1])), vposk[new]], 0)
        print(f"[stems] voice: {len(vn)} notes at {VOCAL_THR}, clean-up kept {len(vk)} (keep {VOCAL_KEEP}), "
              f"{len(new)} new over the rest;  rest: {len(rn)} notes at {REST_THR}")
        if notes:
            notes = cleanup.quantize(notes, pos, zb)
        return notes
    finally:
        cleanup.CKPT = keep_cleanup
        _busy[0] = False
