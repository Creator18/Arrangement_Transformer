"""Song -> song hum -> imagined piano hum -> piano notes: the user's hum route, for the app.

  1. song hum      Basic Pitch's pitch map of the song, played as sines whose
                   loudness follows the song (eval/make_piano_hums.hum_audio —
                   exactly how the training hums were made)
  2. its map       per-pitch loudness, CQT on the hum's own bins (eval/make_hum_maps)
  3. stage 2       step 1: the song hum's own A-B / B-A; step 2: song hum +
                   those differences -> the imagined piano hum (app/ml/spectral/hum2hum.py),
                   run in 20 s windows as trained
  4. piano hum     rendered back to sines by the same renderer
  5. stage 1       the from-scratch hum reader (hum_s1_scratch) turns it into notes

Nothing is snapped or cleaned between the steps.  The hums are saved in the
session folder (song_hum.wav, piano_hum.wav) so they can be heard.

Selected by ARRANGEMENT_SONG_ARRANGER_KIND=hum; checkpoints from
ARRANGEMENT_HUM2HUM (stage 2) and ARRANGEMENT_HUM_READER (stage 1), reader
onset threshold ARRANGEMENT_HUM_READER_THR (0.8: where stage 1 reports about
as many notes as there are).
"""
from __future__ import annotations

import os
import pathlib
import sys
from typing import Any, Dict, List

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
CK = ROOT / "app" / "ml" / "spectral" / "checkpoints"
HUM2HUM = pathlib.Path(os.environ.get("ARRANGEMENT_HUM2HUM", CK / "hum2hum" / "best.pt"))
READER = pathlib.Path(os.environ.get("ARRANGEMENT_HUM_READER", CK / "hum_s1_scratch" / "best.pt"))
READER_THR = float(os.environ.get("ARRANGEMENT_HUM_READER_THR", "0.8"))
WIN = 1723                      # frames per stage-2 window: 20 s at 22.05 kHz / 256, as trained

_cache: Dict[str, Any] = {}


def _stage2():
    if "s2" not in _cache:
        import torch
        from app.ml.spectral.hum2hum import HumNet
        c = torch.load(HUM2HUM, map_location="cpu", weights_only=False); g = c["args"]
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        s1 = HumNet(1, 2, g["d"], g["depth"]); s2 = HumNet(3, 1, g["d"], g["depth"], residual=True)
        s1.load_state_dict(c["s1"]); s2.load_state_dict(c["s2"])
        _cache["s2"] = (s1.to(dev).eval(), s2.to(dev).eval(), dev)
        print(f"[hum] stage 2 loaded ({HUM2HUM.parent.name}, round {c.get('round')}) on {dev}")
    return _cache["s2"]


def transcribe(audio_path: pathlib.Path) -> List[Dict[str, Any]]:
    import librosa
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly
    sys.path.insert(0, str(ROOT / "eval"))
    from basic_pitch import inference, ICASSP_2022_MODEL_PATH
    from make_piano_hums import hum_audio
    from make_hum_maps import hum_map
    from hum2hum_report import render
    from app.ml.spectral import frontend as S
    from app.ml.spectral.hum2hum import to_log, SCALE
    from app.services import spectral_transcription as ST

    audio_path = pathlib.Path(audio_path); out = audio_path.parent
    y22, _ = librosa.load(str(audio_path), sr=22050, mono=True)
    tmp = out / "_hum_in.wav"; sf.write(tmp, y22, 22050)
    sal = inference.run_inference(tmp, inference.Model(ICASSP_2022_MODEL_PATH))["contour"]
    tmp.unlink(missing_ok=True)
    n24 = int(round(len(y22) * S.SR / 22050))
    song_hum = hum_audio(y22, sal, S.SR, n24)                                   # 1. song hum (24 kHz)
    sf.write(out / "song_hum.wav", song_hum, S.SR)
    A = hum_map(song_hum)                                                       # 2. (264, T)
    s1, s2, dev = _stage2()
    P = np.zeros_like(A, dtype=np.float32)
    with torch.no_grad():                                                       # 3. stage 2, 20 s windows
        for f0 in range(0, A.shape[1], WIN):
            a = to_log(torch.from_numpy(A[:, f0:f0 + WIN].astype(np.float32))[None, None].to(dev))
            p = s2(torch.cat([a, s1(a)], 1))
            P[:, f0:f0 + WIN] = np.expm1(p[0, 0].cpu().numpy() * SCALE)
    piano22 = render(P)                                                         # 4. imagined piano hum
    sf.write(out / "piano_hum.wav", piano22, 22050)
    hum24 = out / "_piano_hum_24k.wav"
    sf.write(hum24, resample_poly(piano22, 160, 147).astype(np.float32), S.SR)
    notes = ST.transcribe(hum24, onset_thr=READER_THR, ckpt=READER, frame_thr=0.5)   # 5. stage 1 reads it
    hum24.unlink(missing_ok=True)
    print(f"[hum] song -> song hum -> piano hum -> {len(notes)} notes (reader thr {READER_THR}); "
          f"hums saved as song_hum.wav / piano_hum.wav in {out.name}")
    return notes
