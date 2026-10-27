"""Solo-voice fill and remove, in the app (the user's idea, 2026-10-26).

The base reads a voice well inside a mix but drops notes when the voice is (nearly) alone.  So, after the
arranger's normal pass over a song:
  1. the solo-voice detector (app/ml/spectral/solo_train.py; ARRANGEMENT_SOLO_DETECTOR, the first one = v1) marks
     the frames where the mix is basically the voice; read per beat, runs of >= 2 beats
  2. the song's own accompaniment (melody/accompaniment split of the normal pass, outside those runs) is copied
     into them bar by bar, chord-matched to the voice and kept below it
  3. that fill is played under the solo runs (a band of FluidSynth instruments + drums when pyfluidsynth is
     installed, else a soft plucked tone) and the arranger transcribes the filled audio again
  4. the notes the fill added are removed; every note of the normal pass stays ("union": never worse than without)

Measured on the 106 real test songs (eval/solo_fill_real.py): sung notes recovered inside the solo parts, by pitch
class within 150 ms, 26.5% (normal pass) -> 32.7% (pluck fill) / 36.8% (band + drums).

    ARRANGEMENT_SOLO_DETECTOR   detector checkpoint; unset = off
    ARRANGEMENT_SOLO_RULE       removal rule (eval/solo_fill_real.py RULES): floor (default), strict, oct+wide,
                                strict+voice, floor+voice, floor+voice-pc, floor+voice all
    ARRANGEMENT_SOLO_FILL       band+drums (default) | band | pluck
    ARRANGEMENT_SOLO_PLACE      floor (default: fill kept below the voice) | octave (fill only in octave bands the
                                normal pass leaves empty in that bar; use with ARRANGEMENT_SOLO_RULE=octave)
    ARRANGEMENT_SOLO_DET_THR    detector bar per beat (0.5)
    ARRANGEMENT_SOLO_SF2        soundfont(s) for the band, comma-separated (default: OmegaGMGS2.sf2 at the repo root)
"""
import os
import pathlib
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
DET = os.environ.get("ARRANGEMENT_SOLO_DETECTOR") or None
RULE = os.environ.get("ARRANGEMENT_SOLO_RULE", "floor")
STYLE = os.environ.get("ARRANGEMENT_SOLO_FILL", "band+drums")
PLACE = os.environ.get("ARRANGEMENT_SOLO_PLACE", "floor")
DET_THR = float(os.environ.get("ARRANGEMENT_SOLO_DET_THR", "0.5"))
SF2 = os.environ.get("ARRANGEMENT_SOLO_SF2", str(ROOT / "OmegaGMGS2.sf2"))

_det = {}


def enabled() -> bool:
    return bool(DET) and pathlib.Path(DET).exists()


def _fluidsynth_ok() -> bool:
    try:
        import fluidsynth  # noqa: F401
        return True
    except Exception:                                   # noqa: BLE001  (missing package or missing DLL)
        return False


def apply(audio_path, plain, ckpt, thr):
    """The normal pass's notes plus the solo-voice notes the fill pass recovers."""
    import librosa
    import pretty_midi  # noqa: F401  (eval.solo_fill_real needs it)

    backend = ROOT / "backend"
    for p in (backend, backend / "eval"):              # eval/ holds render_arrangements (the band's synth)
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from eval import solo_fill_real as F
    from app.ml.spectral import frontend as S

    if not plain:
        return plain
    if DET not in _det:
        _det[DET] = F.detector(DET)
    y, _ = librosa.load(str(audio_path), sr=S.SR, mono=True)
    beats = F.beats_of(y)
    if len(beats) < 6:
        return plain
    runs = F.solo_beats(_det[DET](y), beats, DET_THR)
    if not runs:
        print("[solo-fill] no solo-voice parts found")
        return plain
    added, lo = F.choose_fill(plain, runs, beats, PLACE)
    if not added:
        print(f"[solo-fill] {len(runs)} solo parts, but no accompaniment to copy")
        return plain
    style = STYLE if STYLE == "pluck" or _fluidsynth_ok() else "pluck"
    F.BEATS[:] = list(beats)
    rng = np.random.default_rng(len(y) % (2 ** 31))
    with tempfile.TemporaryDirectory() as td:
        got, played = F.fill_pass(y, runs, added, style, ckpt, thr, 1.0, pathlib.Path(td) / "fill.wav", rng,
                                  [pathlib.Path(x) for x in SF2.split(",")])
    tr = F.voice_track(y, runs) if "voice" in RULE else None
    notes, new = F.union_with(plain, got, played, runs, lo, RULE, tr)
    secs = sum(b - a for a, b in runs)
    print(f"[solo-fill] {len(runs)} solo parts ({secs:.1f} s), fill {style} ({len(played)} notes), "
          f"place {PLACE}, rule {RULE}: +{len(new)} notes")
    return notes
