"""Onset/frame/velocity rolls -> note events, and note-level scoring.

Frame F1 is not the metric the field reports and it is not the metric that
matters here.  It is dense and forgiving: a model that smears sustain across a
whole chord scores well on frames while producing notes nobody can read.  The
standard is NOTE-level precision/recall/F1 with a 50 ms onset tolerance
(mir_eval), which is what published Slakh numbers mean.

Reporting both, with the onset-only and onset+offset variants separated, is
deliberate.  Onset-only says whether the right notes were found at the right
moments; adding offsets says whether their lengths were right too.  Collapsing
them into one number would hide which half is failing — the same error as
running a metric with no chance baseline.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from . import config as C

# A struck key must be released before it can be struck again.  POP909's
# same-pitch gaps have p25 0.068 s, so a human lifts for roughly a sixteenth.
RELEASE_S = 0.05
MIN_NOTE_S = 0.08


def decode_notes(onset: np.ndarray, frame: np.ndarray,
                 velocity: np.ndarray = None,
                 onset_thr: float = 0.5, frame_thr: float = 0.5,
                 min_frames: int = 2, min_note_s: float = None,
                 keep_restruck: bool = False) -> List[Dict]:
    """Rolls (88, T) of probabilities -> [{pitch, onset, offset, velocity}].

    A note begins only where the ONSET head fires, never merely where the
    frame head is active.  Sustain without an attack is the signature of a
    harmonic of something else, which is exactly the ghost the NMF's dictionary
    produces; requiring an attack is what lets this model reject it.
    """
    notes: List[Dict] = []
    n_pitch, T = onset.shape
    on = onset >= onset_thr
    fr = frame >= frame_thr
    for p in range(n_pitch):
        t = 0
        while t < T:
            if not on[p, t]:
                t += 1
                continue
            # the onset target was widened to +-1 frame, so a single attack
            # appears as a short run; take its first frame as the strike
            start = t
            while t < T and on[p, t]:
                t += 1
            end = t
            while end < T and fr[p, end]:
                end += 1
            if end - start < min_frames:
                continue
            v = 80
            if velocity is not None:
                v = int(np.clip(float(velocity[p, start:t].max()) * 127, 1, 127))
            notes.append({"pitch": p + C.MIDI_LO,
                          "onset": start / C.FPS,
                          "offset": max(end, start + min_frames) / C.FPS,
                          "velocity": v})
    # A piano key is ONE string.  Striking it again while it rings cuts the
    # note that was sounding — so two notes of the same pitch must never
    # overlap, and a real player lifts the key first.  Without this the decoder
    # emitted 392 overlapping same-pitch pairs on a 2368-note arrangement
    # (16.6%), heard as notes being chopped off mid-tone.  melody_decode has
    # always done this; decode_notes never did.
    by_pitch: Dict[int, List[Dict]] = {}
    for n in notes:
        by_pitch.setdefault(n["pitch"], []).append(n)
    out: List[Dict] = []
    for group in by_pitch.values():
        group.sort(key=lambda n: n["onset"])
        for a, b in zip(group[:-1], group[1:]):
            if b["onset"] < a["offset"]:
                a["offset"] = b["onset"] - RELEASE_S
                if keep_restruck:
                    # Never let the cut push a note below one frame — it is a
                    # real strike, and dropping it is how fast repeats vanish.
                    a["offset"] = min(b["onset"], max(a["offset"], a["onset"] + 1 / C.FPS))
        # Notes shorter than a grace note are clicks, not tone: 10.3% of the
        # output was under 80 ms, which a sampler renders as a transient with
        # no body.
        #
        # That is a RENDERING problem, and deleting the notes was the wrong
        # fix for transcription: measured against an unfiltered ground truth
        # (eval/spectral_detail_tests.py, 600 clips) this rule plus the
        # re-strike cut removed a third of correctly detected notes and three
        # quarters of notes re-struck within 150 ms.  min_note_s=0 with
        # keep_restruck=True keeps them; playback gives short notes an audible
        # minimum length instead (sampler.audible).  The default is unchanged
        # so existing callers behave exactly as before.
        floor = MIN_NOTE_S if min_note_s is None else min_note_s
        out.extend(n for n in group
                   if n["offset"] - n["onset"] >= floor)
    out.sort(key=lambda n: (n["onset"], n["pitch"]))
    return out


def _arrays(notes: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    if not notes:
        return np.zeros((0, 2)), np.zeros((0,))
    iv = np.array([[n["onset"], max(n["offset"], n["onset"] + 1e-3)]
                   for n in notes], dtype=float)
    pitch = np.array([n["pitch"] for n in notes], dtype=float)
    return iv, pitch


def note_scores(ref: List[Dict], est: List[Dict],
                onset_tol: float = 0.05) -> Dict[str, float]:
    """mir_eval note-level scores.  COnP = onset+pitch, COnPOff adds offsets."""
    from mir_eval.transcription import precision_recall_f1_overlap as prf
    ri, rp = _arrays(ref)
    ei, ep = _arrays(est)
    out = {"n_ref": len(ref), "n_est": len(est)}
    if len(ref) == 0 or len(est) == 0:
        out.update(conp_f1=0.0, conp_p=0.0, conp_r=0.0, conpoff_f1=0.0)
        return out
    # librosa/mir_eval need MIDI numbers as Hz for the pitch comparison
    rh = 440.0 * 2 ** ((rp - 69) / 12)
    eh = 440.0 * 2 ** ((ep - 69) / 12)
    p, r, f, _ = prf(ri, rh, ei, eh, onset_tolerance=onset_tol,
                     offset_ratio=None)
    po, ro, fo, _ = prf(ri, rh, ei, eh, onset_tolerance=onset_tol,
                        offset_ratio=0.2)
    out.update(conp_f1=float(f), conp_p=float(p), conp_r=float(r),
               conpoff_f1=float(fo))
    return out
