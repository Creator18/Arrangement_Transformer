"""Audio -> notes, using the trained transcriber instead of the NMF.

Drop-in alternative to piano_projection's output: same note dicts, same keys,
so every downstream stage — alignment, the scale filter, the arrangement
reduction — works unchanged and the two note sources can be A/B'd by ear
without touching anything else.

Why replace the NMF at all: it must explain EVERY scrap of energy handed to
it, because 88 piano templates are its whole vocabulary, so a voice, a cymbal
and a reverb tail all come back as combinations of piano notes.  Measured on
Slakh with ground truth, the NMF reaches note F1 0.203 against a 0.122 chance
floor; the transcriber passed that within one epoch of training.

The model runs on 8 s segments because that is what it was trained on, and
segments OVERLAP with only their centres kept.  A recurrent model has no
context beyond a segment edge, so the first and last moments of each are its
weakest output; butt-joining them would stitch precisely those together once
per segment, forever.  The same reasoning as the ML generator's windowing.
"""
from __future__ import annotations

import pathlib
from typing import Any, Dict, List, Optional

import numpy as np

# Seconds discarded at each segment edge.
#
# Was 1.0, justified by reasoning rather than measurement: a recurrent model
# has no context past a segment edge, so its first and last moments are its
# weakest output.  That is true, and it is not the whole story — the margin
# also throws away notes that only one chunk ever saw.  Measured on
# Complication.mp3, hunting a 3.4 s sustained B2 the user could hear in the
# original and not in the render:
#
#     EDGE_S   notes   sustained note 38.8-42.4s
#       1.0     2615   MISSING
#       0.5     2752   found (38.85 -> 42.37)
#       0.0     2640   found (38.85 -> 42.24)
#
# 0.5 keeps half the edge protection and recovers the note; it also yields the
# most notes of the three, so the trimming was not merely removing junk.
EDGE_S = 0.5


def available(ckpt: Optional[pathlib.Path] = None) -> bool:
    """True when a trained checkpoint exists — callers fall back to the NMF."""
    from app.ml.transcriber import config as C
    p = ckpt or (C.CKPT_DIR / "transcriber.pt")
    return p.exists()


def _load(ckpt: Optional[pathlib.Path] = None):
    import torch
    from app.ml.transcriber import config as C
    from app.ml.transcriber.data import MelFront
    from app.ml.transcriber.model import from_checkpoint
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p = ckpt or (C.CKPT_DIR / "transcriber.pt")
    ck = torch.load(p, map_location=dev, weights_only=False)
    # Built from the checkpoint's own recorded architecture, not from config.
    # More than one model size exists now, so rebuilding from current constants
    # would mean a config edit silently breaks whatever is deployed.
    net = from_checkpoint(ck).to(dev).eval()
    return net, MelFront().to(dev), dev, ck


def transcribe(audio_path: pathlib.Path,
               ckpt: Optional[pathlib.Path] = None,
               onset_thr: float = 0.90,
               frame_thr: float = 0.5) -> List[Dict[str, Any]]:
    """Notes in the pipeline's dict shape: pitch / onset / offset / velocity.

    `onset_thr` is the precision/recall dial, and it matters far more than it
    looks.  The model is deliberately biased toward firing — onsets are 0.28%
    of the target, so without a heavy positive weight it would learn to predict
    silence — and the threshold is what undoes that bias at inference, with no
    retraining.  Swept on 8 held-out Slakh tracks (mean 3875 reference notes):

        thr    notes      F1       P       R
        0.3     9027   0.377   0.272   0.644
        0.5     6961   0.468   0.366   0.671
        0.7     5279   0.545   0.473   0.659
        0.90    3181   0.617   0.682   0.570   <- ships
        0.94    2532   0.608   0.769   0.509
        0.98    1473   0.493   0.901   0.346

    An interior optimum, so it is a measured setting rather than "higher is
    better".  0.90 is also where the note count stops being wrong: 3181 emitted
    against 3875 real, where 0.5 emitted nearly twice too many.
    """
    import torch, librosa
    from app.ml.transcriber import config as C
    from app.ml.transcriber.decode import decode_notes

    net, front, dev, _ = _load(ckpt)
    y, _ = librosa.load(str(audio_path), sr=C.SR, mono=True)

    # All window arithmetic is done in FRAMES, not samples.
    #
    # Doing it in samples is how this broke twice.  MelSpectrogram(center=True)
    # returns n//hop + 1 frames, so a chunk carries one frame more than its
    # duration; and keeping `hi - lo` frames while advancing `seg - 2*edge`
    # SAMPLES kept 376 frames per step while the step was 375 frames long.
    # Either way the roll grows faster than the audio and every note drifts
    # later — about 640 ms across a four-minute track, which silently deleted a
    # 3.4 s sustained note from the output because it no longer lined up.
    #
    # Frames are the unit the decoder reads, so they are the unit to count in.
    keep = int(C.SEGMENT_S * C.FPS)          # frames kept from one chunk
    ef = int(EDGE_S * C.FPS)                 # frames discarded at each edge
    step = keep - 2 * ef                     # frames advanced per chunk
    seg = keep * C.HOP                       # samples the model needs as input

    parts = {"on": [], "fr": [], "vl": []}
    with torch.no_grad():
        f0 = 0                                # first frame this chunk contributes
        while f0 * C.HOP < len(y):
            a = max(0, f0 - ef) * C.HOP       # chunk start, in samples
            chunk = y[a:a + seg]
            if len(chunk) < seg:
                chunk = np.pad(chunk, (0, seg - len(chunk)))
            mel = front(torch.from_numpy(chunk).float().unsqueeze(0).to(dev))
            po, pf, pv = net(mel)
            lo = f0 - a // C.HOP              # skip the leading margin
            hi = min(lo + step, keep)         # ...and the trailing one
            for k, t in (("on", po), ("fr", pf), ("vl", pv)):
                parts[k].append(torch.sigmoid(t)[0, lo:hi].cpu().numpy().T)
            f0 += hi - lo

    on = np.concatenate(parts["on"], axis=1)
    fr = np.concatenate(parts["fr"], axis=1)
    vl = np.concatenate(parts["vl"], axis=1)
    notes = decode_notes(on, fr, vl, onset_thr, frame_thr)
    for n in notes:
        # Downstream stages read `role`; the transcriber does not predict
        # instrument, so everything is unassigned rather than mislabelled.
        # enforce_melody_on_top and the two-voice logic derive role from
        # register anyway, which is the structural version of the same thing.
        n["role"] = "all"
    print(f"[ml-transcribe] {len(notes)} notes from {audio_path.name} "
          f"(onset_thr {onset_thr})")
    return notes
