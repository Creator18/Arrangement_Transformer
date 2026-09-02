"""Configuration for the note transcriber.

Why a trained transcriber at all: the NMF must explain EVERY scrap of energy
it is handed, because a fixed dictionary of 88 piano keys is its only
vocabulary.  A singer's vibrato, cymbal wash and reverb tails all get rendered
as combinations of piano notes, and no downstream filter can undo that — three
different surplus rules were built and measured on 2026-08-15 (repetition,
harmonic fit, attack evidence) and all three made the arrangement worse by ear.
A supervised model is trained on examples where the correct answer for most of
the spectrogram is NOTHING, so it learns to ignore.  That is the capability the
current architecture cannot have.

Sized for an 8 GB laptop GPU, which is the binding constraint here rather than
disk or data.  That rules out MT3/YourMT3-scale transformers and points at the
Onsets-and-Frames shape: a convolutional front end, a small recurrent stack,
and separate onset and frame heads.
"""
from __future__ import annotations

import pathlib

# ── audio ────────────────────────────────────────────────────────────────────
SR = 16000                 # Slakh's own rate; no resampling, no loss
HOP = 256                  # 16 ms frames — 1.5x finer than the NMF's 23 ms
N_FFT = 2048
N_MELS = 229               # the Onsets-and-Frames front end, unchanged
FMIN, FMAX = 30.0, 8000.0  # 30 Hz reaches below B0; 8 kHz is Nyquist here
FPS = SR / HOP             # 62.5

# ── targets ──────────────────────────────────────────────────────────────────
MIDI_LO, MIDI_HI = 21, 108     # the 88 piano keys
N_PITCH = MIDI_HI - MIDI_LO + 1

# Drums are excluded from the targets.  They carry no pitch, Slakh flags them
# per stem (`is_drum`), and the arrangement pipeline drops them anyway — so
# training the model to predict them would be teaching it to produce something
# every downstream stage then throws away.
EXCLUDE_DRUMS = True

# An onset is one frame in the ground truth, which is far too sparse to learn
# from: at 62.5 fps a 4-minute track has ~15,000 frames per pitch and maybe 50
# onsets.  Widening the positive region to +-1 frame (48 ms total) is the
# standard fix and matches the 50 ms tolerance the note-level metric uses.
ONSET_SPREAD = 1

# ── segments ─────────────────────────────────────────────────────────────────
# Whole tracks do not fit in 8 GB.  8 s at 62.5 fps is 500 frames, which with
# batch 8 and 229 mels is comfortable and still long enough to contain several
# bars of musical context.
SEGMENT_S = 8.0
SEGMENT_FRAMES = int(SEGMENT_S * FPS)

# ── training ─────────────────────────────────────────────────────────────────
BATCH = 8
LR = 6e-4
EPOCHS = 40
# Positives are ~1% of the onset target, so unweighted BCE converges happily on
# "predict nothing" — 99% accurate and completely useless.  This is the same
# failure mode as a metric without a chance baseline, and it needs the same
# correction: state the prior explicitly.
# Swept on the overfit check (2 tracks, val IS train), note F1 at 25 epochs:
#     weight    8 -> 0.062      onset head barely moves
#     weight   30 -> 0.219      <- ships
#     weight  100 -> 0.124      over-fires; precision collapses to 0.071
# Interior optimum, so this is a real setting rather than "more is better".
ONSET_POS_WEIGHT = 30.0
FRAME_POS_WEIGHT = 2.0

# ── paths ────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parents[4]
SLAKH_DIRS = [
    pathlib.Path("D:/slakh/slakh2100_redux_16k"),   # the full set, when it lands
    pathlib.Path("D:/slakh/babyslakh_16k"),         # 20 tracks, for the harness
]
CKPT_DIR = ROOT / "backend" / "app" / "ml" / "transcriber" / "checkpoints"
