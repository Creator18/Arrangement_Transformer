"""
Per-note pattern / melody classifier.

A small symbolic-feature MLP that decides, for each transcribed note,
whether it belongs to a repeating accompaniment pattern or to a free
melodic line.  Used by `services/pattern_extraction.py` to label upper-
register notes when the upstream transcription doesn't provide labels
(which has been the case since the NMF→Demucs swap in phase 3).

Replaces the cycle-mining fallback path with something that handles
higher transcription densities.  This is the prerequisite for switching
the upper-register transcriber back to Onsets & Frames without the
noise penalty we saw before.

Layout
------
    features.py   — per-note feature vectors (symbolic only)
    labels.py     — heuristic ground-truth label generation
    model.py      — small MLP classifier
    dataset.py    — MAESTRO-driven training set
    train.py      — training loop + CLI integration
    inference.py  — adapter for pattern_extraction.py
"""
from .features  import note_features, FEATURE_DIM       # noqa: F401
from .model     import PatternClassifier                  # noqa: F401
from .inference import classify_notes, model_available    # noqa: F401
