"""
Inference adapter for the pattern classifier.

The single public entry point is `classify_notes(notes, tempo, ...)` —
given a flat note list, it stamps each note's `layer` field with
'pattern' or 'melody'.  Used by `services/pattern_extraction.py` as a
drop-in replacement for the cycle-mining fallback when a trained
checkpoint is available.

Falls back gracefully (returns the input unchanged) when the checkpoint
file is missing — so the pipeline keeps working before phase 7 has
actually been trained.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from app.ml.config import PATHS

from .features import note_features, FEATURE_DIM
from .model    import PatternClassifier


DEFAULT_CKPT = PATHS.checkpoint_dir / "pattern_classifier.pt"


# Lazy global so the (~10K-param) model loads once per worker
_pattern_model        = None
_pattern_model_device = None


def model_available(ckpt_path: Path = DEFAULT_CKPT) -> bool:
    return Path(ckpt_path).exists()


def _get_model(device: Optional[str] = None, ckpt_path: Path = DEFAULT_CKPT):
    global _pattern_model, _pattern_model_device
    if _pattern_model is not None:
        return _pattern_model, _pattern_model_device

    import torch
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Pattern classifier checkpoint not found at {ckpt_path}")

    state = torch.load(str(ckpt_path), map_location=device)
    model = PatternClassifier(in_dim=int(state.get("in_dim", FEATURE_DIM))).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    _pattern_model        = model
    _pattern_model_device = device
    return model, device


def classify_notes(
    notes:           List[Dict],
    tempo:           float,
    beats_per_bar:   int   = 4,
    threshold:       float = 0.5,
    ckpt_path:       Path  = DEFAULT_CKPT,
) -> List[Dict]:
    """
    Stamp each note with `layer` ∈ {'pattern', 'melody'}.

    On a missing or unloadable checkpoint, returns the input unchanged
    (caller falls back to cycle-mining classification downstream).
    """
    if not notes or not model_available(ckpt_path):
        return notes

    try:
        import torch
        model, device = _get_model(ckpt_path=ckpt_path)
        X = note_features(notes, tempo=tempo, beats_per_bar=beats_per_bar)
        if X.shape[0] == 0:
            return notes
        with torch.no_grad():
            logits = model(torch.from_numpy(X).to(device))
            probs  = torch.sigmoid(logits).cpu().numpy()
        for n, p in zip(notes, probs):
            n["layer"] = "pattern" if float(p) >= threshold else "melody"
        return notes
    except Exception as e:
        print(f"[pattern_classifier] inference failed: {e}; returning unlabeled notes")
        return notes
