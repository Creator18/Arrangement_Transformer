"""
corrector — score-informed note correction at inference time.

Trained on (our extraction, real notation) pairs from ASAP: 140 pieces,
180k matched notes.  At inference there is no score, so the model applies
the correction habits it learned instead of looking anything up.

Three heads were trained; only two are applied here:

    duration   scales each note's length by exp(predicted log-ratio).
               Measured on held-out pieces: mean |log duration error|
               0.504 -> 0.306 (-39%), notes within 25% of the correct
               duration 39.0% -> 51.1%.  This is the COnPOff lever.

    hand       predicts the staff the score would put the note on,
               0.883 vs 0.788 for the pitch>=C4 heuristic.  Written to
               n['hand'] for the arrangement stage; it does not change
               pitch or timing.

    keep       NOT used to delete.  It lifts precision 0.848 -> 0.926 but
               costs recall 1.000 -> 0.765, and this pipeline already
               under-emits by ~29% — paying 23 points of recall for 8 of
               precision is the wrong trade here.  The probability is
               attached as n['confidence'] so later stages can use it
               without anything being thrown away.

The featuriser must stay byte-identical to the notebook's, or the model
sees a different input distribution than it trained on.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_CKPT = Path(__file__).resolve().parent.parent.parent.parent / \
        "correction" / "data" / "corrector.pt"

_model = None
_stats: Optional[Dict] = None


def available() -> bool:
    return _CKPT.exists()


def _load():
    global _model, _stats
    if _model is not None:
        return _model, _stats
    import torch, torch.nn as nn

    ck = torch.load(str(_CKPT), map_location="cpu", weights_only=False)
    n_feat = int(ck["n_feat"])

    class Corrector(nn.Module):
        def __init__(self, n_in, h=192):
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(n_in, h), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(h, h),    nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(h, h // 2), nn.ReLU())
            self.keep = nn.Linear(h // 2, 1)
            self.dur  = nn.Linear(h // 2, 1)
            self.hand = nn.Linear(h // 2, 1)

        def forward(self, x):
            z = self.trunk(x)
            return (self.keep(z).squeeze(-1), self.dur(z).squeeze(-1),
                    self.hand(z).squeeze(-1))

    m = Corrector(n_feat)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    _model, _stats = m, {"mu": np.asarray(ck["mu"]), "sd": np.asarray(ck["sd"])}
    return _model, _stats


def _featurise(notes: List[Dict], beat_times: np.ndarray) -> Optional[np.ndarray]:
    """
    Must match make_notebook.py's featurise_piece exactly (15 features).
    Times are converted to beats first, since the model was trained in
    beat space.
    """
    if len(notes) < 5 or beat_times is None or len(beat_times) < 2:
        return None
    bt = np.asarray(beat_times, dtype=float)

    def to_beats(t: float) -> float:
        i = int(np.searchsorted(bt, t, side="right")) - 1
        i = min(max(i, 0), bt.size - 2)
        per = bt[i + 1] - bt[i]
        return float(i + ((t - bt[i]) / per if per > 1e-6 else 0.0))

    on = np.array([to_beats(float(n["onset"])) for n in notes])
    off = np.array([to_beats(float(n.get("offset", n["onset"]))) for n in notes])
    du = np.maximum(1e-3, off - on)
    pi = np.array([int(n["pitch"]) for n in notes], dtype=float)
    med_p = float(np.median(pi))

    X = []
    for i in range(len(notes)):
        lo, hi = max(0, i - 4), min(len(notes), i + 5)
        win_p, win_on = pi[lo:hi], on[lo:hi]
        simul = float(np.sum(np.abs(on - on[i]) < 0.06))
        prev_gap = on[i] - on[i - 1] if i > 0 else 2.0
        next_gap = on[i + 1] - on[i] if i < len(notes) - 1 else 2.0
        span = (win_on.max() - win_on.min()) or 1.0
        dens = len(win_on) / max(1e-6, span)
        X.append([(pi[i] - 60) / 24, (pi[i] - med_p) / 12, np.log1p(du[i]),
                  np.log1p(min(prev_gap, 4)), np.log1p(min(next_gap, 4)),
                  simul / 6, min(dens, 20) / 20,
                  (pi[i] - win_p.min()) / 12, (win_p.max() - pi[i]) / 12,
                  float(pi[i] == win_p.max()), float(pi[i] == win_p.min()),
                  (win_p.mean() - 60) / 24, win_p.std() / 12,
                  on[i] - np.floor(on[i]),
                  min(du[i] / max(1e-6, next_gap), 4) / 4])
    return np.asarray(X, dtype=np.float32)


def correct_notes(
    notes: List[Dict],
    beat_times,
    apply_duration: bool = True,
    apply_hand: bool = True,
    max_scale: float = 4.0,
) -> List[Dict]:
    """
    Apply learned corrections.  Never deletes; note count is preserved.
    Returns a new list; the input is not mutated.
    """
    if not notes or not available():
        return notes
    try:
        import torch
        model, st = _load()
        X = _featurise(notes, beat_times)
        if X is None:
            return notes
        Z = (X - st["mu"]) / st["sd"]
        with torch.no_grad():
            k, d, h = model(torch.tensor(Z, dtype=torch.float32))
            keep_p = torch.sigmoid(k).numpy()
            log_r  = d.numpy()
            hand_p = torch.sigmoid(h).numpy()
    except Exception as e:
        print(f"[corrector] skipped: {type(e).__name__}: {e}")
        return notes

    out: List[Dict] = []
    for i, n in enumerate(notes):
        m = dict(n)
        m["confidence"] = round(float(keep_p[i]), 4)
        if apply_hand:
            m["hand"] = "RH" if hand_p[i] >= 0.5 else "LH"
        if apply_duration:
            onset = float(n["onset"])
            dur = max(1e-3, float(n.get("offset", onset + 0.25)) - onset)
            # The model predicts log(ours / truth), so DIVIDE to correct.
            scale = float(np.clip(np.exp(-log_r[i]), 1.0 / max_scale, max_scale))
            m["offset"] = round(onset + dur * scale, 4)
        out.append(m)
    n_grew = sum(1 for a, b in zip(notes, out)
                 if float(b["offset"]) > float(a.get("offset", 0)) + 1e-6)
    print(f"[corrector] {len(out)} notes | durations lengthened {n_grew}, "
          f"shortened {len(out)-n_grew} | mean confidence "
          f"{float(np.mean(keep_p)):.3f}")
    return out
