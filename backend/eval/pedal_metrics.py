"""
Score the predicted sustain pedal against ground truth.

Until now this was impossible: the eval audio was synthesised from MAESTRO
with CC64 stripped, so the model's pedal head had nothing to detect and any
pedal figure was meaningless.  With pedal in the audio it becomes a real
measurement, and it is the prerequisite for touching pedal behaviour at all.

Frame-level rather than event-level, deliberately
--------------------------------------------------
Pedal is a binary STATE over time, not a set of events.  Two pedallings can
have completely different numbers of presses and still sound the same — a
pianist who lifts and re-presses at a chord change produces two events where
another produces one long one, and the damper is up for a few milliseconds
either way.  Event matching would call that a total mismatch.  What actually
matters is: at any given instant, is the damper where it should be.

So the signals are rasterised to a 10 ms grid and compared frame by frame.
That is also how pedal is scored in the AMT literature.

Reported:
  frame F1     the headline: how often the damper state is right
  IoU          overlap of the pedal-down regions, which punishes a model
               that gets the state right mostly by predicting "down"
  down_rate    fraction of frames each signal calls pedal-down.  Printed for
               both, because a detector that simply holds the pedal for the
               whole piece can score a deceptively good F1 when the truth is
               63% down — this is the column that exposes it.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

FRAME_S = 0.010


def rasterise(intervals: Sequence[Dict], duration: float,
              frame_s: float = FRAME_S) -> np.ndarray:
    """Damper-down intervals -> boolean frames."""
    n = max(1, int(np.ceil(duration / frame_s)))
    g = np.zeros(n, dtype=bool)
    for iv in intervals or ():
        a = float(iv.get("onset", 0.0))
        b = float(iv.get("offset", a))
        if b <= a:
            continue
        i0 = max(0, int(a / frame_s))
        i1 = min(n, int(np.ceil(b / frame_s)))
        if i1 > i0:
            g[i0:i1] = True
    return g


def pedal_scores(pred: Sequence[Dict], truth: Sequence[Dict],
                 duration: float) -> Dict:
    p = rasterise(pred, duration)
    t = rasterise(truth, duration)
    n = min(len(p), len(t))
    p, t = p[:n], t[:n]

    tp = int(np.sum(p & t))
    fp = int(np.sum(p & ~t))
    fn = int(np.sum(~p & t))
    union = int(np.sum(p | t))

    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    return {
        "frames": n,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "iou": tp / union if union else 0.0,
        "down_rate_pred": float(np.mean(p)),
        "down_rate_truth": float(np.mean(t)),
        # What a "hold the pedal down forever" detector would score.  A
        # headline F1 only means something next to this.
        "f1_always_down": (2 * float(np.mean(t))) / (1 + float(np.mean(t)))
        if np.mean(t) else 0.0,
    }


def aggregate(rows: List[Dict]) -> Dict:
    if not rows:
        return {}
    keys = ("precision", "recall", "f1", "iou",
            "down_rate_pred", "down_rate_truth", "f1_always_down")
    out = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    out["pieces"] = len(rows)
    return out


def report(agg: Dict) -> str:
    if not agg:
        return "no pieces scored"
    return (
        f"\npedal — {agg['pieces']} pieces, 10 ms frames\n"
        f"  frame F1        {agg['f1']:.4f}\n"
        f"  precision       {agg['precision']:.4f}\n"
        f"  recall          {agg['recall']:.4f}\n"
        f"  IoU             {agg['iou']:.4f}\n"
        f"  down-rate  pred {agg['down_rate_pred']:.3f}   "
        f"truth {agg['down_rate_truth']:.3f}\n"
        f"  ---\n"
        f"  a detector that just held the pedal down would score "
        f"F1 {agg['f1_always_down']:.4f}\n"
        f"  so the headline is only meaningful above that line.\n"
    )


# ── First real measurement, 2026-08-10 (49 pieces, pedalled eval audio) ──────
#
#   frame F1        0.7522
#   precision       0.8085
#   recall          0.7190
#   IoU             0.6211
#   down-rate  pred 0.511    truth 0.592
#
#   always-down baseline    0.7025
#   MARGIN                 +0.0498
#
# Read that margin, not the headline.  0.75 sounds respectable and is only
# five points better than a detector that holds the pedal down for the entire
# piece and thinks about nothing.  IoU 0.62 says the same thing more bluntly.
#
# We also UNDER-pedal slightly (0.511 vs 0.592), which is worth noting against
# the offset-inflation finding: our note offsets run 2.10x long, so the
# pipeline is simultaneously holding notes too long and the pedal not long
# enough — two different mechanisms pushing sustain in opposite directions.
