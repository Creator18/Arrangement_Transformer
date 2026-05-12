"""
Pattern-classifier training entry point.

Walks MAESTRO MIDIs, extracts symbolic features, generates heuristic
labels, trains the small MLP, saves the checkpoint.  Tiny model + small
data → seconds-to-minutes on CPU, instant on GPU.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from app.ml.config import PATHS

from .dataset import build_dataset, TensorBatchedDataset
from .model   import PatternClassifier


DEFAULT_CKPT = PATHS.checkpoint_dir / "pattern_classifier.pt"


def train_classifier(
    root:        Path  = PATHS.maestro_root,
    max_pieces:  int   = 0,
    steps:       int   = 2_000,
    batch_size:  int   = 256,
    lr:          float = 1e-3,
    weight_decay:float = 1e-4,
    device:      Optional[str] = None,
    ckpt_path:   Path  = DEFAULT_CKPT,
    log_every:   int   = 100,
) -> dict:
    """
    Train the pattern classifier end-to-end.  Returns a small summary dict.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[pattern_clf] device = {device}")

    # ── Build training set ────────────────────────────────────────────────
    print("[pattern_clf] building features + labels from MAESTRO…")
    t0 = time.time()
    X, y = build_dataset(root=root, split="train", max_pieces=max_pieces)
    print(f"[pattern_clf] {X.shape[0]} labeled notes  "
          f"({y.mean():.3f} pattern, {1-y.mean():.3f} melody)  "
          f"in {time.time()-t0:.1f}s")

    if X.shape[0] == 0:
        print("[pattern_clf] no training data — aborting")
        return {"trained": False}

    ds = TensorBatchedDataset(X, y, batch_size=batch_size)
    it = iter(ds)

    # ── Model + optimiser ──────────────────────────────────────────────────
    model = PatternClassifier().to(device)
    opt   = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    print(f"[pattern_clf] training for {steps} steps, batch {batch_size}…")
    model.train()
    running = 0.0
    t0 = time.time()
    for step in range(1, steps + 1):
        xb, yb = next(it)
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits = model(xb)
        loss   = loss_fn(logits, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        running += float(loss.item())

        if step % log_every == 0:
            avg = running / log_every
            with torch.no_grad():
                p = torch.sigmoid(logits) >= 0.5
                acc = float((p == (yb >= 0.5)).float().mean())
            print(f"  step {step:5d}  loss={avg:.4f}  acc={acc:.3f}")
            running = 0.0

    elapsed = time.time() - t0
    print(f"[pattern_clf] done in {elapsed:.1f}s")

    # ── Save checkpoint ────────────────────────────────────────────────────
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model":       model.state_dict(),
        "in_dim":      X.shape[1],
        "total_notes": int(X.shape[0]),
        "pattern_frac":float(y.mean()),
    }, ckpt_path)
    print(f"[pattern_clf] saved checkpoint to {ckpt_path}")

    return {
        "trained":     True,
        "total_notes": int(X.shape[0]),
        "elapsed_s":   elapsed,
        "ckpt_path":   str(ckpt_path),
    }
