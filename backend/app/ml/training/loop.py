"""
Training loop skeleton.

The dataset and conditioned-noise generator are deliberately accepted as
ARGUMENTS (an Iterable of training batches) rather than imported.  Phase 2
will plug in the real MAESTRO-based dataset; this lets phase 1 land with a
smoke-test dataset (random rolls) so we can verify the loss + backward pass
runs end-to-end before any real data is involved.

Public entry point: `train(dataset, device, total_steps, ...)`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from app.ml.config import GENERATOR, TRAIN, PATHS
from app.ml.models.generator import ArrangementTransformer
from app.ml.losses import CompositeLoss
from .checkpoint import save_checkpoint, load_checkpoint


@dataclass
class Batch:
    """One training batch from the dataset."""
    roll:  torch.Tensor    # (B, T, 88)  binary or soft
    cond:  torch.Tensor    # (B, T, C)   chroma + chord
    noise: torch.Tensor    # (B, T, 88)  scale/chord-conditioned


@dataclass
class TrainState:
    step:        int
    last_losses: dict
    elapsed_s:   float


def train(
    dataset:     Iterable[Batch],
    device:      str | torch.device | None = None,
    total_steps: int = TRAIN.total_steps,
    resume:      bool = True,
    ckpt_path:   Path = PATHS.default_ckpt,
) -> TrainState:
    """
    Train the generator end-to-end against the composite loss.

    The dataset is expected to yield `Batch` objects indefinitely (cycle
    through your data in the iterator implementation).  We pull `total_steps`
    batches.

    Resumes from `ckpt_path` if it exists and `resume=True`.

    `device=None` (default) auto-detects CUDA → falls back to CPU.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"[train] device = {device}")
    torch.manual_seed(TRAIN.seed)

    # ── Model / loss / optimizer ────────────────────────────────────────────
    model = ArrangementTransformer().to(device)
    loss_fn = CompositeLoss().to(device)
    opt = AdamW(model.parameters(), lr=TRAIN.learning_rate,
                weight_decay=TRAIN.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=total_steps)

    start_step = 0
    if resume:
        start_step = load_checkpoint(ckpt_path, model, opt, sched, map_location=device)
        if start_step:
            print(f"[train] resumed from step {start_step}")

    # ── Loop ────────────────────────────────────────────────────────────────
    model.train()
    t0 = time.time()
    last_breakdown = {}
    it = iter(dataset)

    for step in range(start_step + 1, total_steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dataset)
            batch = next(it)

        roll  = batch.roll.to(device,  non_blocking=True)
        cond  = batch.cond.to(device,  non_blocking=True)
        noise = batch.noise.to(device, non_blocking=True)

        # Forward
        out = model(roll, cond, noise)

        # Composite loss — pass `step` so the schedule ramps the right terms
        total, breakdown = loss_fn(roll, out, step=step)

        # Backward + step
        opt.zero_grad(set_to_none=True)
        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), TRAIN.grad_clip)
        opt.step()
        sched.step()

        last_breakdown = breakdown.as_dict()

        # ── Logging / checkpoint ──────────────────────────────────────────
        if step % TRAIN.log_every == 0:
            lr = sched.get_last_lr()[0]
            print(
                f"[step {step:>6}] lr={lr:.2e} "
                f"total={breakdown.total:.4f} "
                f"spec={breakdown.spectral:.4f} "
                f"rhy={breakdown.rhythm:.4f} "
                f"con={breakdown.contour:.4f} "
                f"spa={breakdown.sparsity:.4f} "
                f"tem={breakdown.tempo:.4f}"
            )

        if step % TRAIN.checkpoint_every == 0:
            save_checkpoint(ckpt_path, model, opt, step, sched,
                            extra={"breakdown": last_breakdown})
            print(f"[train] checkpointed at step {step}")

    # Final save
    save_checkpoint(ckpt_path, model, opt, total_steps, sched,
                    extra={"breakdown": last_breakdown})

    return TrainState(
        step=total_steps,
        last_losses=last_breakdown,
        elapsed_s=time.time() - t0,
    )
