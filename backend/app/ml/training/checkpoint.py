"""
Checkpoint save/load.

A checkpoint stores everything needed to *resume* training (model weights,
optimizer state, training step, RNG state) plus the config that produced
it (so inference can never load a checkpoint trained with mismatched
architecture).
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from app.ml.config import GENERATOR


def save_checkpoint(
    path:       Path,
    model:      torch.nn.Module,
    optimizer:  torch.optim.Optimizer,
    step:       int,
    scheduler:  Optional[Any] = None,
    extra:      Optional[Dict[str, Any]] = None,
) -> None:
    """Atomic write — save to a temp file then rename, so a crashed save
    can never leave a half-written checkpoint behind."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    state = {
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step":      step,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "config":    asdict(GENERATOR),
        "rng":       torch.get_rng_state(),
        "cuda_rng":  torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "extra":     extra or {},
    }
    torch.save(state, tmp)
    tmp.replace(path)


def load_checkpoint(
    path:       Path,
    model:      torch.nn.Module,
    optimizer:  Optional[torch.optim.Optimizer] = None,
    scheduler:  Optional[Any] = None,
    map_location: str | torch.device = "cpu",
    strict_config: bool = True,
) -> int:
    """
    Returns the step the checkpoint was saved at, or 0 if no checkpoint
    exists.  Raises ValueError on a config mismatch when strict_config=True
    (default — guards against silently loading a checkpoint from a different
    architecture).
    """
    path = Path(path)
    if not path.exists():
        return 0

    state = torch.load(path, map_location=map_location)

    if strict_config:
        saved_cfg = state.get("config", {})
        live_cfg  = asdict(GENERATOR)
        # Compare only architecture-affecting fields
        arch_fields = {"in_dim", "out_dim", "cond_dim", "d_model", "n_heads", "n_layers", "ff_dim"}
        diffs = {k: (saved_cfg.get(k), live_cfg.get(k))
                 for k in arch_fields if saved_cfg.get(k) != live_cfg.get(k)}
        if diffs:
            raise ValueError(
                f"Checkpoint architecture mismatch at {path}: {diffs}.  "
                f"Either retrain or set strict_config=False to load anyway."
            )

    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    if "rng" in state:
        torch.set_rng_state(state["rng"])
    if state.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])

    return int(state.get("step", 0))
