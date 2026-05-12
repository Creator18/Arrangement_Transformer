"""
CLI entrypoint for training the per-note pattern/melody classifier.

Usage from `backend/`:
    python -m scripts.train_pattern_classifier                  # full MAESTRO
    python -m scripts.train_pattern_classifier --max-pieces 20  # smoke test
    python -m scripts.train_pattern_classifier --steps 5000

The checkpoint lands at <repo-root>/checkpoints/pattern_classifier.pt and
is picked up automatically by app.ml.pattern_classifier.inference at the
next /transform call.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.ml.config import PATHS
from app.ml.pattern_classifier.train import train_classifier, DEFAULT_CKPT


def main() -> int:
    p = argparse.ArgumentParser(description="Train per-note pattern classifier")
    p.add_argument("--maestro-root", type=Path, default=PATHS.maestro_root,
                   help=f"MAESTRO root (default: {PATHS.maestro_root})")
    p.add_argument("--max-pieces", type=int, default=0,
                   help="Limit pieces (0 = all, useful for smoke tests)")
    p.add_argument("--steps", type=int, default=2_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default=None)
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    args = p.parse_args()

    summary = train_classifier(
        root        = args.maestro_root,
        max_pieces  = args.max_pieces,
        steps       = args.steps,
        batch_size  = args.batch_size,
        lr          = args.lr,
        device      = args.device,
        ckpt_path   = args.ckpt,
    )
    print(f"[pattern_clf] summary: {summary}")
    return 0 if summary.get("trained") else 1


if __name__ == "__main__":
    raise SystemExit(main())
