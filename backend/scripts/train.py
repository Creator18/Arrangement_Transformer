"""
Training CLI.

Usage:
    # End-to-end smoke test (no real data needed)
    python -m scripts.train --synthetic --steps 200

    # Real MAESTRO training (default — assumes data at PATHS.maestro_root)
    python -m scripts.train --steps 50000

    # Resume from a specific checkpoint
    python -m scripts.train --steps 50000 --ckpt path/to/ckpt.pt

Run from `backend/` so `app.*` imports resolve.  The smoke-test mode pulls
data from SyntheticDataset and is the fastest way to confirm that the loss
graph compiles and the gradients flow.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/train.py` from the backend/ folder
_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.ml.config import PATHS, TRAIN
from app.ml.training import train
from app.ml.data import MaestroDataset, SyntheticDataset


def main() -> int:
    p = argparse.ArgumentParser(description="Train ArrangementTransformer")
    p.add_argument("--synthetic", action="store_true",
                   help="Use SyntheticDataset (no MAESTRO required) — smoke test")
    p.add_argument("--steps", type=int, default=TRAIN.total_steps,
                   help="Total training steps")
    p.add_argument("--ckpt", type=Path, default=PATHS.default_ckpt,
                   help="Checkpoint path (load + save)")
    p.add_argument("--no-resume", action="store_true",
                   help="Ignore any existing checkpoint and start fresh")
    p.add_argument("--maestro-root", type=Path, default=PATHS.maestro_root,
                   help="MAESTRO v3.0.0 root directory")
    p.add_argument("--device", default=None,
                   help="Force device (cuda / cpu); default auto-detect")
    args = p.parse_args()

    if args.synthetic:
        print("[train] dataset = SyntheticDataset (smoke test)")
        dataset = SyntheticDataset()
    else:
        if not args.maestro_root.exists():
            print(f"[train] MAESTRO root not found: {args.maestro_root}")
            print(f"[train] hint: pass --synthetic for a quick smoke test, "
                  f"or download MAESTRO and pass --maestro-root <path>")
            return 2
        print(f"[train] dataset = MaestroDataset({args.maestro_root})")
        dataset = MaestroDataset(root=args.maestro_root)

    state = train(
        dataset     = dataset,
        device      = args.device,
        total_steps = args.steps,
        resume      = not args.no_resume,
        ckpt_path   = args.ckpt,
    )
    print(f"[train] done in {state.elapsed_s:.1f}s — last loss = {state.last_losses}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
