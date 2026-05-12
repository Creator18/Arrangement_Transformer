# Arrangement Transformer — ML Core

Phase 1 scaffolding for the learned 88-key → 61-key arrangement
transformation. Replaces the existing rule-based pipeline once a trained
checkpoint exists and `USE_ML_PIPELINE` is flipped to `True` in
`config.py`.

NMF appears nowhere in this package — by design.

## Layout

```
ml/
  config.py                 single source of truth for hyperparameters
  models/
    spectrogram.py          differentiable piano-roll → mel-spectrogram
    generator.py            Transformer encoder (88+cond → 61)
  losses/
    spectral.py             α — JS divergence on cropped mel
    rhythm.py               β — onset-density MSE per beat
    contour.py              γ — melodic-contour cosine distance
    sparsity.py             δ — penalty above 10 simultaneous keys
    tempo.py                ε — implied-BPM deviation
    composite.py            weighted sum + schedule + breakdown logging
  training/
    checkpoint.py           atomic save / strict-config load
    loop.py                 main training loop with weight schedule
  inference/
    transform.py            adapter the /transform route will call
  data/                     (phase 2) MAESTRO loader, noise generator
```

## Setup

```bash
cd backend
pip install -r requirements.txt
```

The default `torch` install is CPU-only. For GPU training, install the
matching CUDA build from <https://pytorch.org/get-started/locally/>
*before* `pip install -r requirements.txt` so it doesn't get downgraded.

## Smoke-test (no real data)

```python
import torch
from app.ml.models import ArrangementTransformer, PianoRollSpectrogram
from app.ml.losses import CompositeLoss
from app.ml.config import PIANO_DIM, TARGET_DIM, GENERATOR

B, T = 2, 400
roll  = torch.zeros(B, T, PIANO_DIM); roll[:, ::8, 40:60] = 1.0
cond  = torch.zeros(B, T, GENERATOR.cond_dim)
noise = torch.zeros(B, T, PIANO_DIM)

model = ArrangementTransformer()
loss  = CompositeLoss()

out = model(roll, cond, noise)
total, breakdown = loss(roll, out, step=0)
total.backward()
print(breakdown)
```

If that prints a `LossBreakdown` and doesn't crash, the network and loss
graph are wired correctly.

## Training

Two CLI modes — pick whichever applies:

```bash
# Smoke test — no MAESTRO required.  Verifies the full forward + backward
# + loss + checkpoint round-trip end-to-end with synthetic rolls.
cd backend
python -m scripts.train --synthetic --steps 200

# Real training against MAESTRO v3.0.0.
# Download from https://magenta.tensorflow.org/datasets/maestro and
# extract under <repo-root>/data/maestro-v3.0.0/ first.
python -m scripts.train --steps 50000
```

Checkpoints land in `<repo-root>/checkpoints/generator_latest.pt`. The
loop resumes from this file automatically — `--no-resume` forces a fresh
start.

The dataset is an `IterableDataset`; it yields `Batch(roll, cond, noise)`
tuples indefinitely, so the `--steps` argument is what controls run length
(not "epochs").

### Loss breakdown logging

Every `TRAIN.log_every` steps you'll see a line like:

```
[step    150] lr=2.00e-04 total=0.4231 spec=0.4188 rhy=0.0000 con=0.0000 spa=0.0043 tem=0.0000
```

`spec` and `spa` are non-zero from step 0; the other three ramp in per the
schedule below.  When debugging plateaus this breakdown is the first place
to look.

## Integration with `/transform`

When a checkpoint exists and validates:

1. Set `USE_ML_PIPELINE = True` in `config.py`.
2. The existing `routers/transform.py` calls
   `app.ml.inference.transform.transform_with_model(...)` instead of the
   rule-based path (single import switch — see phase 3).
3. Post-processing rules (priority hierarchy, star protection, ghost
   notes, tempo-grid snapping) still run **after** the model output —
   they're deterministic safety rails, not the primary transformer.

## Loss schedule

From `config.SCHEDULE`:

| step      | spectral | sparsity | rhythm | contour | tempo |
|-----------|:--------:|:--------:|:------:|:-------:|:-----:|
| 0–2000    | ✓        | ✓        | —      | —       | —     |
| 4000+     | ✓        | ✓        | ramp   | —       | —     |
| 6000+     | ✓        | ✓        | ✓      | ramp    | —     |
| 8000+     | ✓        | ✓        | ✓      | ✓       | ramp  |
| 12000+    | all five fully active                        |

Spec rationale: spectral + sparsity from step 0 keeps the model from
collapsing into all-keys-on; later terms ramp in once the spectrogram
match is stable, so they refine rather than fight the primary objective.
