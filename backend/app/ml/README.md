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

The default install pulls **CPU-only** torch wheels. For GPU training,
swap them for a matching CUDA build:

```bash
pip uninstall torch torchaudio -y
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

`cu124` covers everything from Ampere through Ada Lovelace (RTX 30/40
series, A100, H100).  Verify with:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

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

### 1. Get the data

MAESTRO MIDI-only is ~80 MB compressed / ~250 MB extracted (1276
performances).  A bundled downloader handles this:

```bash
cd backend
python -m scripts.download_maestro
```

Extracts into `<repo-root>/data/maestro-v3.0.0/` — the path
`app.ml.config.PATHS.maestro_root` expects.  Re-running is a no-op.

### 2. Run training

```bash
# (a) Smoke test — no MAESTRO required.  Verifies forward + backward +
#     loss + checkpoint round-trip end-to-end with synthetic rolls.
python -m scripts.train --synthetic --steps 200

# (b) Short verification run on real data (~10 min on CPU).
#     Confirms loss decreases on actual music.
python -m scripts.train --steps 200 --no-resume

# (c) Full training run.
#     CPU: ~3 s/step → 50 000 steps ≈ 42 hours.
#     CUDA (single consumer GPU): ~10× faster → ~4 hours.
python -m scripts.train --steps 50000
```

Checkpoints land in `<repo-root>/checkpoints/generator_latest.pt`.
The loop resumes from this file automatically on subsequent runs;
pass `--no-resume` to force a fresh start.

### 3. Flip the switch

After training:

```python
# in app/ml/config.py
USE_ML_PIPELINE = True
```

The `/transform` route will now call `transform_with_model(...)` from
`app.ml.inference` instead of the rule-based `extract_layers` path.
Post-processing (compress, walking bass, voice leading, tempo
enforcement) still runs on top of the ML output as the safety layer.

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
