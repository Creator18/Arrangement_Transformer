"""
Single source of truth for hyperparameters, paths, and loss-weight schedules.

Anything you might want to tune from a sweep or a training command lives here.
Code in models/, losses/, training/ reads from this module — do not hardcode.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


# ── Keyboard geometry ─────────────────────────────────────────────────────────

# Full piano = 88 keys, MIDI 21 (A0) → 108 (C8)
PIANO_LO_MIDI = 21
PIANO_HI_MIDI = 108
PIANO_DIM     = PIANO_HI_MIDI - PIANO_LO_MIDI + 1   # = 88

# Target 5-octave keyboard = 61 keys, MIDI 36 (C2) → 96 (C7)
TARGET_LO_MIDI = 36
TARGET_HI_MIDI = 96
TARGET_DIM     = TARGET_HI_MIDI - TARGET_LO_MIDI + 1   # = 61

# Index slice of the 88-dim roll that the 61-dim target maps to.
# Used to align discriminator spectrograms.
TARGET_SLICE: Tuple[int, int] = (
    TARGET_LO_MIDI - PIANO_LO_MIDI,
    TARGET_HI_MIDI - PIANO_LO_MIDI + 1,
)


# ── Time / audio resolution ───────────────────────────────────────────────────

FRAME_RATE_HZ = 100             # piano-roll frames per second  (10 ms / frame)
SAMPLE_RATE   = 22050           # mel-spectrogram audio sample rate
HOP_LENGTH    = SAMPLE_RATE // FRAME_RATE_HZ   # = 220 — keeps mel frames aligned

# Mel spectrogram parameters
N_MELS   = 128
N_FFT    = 2048
F_MIN_HZ = 27.5                 # below A0 — covers full piano
F_MAX_HZ = 8000.0               # well above C8 harmonics we care about

# Frequency-cropping band for the 5-octave target keyboard (C3 ≈ 130 Hz, C8 ≈ 4186 Hz)
TARGET_F_LO_HZ = 130.0
TARGET_F_HI_HZ = 4200.0


# ── Generator architecture ────────────────────────────────────────────────────

@dataclass(frozen=True)
class GeneratorConfig:
    in_dim:      int   = PIANO_DIM               # 88 piano keys
    out_dim:     int   = TARGET_DIM              # 61 keys
    cond_dim:    int   = 12 + 12                 # chroma (12) + chord-quality one-hot (12)
    d_model:     int   = 256
    n_heads:     int   = 8
    n_layers:    int   = 6
    ff_dim:      int   = 1024
    dropout:     float = 0.1
    max_seq_len: int   = 4096                    # ~40 s at 100 Hz
    noise_scale: float = 0.10                    # σ of conditioned noise injection


# ── Loss weights and training schedule ────────────────────────────────────────

@dataclass(frozen=True)
class LossWeights:
    """
    Loss weights from the architecture spec:
        L = α · spectral + β · rhythm + γ · contour + δ · sparsity + ε · tempo
    """
    spectral:  float = 1.0    # α — JS divergence on cropped mel
    rhythm:    float = 0.5    # β — onset-density MSE per beat
    contour:   float = 0.3    # γ — melodic-contour cosine distance
    sparsity:  float = 0.05   # δ — >10 simultaneous keys
    tempo:     float = 0.4    # ε — implied-BPM deviation


@dataclass(frozen=True)
class WeightSchedule:
    """
    Schedule for ramping in loss terms during training.

    The spec says: start with α only, ramp β/γ/ε in as training stabilises,
    and keep δ active from the start to prevent dense-activation mode collapse.
    """
    warmup_steps:        int = 2_000     # spectral + sparsity only
    rhythm_intro_step:   int = 4_000
    contour_intro_step:  int = 6_000
    tempo_intro_step:    int = 8_000
    fully_active_step:   int = 12_000


# ── Sparsity constraint ───────────────────────────────────────────────────────

MAX_SIMULTANEOUS_KEYS = 10                       # ≈ what two human hands can play
SPARSITY_BINARIZATION_THRESHOLD = 0.5            # for inference only


# ── Optimization ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TrainConfig:
    batch_size:      int   = 8
    segment_seconds: float = 4.0                  # one training example = 4 s of music
    learning_rate:   float = 2e-4
    weight_decay:    float = 1e-2
    grad_clip:       float = 1.0
    total_steps:     int   = 50_000
    eval_every:      int   = 1_000
    checkpoint_every:int   = 2_000
    log_every:       int   = 50
    seed:            int   = 1729


# ── Paths ─────────────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parents[3]   # …/Arrangement_Transformer

@dataclass(frozen=True)
class Paths:
    maestro_root:   Path = _REPO_ROOT / "data" / "maestro-v3.0.0"
    checkpoint_dir: Path = _REPO_ROOT / "checkpoints"
    log_dir:        Path = _REPO_ROOT / "training_logs"
    default_ckpt:   Path = _REPO_ROOT / "checkpoints" / "generator_latest.pt"


# ── Instances exposed for convenient import ───────────────────────────────────

GENERATOR   = GeneratorConfig()
LOSS_WEIGHT = LossWeights()
SCHEDULE    = WeightSchedule()
TRAIN       = TrainConfig()
PATHS       = Paths()


# ── Inference toggle ──────────────────────────────────────────────────────────
# When True, /transform uses the trained generator for the 88→61 transformation.
# The 88-key transcription itself is always produced by the rule-based pipeline
# (pYIN + Basic Pitch in services/transcription.py) regardless of this flag.
# Falls back gracefully to the rule-based 88→61 path if either the flag is
# False OR no checkpoint is found at PATHS.default_ckpt.
USE_ML_PIPELINE = True
