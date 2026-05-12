"""
Assemble the five loss terms into a single trainable objective, applying the
weight schedule from config.

Returns both the scalar total (for backprop) and a structured breakdown
(for logging / debugging).  Keeping the breakdown live during training is
essential — when the network plateaus you almost always need to see which
of the five terms is fighting which.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

import torch
import torch.nn as nn

from app.ml.config import (
    LOSS_WEIGHT, SCHEDULE,
    PIANO_LO_MIDI, PIANO_DIM, TARGET_SLICE,
    TARGET_F_LO_HZ, TARGET_F_HI_HZ,
)
from app.ml.models.spectrogram import PianoRollSpectrogram, crop_mel_to_band

from .spectral import spectral_js_loss
from .rhythm   import rhythmic_envelope_loss
from .contour  import melodic_contour_loss
from .sparsity import sparsity_loss
from .tempo    import tempo_consistency_loss


@dataclass
class LossBreakdown:
    total:    float
    spectral: float
    rhythm:   float
    contour:  float
    sparsity: float
    tempo:    float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def _schedule_factor(name: str, step: int) -> float:
    """
    Linearly ramp a term in over a short window starting at its intro_step.
    Returns a multiplier in [0, 1].
    """
    intro = {
        "spectral": 0,                              # active from step 0
        "sparsity": 0,                              # active from step 0
        "rhythm":   SCHEDULE.rhythm_intro_step,
        "contour":  SCHEDULE.contour_intro_step,
        "tempo":    SCHEDULE.tempo_intro_step,
    }[name]
    if step < intro:
        return 0.0
    ramp_end = min(intro + 1_000, SCHEDULE.fully_active_step)
    if step >= ramp_end:
        return 1.0
    return (step - intro) / max(1, ramp_end - intro)


class CompositeLoss(nn.Module):
    """
    Forward computes all five loss terms plus the weighted total.

    Owns one PianoRollSpectrogram instance for the 88-dim target (it's
    expensive to build and we always crop to the same band, so reusing
    saves both memory and time).
    """

    def __init__(self) -> None:
        super().__init__()
        # One synthesizer covers both 88- and 61-key roll rendering since
        # the harmonic matrix is keyed on absolute MIDI pitch.  We render
        # the 61-dim output by zero-padding it back to 88 dims at the
        # correct slice — that way the same module produces comparable
        # spectrograms in the same frequency basis.
        self.spec = PianoRollSpectrogram()

    def _to_88dim(self, roll_61: torch.Tensor) -> torch.Tensor:
        """
        Zero-pad a (B, T, 61) roll back to (B, T, 88) at the right pitch slice
        so the same spectrogram synthesizer can render it on the shared mel
        basis as the original.
        """
        B, T, _ = roll_61.shape
        out = roll_61.new_zeros(B, T, PIANO_DIM)
        lo, hi = TARGET_SLICE
        out[:, :, lo:hi] = roll_61
        return out

    def forward(
        self,
        roll_target_88: torch.Tensor,
        roll_output_61: torch.Tensor,
        step: int = 0,
    ) -> tuple[torch.Tensor, LossBreakdown]:
        # Render both rolls into the same mel basis, then crop to the band
        # the 5-octave keyboard can produce.
        mel_t = self.spec(roll_target_88)
        mel_o = self.spec(self._to_88dim(roll_output_61))
        mel_t_band = crop_mel_to_band(mel_t, self.spec.mel_centre_hz,
                                      TARGET_F_LO_HZ, TARGET_F_HI_HZ)
        mel_o_band = crop_mel_to_band(mel_o, self.spec.mel_centre_hz,
                                      TARGET_F_LO_HZ, TARGET_F_HI_HZ)

        L_spec = spectral_js_loss(mel_t_band, mel_o_band)
        L_rhy  = rhythmic_envelope_loss(roll_target_88, roll_output_61)
        L_con  = melodic_contour_loss(roll_target_88, roll_output_61)
        L_spa  = sparsity_loss(roll_output_61)
        L_tem  = tempo_consistency_loss(roll_target_88, roll_output_61)

        # Schedule factors
        f_spec = _schedule_factor("spectral", step)
        f_rhy  = _schedule_factor("rhythm",   step)
        f_con  = _schedule_factor("contour",  step)
        f_spa  = _schedule_factor("sparsity", step)
        f_tem  = _schedule_factor("tempo",    step)

        total = (
            LOSS_WEIGHT.spectral * f_spec * L_spec
          + LOSS_WEIGHT.rhythm   * f_rhy  * L_rhy
          + LOSS_WEIGHT.contour  * f_con  * L_con
          + LOSS_WEIGHT.sparsity * f_spa  * L_spa
          + LOSS_WEIGHT.tempo    * f_tem  * L_tem
        )

        breakdown = LossBreakdown(
            total    = float(total.detach().cpu()),
            spectral = float(L_spec.detach().cpu()),
            rhythm   = float(L_rhy.detach().cpu()),
            contour  = float(L_con.detach().cpu()),
            sparsity = float(L_spa.detach().cpu()),
            tempo    = float(L_tem.detach().cpu()),
        )
        return total, breakdown
