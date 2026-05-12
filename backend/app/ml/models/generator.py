"""
ArrangementTransformer — encoder-only Transformer that maps an 88-dim piano
roll to a 61-dim piano roll, conditioned on per-frame scale/chord context.

Why encoder-only (no separate decoder)
--------------------------------------
The input and output sequences have the same time dimension — every frame of
the source maps to one frame of the target.  A full encoder-decoder is the
right tool when input and output sequences have *different* lengths and the
decoder must consult the encoder cross-attention-wise (translation, summary
generation, etc.).  Here, an encoder stack with a linear output head is
strictly more parameter-efficient and easier to train: self-attention over
each timestep already learns "which input keys, at which time steps, are
relevant to the output key I'm currently emitting", which is exactly the
priority hierarchy the spec describes.

Inputs
------
    roll        (B, T, 88)   — soft activations in [0, 1]
    cond        (B, T, C)    — chroma(12) + chord-quality(12) per frame
    noise       (B, T, 88)   — scale/chord-conditioned noise (already biased)

Output
------
    out         (B, T, 61)   — sigmoid activations in [0, 1]
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn

from app.ml.config import GENERATOR, PIANO_DIM, TARGET_DIM


class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard non-learned positional encoding.  Saved as buffer; no params.
    Truncates/expands to the input length at forward time.
    """

    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        return x + self.pe[:, : x.shape[1], :]


class ArrangementTransformer(nn.Module):
    """
    Conditional generator.  See module docstring for IO shapes.
    """

    def __init__(self, cfg=GENERATOR) -> None:
        super().__init__()
        self.cfg = cfg

        # Input projection: concatenate roll + noise + cond, then project to d_model.
        # Total input width = in_dim (88) + in_dim (88 noise) + cond_dim.
        in_total = cfg.in_dim + cfg.in_dim + cfg.cond_dim
        self.input_proj = nn.Linear(in_total, cfg.d_model)
        self.input_norm = nn.LayerNorm(cfg.d_model)

        self.pos_enc = SinusoidalPositionalEncoding(cfg.d_model, cfg.max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = cfg.d_model,
            nhead           = cfg.n_heads,
            dim_feedforward = cfg.ff_dim,
            dropout         = cfg.dropout,
            batch_first     = True,
            norm_first      = True,           # pre-LN — more stable for deep stacks
            activation      = "gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)

        # Output head: project to 61 keys, sigmoid for soft activations.
        self.out_norm = nn.LayerNorm(cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.out_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        # Xavier on the projections, small init on the output bias so the
        # initial sigmoid output is near 0.5 (random; not collapsed to 0 or 1).
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        roll:  torch.Tensor,
        cond:  torch.Tensor,
        noise: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        roll:  (B, T, 88)   piano roll, float in [0, 1]
        cond:  (B, T, C)    chroma + chord quality
        noise: (B, T, 88)   scale/chord-conditioned noise (same shape as roll)
        src_key_padding_mask: (B, T) bool  — True = padding (ignored by attn)

        Returns out: (B, T, 61) soft activations
        """
        # Concatenate along feature dim so the encoder sees roll + noise + cond
        # as one fused input per time step.  Adding noise (instead of concat)
        # would mean the model can't distinguish signal from the noise prior —
        # we want it to learn that distinction.
        x = torch.cat([roll, noise, cond], dim=-1)            # (B, T, in_total)

        x = self.input_proj(x)
        x = self.input_norm(x)
        x = self.pos_enc(x)

        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = self.out_norm(x)

        logits = self.out_proj(x)                              # (B, T, 61)
        return torch.sigmoid(logits)

    @torch.no_grad()
    def generate(
        self,
        roll:  torch.Tensor,
        cond:  torch.Tensor,
        noise: torch.Tensor | None = None,
        binarize_threshold: float = 0.5,
    ) -> torch.Tensor:
        """
        Inference path — same as forward but optionally binarises the output
        for direct conversion to a hard piano roll.  When noise is None, we
        use zeros (deterministic generation; the spec says noise is a soft
        prior during training, not strictly required at inference time).
        """
        self.eval()
        if noise is None:
            noise = torch.zeros_like(roll)
        out = self.forward(roll, cond, noise)
        if binarize_threshold is not None:
            out = (out >= binarize_threshold).float()
        return out
