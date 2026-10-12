"""Patch transformer over the pitch-aligned spectrum: audio features -> notes.

Input is the SpectralFrontend's (B, 7, 264, T) grid — six harmonic channels
and an onset stream over 264 pitch bins at 16 ms.  Output is per frame and per
key: was it struck, is it sounding, how hard — exactly the transcriber's
(B, T, 88) format, so its note decoder and mir_eval scoring are reused
unchanged rather than rewritten.

Every structural choice traces to a Phase 2 probe (eval/phase2_probes.py):

STEM — a small 2-D convolution BEFORE patching.  Two probes asked for it.
  Linear probes ranked HCQT below plain CQT: combining evidence across
  harmonic channels is not linear ("this AND its overtones"), so channels are
  mixed by a nonlinear layer first.  And patches step half an octave, so a
  transposition by a semitone moves a chord INSIDE its patch and changes the
  token.  A convolution with weights shared across pitch is equivariant at
  every bin, so semitone shifts are handled before the coarse patch grid.

PATCHES — 1 octave x 32 ms, rows overlapping by half an octave, projected by
  one learned matrix shared by every patch: the "pixels -> token" step, with
  the matrix learned in training rather than designed.

FACTORISED ATTENTION — alternating along TIME within a pitch row (rhythm,
  sustain, decay) and across PITCH within a 32 ms column (chords, voicing,
  and the overtone stacks that sound at the same moment).  Full attention over
  ~440 tokens per second would be ~12M pairs per 8 s clip; factorised it is a
  few hundred thousand.

TWO PITCH POSITIONS — a learned ABSOLUTE embedding per row (register: melody
  on top, bass below) and a learned RELATIVE bias inside pitch attention (an
  interval means the same at any height).  The transposition probe showed a
  flat, position-specific model collapsing to 0.21 on unseen registers while
  the same features with weights shared across pitch transferred at 0.81.
  Time uses rotary positions, which are relative and so independent of where
  a crop happens to start.

SKIP TO THE OUTPUT — patches are 32 ms wide but onsets need 16 ms.  The
  stem's full-resolution features bypass the attention stack and rejoin it at
  the head, so the encoder supplies context and the skip supplies precision.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import frontend as S

PATCH_H, PATCH_H_STEP = S.PATCH_H, S.PATCH_H_STEP        # 36 bins, step 18
PATCH_W = S.PATCH_W                                      # 2 frames
ROWS = math.ceil((S.N_BINS - PATCH_H) / PATCH_H_STEP) + 1  # 14
F_PAD = (ROWS - 1) * PATCH_H_STEP + PATCH_H             # 270


def rotary(x: torch.Tensor) -> torch.Tensor:
    """Rotary position embedding along dim -2 of (..., N, head_dim)."""
    n, d = x.shape[-2], x.shape[-1]
    half = d // 2
    freq = 1.0 / (10000 ** (torch.arange(half, device=x.device, dtype=torch.float32) / half))
    ang = torch.arange(n, device=x.device, dtype=torch.float32)[:, None] * freq[None]
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    a, b = x[..., :half], x[..., half:2 * half]
    out = torch.cat([a * cos - b * sin, a * sin + b * cos], dim=-1)
    return torch.cat([out, x[..., 2 * half:]], dim=-1) if d > 2 * half else out


class Attention(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.h = heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)

    def forward(self, x, rope: bool = False, bias=None):
        B, N, D = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        if rope:
            q, k = rotary(q), rotary(k)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        return self.out(y.transpose(1, 2).reshape(B, N, D))


class Block(nn.Module):
    """Time attention per pitch row, then pitch attention per time column."""

    def __init__(self, d: int, heads: int, ff: int = 4):
        super().__init__()
        self.n1, self.time = nn.LayerNorm(d), Attention(d, heads)
        self.n2, self.pitch = nn.LayerNorm(d), Attention(d, heads)
        self.n3 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff * d), nn.GELU(), nn.Linear(ff * d, d))
        # relative pitch bias: one learned value per (row distance, head)
        self.rel = nn.Parameter(torch.zeros(heads, 2 * ROWS - 1))
        idx = torch.arange(ROWS)
        self.register_buffer("rel_idx", (idx[:, None] - idx[None] + ROWS - 1), persistent=False)

    def forward(self, x):                                  # (B, R, N, D)
        B, R, N, D = x.shape
        t = x.reshape(B * R, N, D)
        t = t + self.time(self.n1(t), rope=True)
        x = t.view(B, R, N, D)
        p = x.transpose(1, 2).reshape(B * N, R, D)
        bias = self.rel[:, self.rel_idx].unsqueeze(0).to(p.dtype)     # (1, h, R, R)
        p = p + self.pitch(self.n2(p), bias=bias)
        p = p + self.ff(self.n3(p))
        return p.view(B, N, R, D).transpose(1, 2)


class SpectralTranscriber(nn.Module):
    def __init__(self, d: int = 256, depth: int = 6, heads: int = 4, stem: int = 32,
                 n_cond: int = 0, roles: bool = False):
        super().__init__()
        C = len(S.HARMONICS) + 1
        self.arch = {"d": d, "depth": depth, "heads": heads, "stem": stem}
        if n_cond:
            self.arch["n_cond"] = n_cond
        if roles:
            self.arch["roles"] = True
        self.stem = nn.Sequential(
            nn.Conv2d(C, stem, 3, padding=1), nn.GELU(),
            nn.Conv2d(stem, stem, 3, padding=1), nn.GELU())
        # the learned patch projection: one matrix applied to every patch
        self.patch = nn.Conv2d(stem, d, (PATCH_H, PATCH_W), stride=(PATCH_H_STEP, PATCH_W))
        self.pitch_emb = nn.Parameter(torch.zeros(1, ROWS, 1, d))      # absolute register
        nn.init.normal_(self.pitch_emb, std=0.02)
        # Optional conditioning label (song-to-piano: arrangement difficulty),
        # added to every token.  Zero-initialised, so a model fine-tuned from a
        # transcription checkpoint starts out exactly as that checkpoint.
        self.cond_emb = nn.Embedding(n_cond, d) if n_cond else None
        if n_cond:
            nn.init.zeros_(self.cond_emb.weight)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(d)
        # back to the full grid: overlapping rows sum, as patches overlapped
        self.unpatch = nn.ConvTranspose2d(d, stem, (PATCH_H, PATCH_W),
                                          stride=(PATCH_H_STEP, PATCH_W))
        self.fuse = nn.Sequential(nn.Conv2d(2 * stem, stem, 3, padding=1), nn.GELU())
        # 3 bins per key -> one value per key, three outputs (onset/frame/velocity)
        self.head = nn.Conv2d(stem, 3, (3, 1), stride=(3, 1))
        # Optional role outputs (curriculum training): is this key, at this
        # frame, part of the MELODY / the BASS line.  A separate head so a
        # transcription checkpoint loads unchanged; zero-initialised weights
        # with a negative bias start it at "not melody, not bass" everywhere.
        self.role_head = nn.Conv2d(stem, 2, (3, 1), stride=(3, 1)) if roles else None
        if roles:
            nn.init.zeros_(self.role_head.weight); nn.init.constant_(self.role_head.bias, -4.0)

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None):
        """(B, 7, 264, T) features -> onset, frame, velocity logits, each (B, T, 88).

        cond: (B,) integer labels, used only by a model built with n_cond.
        A model built with roles=True returns two more: melody, bass logits."""
        B, _, Fb, T = x.shape
        s = self.stem(x)                                               # (B, c, 264, T)
        Tp = math.ceil(T / PATCH_W) * PATCH_W
        z = self.patch(F.pad(s, (0, Tp - T, 0, F_PAD - Fb)))           # (B, d, R, N)
        z = z.permute(0, 2, 3, 1) + self.pitch_emb                     # (B, R, N, d)
        if self.cond_emb is not None and cond is not None:
            z = z + self.cond_emb(cond.long())[:, None, None, :]
        for blk in self.blocks:
            z = blk(z)
        z = self.norm(z).permute(0, 3, 1, 2)                           # (B, d, R, N)
        u = self.unpatch(z)[:, :, :Fb, :T]                             # (B, c, 264, T)
        h = self.fuse(torch.cat([u, s], dim=1))
        y = self.head(h).permute(1, 0, 3, 2)                           # (3, B, T, 88)
        if self.role_head is None:
            return y[0], y[1], y[2]
        r = self.role_head(h).permute(1, 0, 3, 2)                      # (2, B, T, 88)
        return y[0], y[1], y[2], r[0], r[1]

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def from_checkpoint(ck: dict) -> SpectralTranscriber:
    net = SpectralTranscriber(**ck["arch"])
    net.load_state_dict(ck["model"])
    return net
