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
    """Time attention per pitch row, then pitch attention per time column.

    time_win sets how far along time a token may look (tokens are 32 ms): 0 = the whole window
    (global), w > 0 = only +-w tokens (hard local window), -1 = a learned soft window per head (a
    penalty growing with distance, its steepness learned).  Pitch attention is always full."""

    def __init__(self, d: int, heads: int, ff: int = 4, time_win: int = 0, rows: int = ROWS):
        super().__init__()
        self.time_win = time_win
        if time_win == -1:
            # start the heads at different widths (half-weight at ~2, 6, 12, 24 tokens), then learn them
            half = torch.tensor([2.0, 6.0, 12.0, 24.0][:heads] + [24.0] * max(0, heads - 4))
            slope = math.log(2.0) / half
            self.time_slope = nn.Parameter(torch.log(torch.expm1(slope)))   # softplus^-1
        self.n1, self.time = nn.LayerNorm(d), Attention(d, heads)
        self.n2, self.pitch = nn.LayerNorm(d), Attention(d, heads)
        self.n3 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff * d), nn.GELU(), nn.Linear(ff * d, d))
        # relative pitch bias: one learned value per (row distance, head)
        self.rel = nn.Parameter(torch.zeros(heads, 2 * rows - 1))
        idx = torch.arange(rows)
        self.register_buffer("rel_idx", (idx[:, None] - idx[None] + rows - 1), persistent=False)

    def time_bias(self, N, device, dtype):
        if self.time_win == 0:
            return None
        i = torch.arange(N, device=device)
        dist = (i[:, None] - i[None]).abs()
        if self.time_win > 0:
            return dist <= self.time_win                                    # bool: True = may attend
        return (-F.softplus(self.time_slope)[:, None, None] * dist.float()).unsqueeze(0).to(dtype)

    def forward(self, x):                                  # (B, R, N, D)
        B, R, N, D = x.shape
        t = x.reshape(B * R, N, D)
        t = t + self.time(self.n1(t), rope=True, bias=self.time_bias(N, x.device, x.dtype))
        x = t.view(B, R, N, D)
        p = x.transpose(1, 2).reshape(B * N, R, D)
        bias = self.rel[:, self.rel_idx].unsqueeze(0).to(p.dtype)     # (1, h, R, R)
        p = p + self.pitch(self.n2(p), bias=bias)
        p = p + self.ff(self.n3(p))
        return p.view(B, N, R, D).transpose(1, 2)


class SpectralTranscriber(nn.Module):
    def __init__(self, d: int = 256, depth: int = 6, heads: int = 4, stem: int = 32,
                 n_cond: int = 0, roles: bool = False, offset: bool = False,
                 time_windows: Optional[list] = None, patch_h: int = PATCH_H):
        super().__init__()
        # token height in pitch bins (3 per semitone): 36 = one octave (default), 18 = half an octave;
        # rows overlap by half a token either way
        self.patch_h, self.patch_step = patch_h, patch_h // 2
        self.rows = math.ceil((S.N_BINS - patch_h) / self.patch_step) + 1
        self.f_pad = (self.rows - 1) * self.patch_step + patch_h
        C = len(S.HARMONICS) + 1
        self.arch = {"d": d, "depth": depth, "heads": heads, "stem": stem}
        if patch_h != PATCH_H:
            self.arch["patch_h"] = patch_h
        if time_windows:                                   # per-layer time attention span (Block.time_win)
            assert len(time_windows) == depth, "one time window per layer"
            self.arch["time_windows"] = list(time_windows)
        if n_cond:
            self.arch["n_cond"] = n_cond
        if roles:
            self.arch["roles"] = True
        if offset:
            self.arch["offset"] = True
        self.stem = nn.Sequential(
            nn.Conv2d(C, stem, 3, padding=1), nn.GELU(),
            nn.Conv2d(stem, stem, 3, padding=1), nn.GELU())
        # the learned patch projection: one matrix applied to every patch
        self.patch = nn.Conv2d(stem, d, (patch_h, PATCH_W), stride=(self.patch_step, PATCH_W))
        self.pitch_emb = nn.Parameter(torch.zeros(1, self.rows, 1, d))      # absolute register
        nn.init.normal_(self.pitch_emb, std=0.02)
        # Optional conditioning label (song-to-piano: arrangement difficulty),
        # added to every token.  Zero-initialised, so a model fine-tuned from a
        # transcription checkpoint starts out exactly as that checkpoint.
        self.cond_emb = nn.Embedding(n_cond, d) if n_cond else None
        if n_cond:
            nn.init.zeros_(self.cond_emb.weight)
        self.blocks = nn.ModuleList(Block(d, heads, time_win=(time_windows or [0] * depth)[i], rows=self.rows) for i in range(depth))
        self.norm = nn.LayerNorm(d)
        # back to the full grid: overlapping rows sum, as patches overlapped
        self.unpatch = nn.ConvTranspose2d(d, stem, (patch_h, PATCH_W),
                                          stride=(self.patch_step, PATCH_W))
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
        # Optional NOTE-END output: "the key goes up here".  Without it a note ends where the
        # frame confidence fades, which on decaying or ringing sounds is early or late.
        self.offset_head = None
        if offset:
            self.add_offset_head()

    def add_offset_head(self):
        """A fresh note-end head (zero weights, "no end" everywhere at the start), so an existing
        checkpoint gains the output without changing anything it already does."""
        stem = self.arch.get("stem", 32)
        self.offset_head = nn.Conv2d(stem, 1, (3, 1), stride=(3, 1))
        nn.init.zeros_(self.offset_head.weight); nn.init.constant_(self.offset_head.bias, -4.0)
        self.arch["offset"] = True
        return self.offset_head

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None):
        """(B, 7, 264, T) features -> onset, frame, velocity logits, each (B, T, 88).

        cond: (B,) integer labels, used only by a model built with n_cond.
        A model built with roles=True returns two more: melody, bass logits."""
        B, _, Fb, T = x.shape
        s = self.stem(x)                                               # (B, c, 264, T)
        Tp = math.ceil(T / PATCH_W) * PATCH_W
        z = self.patch(F.pad(s, (0, Tp - T, 0, self.f_pad - Fb)))           # (B, d, R, N)
        z = z.permute(0, 2, 3, 1) + self.pitch_emb                     # (B, R, N, d)
        if self.cond_emb is not None and cond is not None:
            z = z + self.cond_emb(cond.long())[:, None, None, :]
        for blk in self.blocks:
            z = blk(z)
        z = self.norm(z).permute(0, 3, 1, 2)                           # (B, d, R, N)
        u = self.unpatch(z)[:, :, :Fb, :T]                             # (B, c, 264, T)
        h = self.fuse(torch.cat([u, s], dim=1))
        y = self.head(h).permute(1, 0, 3, 2)                           # (3, B, T, 88)
        out = (y[0], y[1], y[2])
        if self.role_head is not None:
            r = self.role_head(h).permute(1, 0, 3, 2)                  # (2, B, T, 88)
            out = out + (r[0], r[1])
        if self.offset_head is not None:                               # note-end logits, LAST
            out = out + (self.offset_head(h)[:, 0].permute(0, 2, 1),)  # (B, T, 88)
        return out

    @torch.no_grad()
    def features(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """The trunk's final patch tokens, for a downstream model to listen to.

        (B, 7, 264, T) -> (B, N, ROWS // 2 * d): one vector per 32 ms time patch;
        the 14 pitch rows are averaged in adjacent pairs (7 register bands), so
        the features keep WHERE in the register something sounds, compactly.
        """
        B, _, Fb, T = x.shape
        s = self.stem(x)
        Tp = math.ceil(T / PATCH_W) * PATCH_W
        z = self.patch(F.pad(s, (0, Tp - T, 0, self.f_pad - Fb))).permute(0, 2, 3, 1) + self.pitch_emb
        if self.cond_emb is not None and cond is not None:
            z = z + self.cond_emb(cond.long())[:, None, None, :]
        for blk in self.blocks:
            z = blk(z)
        z = self.norm(z)                                               # (B, R, N, d)
        R = z.shape[1] // 2 * 2
        z = z[:, :R].reshape(B, R // 2, 2, z.shape[2], z.shape[3]).mean(2)   # (B, R/2, N, d)
        return z.permute(0, 2, 1, 3).reshape(B, z.shape[2], -1)        # (B, N, R/2 * d)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def from_checkpoint(ck: dict) -> SpectralTranscriber:
    net = SpectralTranscriber(**ck["arch"])
    net.load_state_dict(ck["model"])
    return net
