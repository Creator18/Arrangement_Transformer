"""Audio -> pitch-aligned spectral patches, computed on the GPU.

The idea this serves: tokenize audio the way a VLM patchifies an image, but
on a spectrum whose vertical axis IS pitch, so register and interval become
positions a model can learn rules about — the way an LLM learns grammar from
word order.  General audio tokenizers do not do this.  Codec tokens entangle
pitch, timbre and loudness across a stack of codebooks; mel patches use an
axis on which one patch spans two octaves in the bass and a few semitones in
the treble.  Probing work (SynTheory, ISMIR 2024) found plain 12-bin chroma
beating MusicGen's decoder at naming chords, i.e. generic tokens bury exactly
the structure we want.

Three stages, each fixing one thing:

1. HCQT — a constant-Q transform per harmonic.  36 bins per octave from A0,
   offset so every piano key owns three bins and sits on the middle one:
   MIDI p -> bin 3(p-21)+1, A4 -> 145.  A log axis makes transposition a
   vertical SHIFT, so a chord shape learned once is recognised at any height.
   Absolute frequency is kept, not discarded: bin k is always the same Hz.

   One channel per harmonic h in HARMONICS, each channel computed at f*h for
   the SAME bin grid.  A note's overtones then land at the same position in
   every channel instead of at +12, +19, +24 semitones.  This is the
   "harmonic catch" handled by construction rather than left for attention
   to learn: a bass note's 2nd harmonic sits exactly where a melody note an
   octave up would, and stacking harmonics as channels is how deep-salience
   melody extraction (Bittner et al.) separates them.

2. Loudness normalisation — log compression, then per-band standardisation
   against corpus statistics.  Low frequencies carry far more energy than
   high ones by physics; without this the model learns "the bass is loud".
   Per-CLIP normalisation would be simpler and wrong: it erases dynamics,
   which the arrangement needs.

3. Onset stream — spectral flux from a SHORT-window STFT mapped onto the same
   pitch axis.  A constant-Q transform needs long windows in the bass to
   resolve semitones, which smears bass onsets in time; the groove lives
   there.  This channel has blurry pitch and sharp time, the opposite trade,
   and the model gets both.

Frame grid is SR 24000 / hop 384 = 62.5 fps (16 ms), the same rate as the
transcriber, so labels line up with existing tooling.  Frame count is ALWAYS
n // HOP + 1, enforced explicitly — an off-by-one frame between two streams
is how this project once drifted 496 ms across a track.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SR = 24000                      # Nyquist 12 kHz: room for 5th harmonics of mid notes
HOP = 384                       # 16 ms
FPS = SR / HOP                  # 62.5
BINS_PER_OCT = 36               # 3 per semitone: enough to read tuning offsets
N_KEYS = 88
N_BINS = N_KEYS * 3             # 264
# Centre key k on bin 3k+1, so A0 (key 0) is bin 1 and each key owns 3 bins.
FMIN = 27.5 * 2 ** (-1 / BINS_PER_OCT)
HARMONICS = (0.5, 1, 2, 3, 4, 5)

# Longest analysis window.  A true constant-Q kernel at A0 would be ~1.9 s.
# Capping it trades bass pitch resolution in the FUNDAMENTAL channel for time
# resolution; the higher harmonic channels resolve bass pitch instead, which
# is the second reason the transform is harmonic.  Measured in
# eval/spectral_frontend_check.py rather than assumed.
MAX_WIN_S = 0.25

ONSET_NFFT = 1024               # 43 ms: sharp in time, coarse in pitch
LOG_GAIN = 100.0

# Patches: one octave tall, 32 ms wide, rows overlapping by half an octave so
# a chord on a patch boundary is never split across two tokens.
PATCH_H, PATCH_H_STEP = 36, 18
PATCH_W, PATCH_W_STEP = 2, 2


def bin_freqs() -> np.ndarray:
    return FMIN * 2.0 ** (np.arange(N_BINS) / BINS_PER_OCT)


def midi_to_bin(p: int) -> int:
    return 3 * (int(p) - 21) + 1


def n_frames(n_samples: int) -> int:
    return n_samples // HOP + 1


class HCQT(nn.Module):
    """Harmonic constant-Q magnitude: (B, n) -> (B, len(HARMONICS), N_BINS, T)."""

    def __init__(self):
        super().__init__()
        q = 1.0 / (2 ** (1.0 / BINS_PER_OCT) - 1.0)
        L = int(MAX_WIN_S * SR) | 1                     # odd, so kernels centre
        self.L = L
        kr, ki, live = [], [], []
        fb = bin_freqs()
        for h in HARMONICS:
            for f0 in fb:
                f = f0 * h
                if f >= SR / 2 * 0.95:                  # above Nyquist: dead bin
                    kr.append(np.zeros(L)); ki.append(np.zeros(L)); live.append(0.0)
                    continue
                n = min(int(math.ceil(q * SR / f)), L)
                w = np.hanning(n)
                t = (np.arange(n) - (n - 1) / 2) / SR
                # normalise by window SUM so a unit sine reads ~0.5 at every
                # bin regardless of window length
                c = w * np.exp(-2j * np.pi * f * t) / w.sum()
                pad = (L - n) // 2
                full = np.zeros(L, dtype=np.complex128)
                full[pad:pad + n] = c
                kr.append(full.real); ki.append(full.imag); live.append(1.0)
        k = np.stack(kr + ki)[:, None, :].astype(np.float32)
        self.register_buffer("kernel", torch.from_numpy(k))
        self.register_buffer("live", torch.tensor(live).view(len(HARMONICS), N_BINS, 1))
        self.nk = len(kr)

        # Frequency-domain form of the same kernels, for the fast path.
        #
        # Direct convolution pads every kernel to the LONGEST (bass) window,
        # so a treble bin needing ~100 taps pays for 6,001 — measured at 281 ms
        # per batch of 8 x 8 s, nearly half a training step.  Correlating a
        # frame with a kernel equals a dot product of their spectra
        # (Parseval), so the whole transform becomes one FFT per frame and one
        # matrix multiply, which is what GPUs are fastest at.  The kernels are
        # identical; only the arithmetic route changes, and
        # eval/spectral_frontend_check.py checks the two agree.
        self.nfft = 1 << (L - 1).bit_length()
        kt = np.zeros((self.nk, self.nfft), dtype=np.complex128)
        kc = np.stack(kr) + 1j * np.stack(ki)            # (nk, L), centred at L//2
        # torch.stft(center=True) frames are centred at index N/2, so the
        # kernel's centre goes there too.
        s0 = self.nfft // 2 - L // 2
        kt[:, s0:s0 + L] = kc
        # sum_n frame[n] k[n]  ==  (1/N) sum_f X[f] * conj(FFT(conj k))[f]
        Kf = np.conj(np.fft.fft(np.conj(kt), axis=1)) / self.nfft
        self.register_buffer("Kf_re", torch.from_numpy(Kf.real.T.astype(np.float32)))
        self.register_buffer("Kf_im", torch.from_numpy(Kf.imag.T.astype(np.float32)))

    def forward(self, wav: torch.Tensor, method: str = "fft") -> torch.Tensor:
        n = wav.shape[-1]
        T = n_frames(n)
        if method == "conv":                             # reference path
            x = F.pad(wav.unsqueeze(1), (self.L // 2, self.L // 2 + HOP))
            y = F.conv1d(x, self.kernel, stride=HOP)[..., :T]
            re, im = y[:, :self.nk], y[:, self.nk:]
        else:
            # frames centred on t*HOP, rectangular window (kernels carry their
            # own Hann), full two-sided spectrum because the kernels are complex
            X = torch.stft(wav, self.nfft, HOP, window=torch.ones(self.nfft, device=wav.device),
                           center=True, pad_mode="constant", onesided=False,
                           return_complex=True)[..., :T]          # (B, N, T)
            Xr, Xi = X.real.transpose(1, 2), X.imag.transpose(1, 2)  # (B, T, N)
            re = (Xr @ self.Kf_re - Xi @ self.Kf_im).transpose(1, 2)
            im = (Xr @ self.Kf_im + Xi @ self.Kf_re).transpose(1, 2)
        mag = torch.sqrt(re ** 2 + im ** 2 + 1e-12)
        return mag.view(wav.shape[0], len(HARMONICS), N_BINS, T) * self.live


class OnsetStream(nn.Module):
    """Short-window spectral flux on the HCQT pitch axis: (B, n) -> (B, 1, N_BINS, T)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("win", torch.hann_window(ONSET_NFFT))
        fft_f = np.fft.rfftfreq(ONSET_NFFT, 1 / SR)
        # linear interpolation matrix from STFT bins onto the CQT centre freqs
        fb = bin_freqs()
        M = np.zeros((N_BINS, len(fft_f)), dtype=np.float32)
        for i, f in enumerate(fb):
            j = np.searchsorted(fft_f, f)
            if 0 < j < len(fft_f):
                a = (f - fft_f[j - 1]) / (fft_f[j] - fft_f[j - 1])
                M[i, j - 1], M[i, j] = 1 - a, a
        self.register_buffer("interp", torch.from_numpy(M))

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        T = n_frames(wav.shape[-1])
        S = torch.stft(wav, ONSET_NFFT, HOP, window=self.win, center=True,
                       return_complex=True).abs()[..., :T]       # (B, F, T)
        L = torch.log1p(LOG_GAIN * (self.interp @ S))
        flux = F.relu(L[..., 1:] - L[..., :-1])
        flux = F.pad(flux, (1, 0))
        return flux.unsqueeze(1)


class SpectralFrontend(nn.Module):
    """Audio -> normalised (B, C, N_BINS, T) features, C = harmonics + onset."""

    def __init__(self):
        super().__init__()
        self.hcqt = HCQT()
        self.onset = OnsetStream()
        C = len(HARMONICS) + 1
        self.register_buffer("mu", torch.zeros(1, C, N_BINS, 1))
        self.register_buffer("sd", torch.ones(1, C, N_BINS, 1))
        self.register_buffer("calibrated", torch.tensor(False))

    @property
    def channels(self) -> int:
        return len(HARMONICS) + 1

    def raw(self, wav: torch.Tensor) -> torch.Tensor:
        h = torch.log1p(LOG_GAIN * self.hcqt(wav))
        return torch.cat([h, self.onset(wav)], dim=1)

    @torch.no_grad()
    def calibrate(self, wavs) -> None:
        """Per-band mean/std over a corpus sample — NOT per clip (see module doc)."""
        s = s2 = n = 0
        for w in wavs:
            x = self.raw(w)
            s = s + x.sum(dim=(0, 3), keepdim=True)
            s2 = s2 + (x ** 2).sum(dim=(0, 3), keepdim=True)
            n += x.shape[0] * x.shape[-1]
        mu = s / n
        self.mu.copy_(mu)
        self.sd.copy_(torch.sqrt((s2 / n - mu ** 2).clamp(min=1e-6)))
        self.calibrated.fill_(True)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return (self.raw(wav) - self.mu) / self.sd


def patchify(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(B, C, N_BINS, T) -> tokens (B, N, C*PATCH_H*PATCH_W), coords (N, 2).

    coords[:, 0] is the patch's CENTRE BIN (absolute pitch, for register-aware
    position embeddings); coords[:, 1] is its first frame (time).  Relative
    pitch is the difference of the first column, so both kinds of position
    the model needs come from one tensor.
    """
    B, C, Fb, T = x.shape
    rows = math.ceil((Fb - PATCH_H) / PATCH_H_STEP) + 1
    Fp = (rows - 1) * PATCH_H_STEP + PATCH_H
    Tp = math.ceil(T / PATCH_W) * PATCH_W
    x = F.pad(x, (0, Tp - T, 0, Fp - Fb))
    p = x.unfold(2, PATCH_H, PATCH_H_STEP).unfold(3, PATCH_W, PATCH_W_STEP)
    # p: (B, C, rows, cols, PATCH_H, PATCH_W)
    cols = p.shape[3]
    tok = p.permute(0, 2, 3, 1, 4, 5).reshape(B, rows * cols, C * PATCH_H * PATCH_W)
    r = torch.arange(rows).repeat_interleave(cols)
    c = torch.arange(cols).repeat(rows)
    coords = torch.stack([r * PATCH_H_STEP + PATCH_H // 2, c * PATCH_W], dim=1)
    return tok, coords
