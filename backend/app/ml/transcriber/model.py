"""Onsets-and-Frames shaped transcriber, sized for 8 GB.

Two heads, because onset and sustain are different questions and the pipeline
has been failing at both in different ways.  ONSET answers "was a key struck
here", which the NMF cannot answer well: its 371 ms analysis window smears
every attack, and 82% of the chords it produced were spread wider than 20 ms.
FRAME answers "is this pitch sounding", which is closer to what the activation
matrix already gives.

The frame head is conditioned on the onset head's output.  A note exists
because it was STRUCK — sustain without an attack is almost always a harmonic
of something else, which is precisely the ghost the harmonic dictionary
produces (mean cos(p, p+12) = 0.446).  Feeding onsets into the frame decision
lets the model use that reasoning instead of us hand-coding it as a filter,
which is what failed by ear three times over.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from . import config as C


class ConvStack(nn.Module):
    """Mel -> per-frame features.  Frequency is pooled, time never is."""

    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            # (1, 2): pool frequency only.  Pooling time would throw away the
            # onset resolution this model exists to provide.
            nn.MaxPool2d((1, 2)), nn.Dropout(0.25),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d((1, 2)), nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * (C.N_MELS // 4), out_dim), nn.Dropout(0.5), nn.ReLU())

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel (B, n_mels, T) -> (B, T, out_dim)
        x = mel.unsqueeze(1).transpose(2, 3)          # (B, 1, T, n_mels)
        x = self.net(x)                               # (B, 64, T, n_mels/4)
        b, c, t, f = x.shape
        return self.fc(x.permute(0, 2, 1, 3).reshape(b, t, c * f))


class Head(nn.Module):
    """Recurrent stack + per-pitch logits.  Bidirectional on purpose.

    Offline transcription has the whole segment available, and the end of a
    note is strong evidence about its beginning — the decay shape says whether
    a key was struck or merely rang on from something else.  A causal model
    would forfeit that.
    """

    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.rnn = nn.LSTM(in_dim, hidden, batch_first=True, bidirectional=True)
        self.out = nn.Linear(hidden * 2, C.N_PITCH)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn(x)
        return self.out(h)                            # (B, T, 88) logits


class Transcriber(nn.Module):
    def __init__(self, feat: int = 256, hidden: int = 128):
        super().__init__()
        self.conv_on = ConvStack(feat)
        self.conv_fr = ConvStack(feat)
        self.onset = Head(feat, hidden)
        # frame head sees the mel features AND the onset head's belief
        self.frame = Head(feat + C.N_PITCH, hidden)
        # Velocity is a per-pitch regression off the onset features: how hard a
        # key was struck is a property of its attack, so it shares the branch
        # that models attacks rather than the one that models sustain.
        self.vel = nn.Linear(feat, C.N_PITCH)

    def forward(self, mel: torch.Tensor):
        f_on = self.conv_on(mel)
        on_logit = self.onset(f_on)
        vel = self.vel(f_on)
        fr_in = torch.cat([self.conv_fr(mel), torch.sigmoid(on_logit).detach()],
                          dim=-1)
        # detached: the frame head may READ the onset belief, but its gradient
        # must not reshape the onset head to whatever makes sustain easiest.
        # Onset timing is the thing this model exists to get right.
        return on_logit, self.frame(fr_in), vel

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
