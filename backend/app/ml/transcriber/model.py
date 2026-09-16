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

    def __init__(self, out_dim: int = None, ch=None):
        super().__init__()
        out_dim = out_dim or C.FEAT
        a, b, c = ch or C.CONV_CH
        self.net = nn.Sequential(
            nn.Conv2d(1, a, 3, padding=1), nn.BatchNorm2d(a), nn.ReLU(),
            nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(),
            # (1, 2): pool frequency only.  Pooling time would throw away the
            # onset resolution this model exists to provide.
            nn.MaxPool2d((1, 2)), nn.Dropout(0.25),
            nn.Conv2d(b, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(),
            nn.MaxPool2d((1, 2)), nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Linear(c * (C.N_MELS // 4), out_dim), nn.Dropout(0.5), nn.ReLU())

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

    def __init__(self, in_dim: int, hidden: int = None, layers: int = None):
        super().__init__()
        hidden = hidden or C.LSTM_HIDDEN
        layers = layers or C.LSTM_LAYERS
        self.rnn = nn.LSTM(in_dim, hidden, num_layers=layers, batch_first=True,
                           bidirectional=True,
                           dropout=0.25 if layers > 1 else 0.0)
        self.out = nn.Linear(hidden * 2, C.N_PITCH)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn(x)
        return self.out(h)                            # (B, T, 88) logits


class Transcriber(nn.Module):
    def __init__(self, feat: int = None, hidden: int = None,
                 ch=None, layers: int = None):
        super().__init__()
        feat = feat or C.FEAT
        self.arch = {"feat": feat, "hidden": hidden or C.LSTM_HIDDEN,
                     "ch": tuple(ch or C.CONV_CH),
                     "layers": layers or C.LSTM_LAYERS}
        self.conv_on = ConvStack(feat, ch)
        self.conv_fr = ConvStack(feat, ch)
        self.onset = Head(feat, hidden, layers)
        # frame head sees the mel features AND the onset head's belief
        self.frame = Head(feat + C.N_PITCH, hidden, layers)
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


# The shape the first trained checkpoint used, before the size was measured.
# Checkpoints written then carry no "arch" key, so it has to be named here
# rather than inferred — and it cannot simply be "whatever config says today",
# because config is exactly what changed.
LEGACY_ARCH = {"feat": 256, "hidden": 128, "ch": (32, 32, 64), "layers": 1}


def from_checkpoint(ck: dict) -> "Transcriber":
    """Build a model matching the checkpoint, not matching current config.

    Two model sizes now exist and more may follow, so the architecture travels
    WITH the weights.  Rebuilding from config instead would mean that changing
    a constant silently breaks every checkpoint trained before it — and the
    failure surfaces as a shape mismatch at load time in production, which is
    the shipped transcriber refusing to run.
    """
    a = dict(LEGACY_ARCH)
    a.update(ck.get("arch") or {})
    net = Transcriber(feat=a["feat"], hidden=a["hidden"],
                      ch=tuple(a["ch"]), layers=a["layers"])
    net.load_state_dict(ck["model"])
    return net


def flatten_rnns(net: "Transcriber") -> "Transcriber":
    """Re-pack every LSTM's weights into one buffer.  Call AFTER .to(device).

    cuDNN's fused RNN kernels want the weights in a single contiguous block and
    cache a descriptor pointing at it.  Building a model on the CPU, calling
    load_state_dict, and only then moving it to the GPU leaves that descriptor
    stale — the forward pass still works, and the BACKWARD pass dies with
    `cuDNN error: CUDNN_STATUS_INTERNAL_ERROR`, which says nothing about the
    actual cause.

    That is why training from scratch was fine for 40 epochs while the first
    fine-tune step crashed: train.py never loads a state_dict, so it never hits
    this.  Any path that loads weights and then changes device has to re-pack.
    """
    for m in net.modules():
        if isinstance(m, nn.LSTM):
            m.flatten_parameters()
    return net
