"""Bar-constrained greedy decoding for the structure-aware grammar.

The generation check showed the decoder drifting out of the bar grid: on
Music2MIDI it wrote the wrong NUMBER of bars a third of the time, and every
note after an extra or missing BAR lands in the wrong bar — note F1 collapsed
from 0.861 (its corrupted input) to 0.480.  But at inference the bar grid is
KNOWN (beat tracking), so the model should not have to guess it.  This gives
it the skeleton and leaves every musical choice to the model:

  - the passage starts with BAR and contains exactly the known number of bars
    (EOS is blocked until the last bar, BAR is blocked after it)
  - positions stay inside the current bar's length (beats x 24) and never go
    backwards within a bar (the token order the corpus uses)

Nothing about WHICH notes, chords, roles or groups is constrained.
"""
from __future__ import annotations

from typing import Callable, List, Sequence

import torch

from . import tokens as T, tokens2 as T2


@torch.no_grad()
def generate(step: Callable[[torch.Tensor], torch.Tensor], prefix: torch.Tensor,
             bar_lens: Sequence[Sequence[int]], max_len: int = 950) -> List[List[int]]:
    """step(y) -> logits (B, V) for the next token; bar_lens[b] = beats per bar for row b."""
    B, dev = prefix.shape[0], prefix.device
    y = prefix.clone()
    bars = [0] * B                  # BAR tokens written so far
    last_pos = [-1] * B             # last POS in the current bar
    done = [False] * B
    out = [[] for _ in range(B)]
    for t in range(max_len):
        lg = step(y).float()
        V = lg.shape[1]
        mask = torch.zeros_like(lg, dtype=torch.bool)          # True = forbidden
        for b in range(B):
            if done[b]:
                continue
            n = len(bar_lens[b])
            if t == 0:                                         # must open with a bar
                mask[b] = True; mask[b, T2.BAR] = False; continue
            if bars[b] >= n:
                mask[b, T2.BAR] = True                         # no extra bars
            else:
                mask[b, T.EOS] = True                          # no early end
            cur = bar_lens[b][min(bars[b], n) - 1] if bars[b] else bar_lens[b][0]
            hi = T2.POS0 + min(cur, T2.MAX_BEATS) * T2.GRID
            mask[b, hi:T2.CHORD0] = True                       # inside the bar
            if last_pos[b] >= 0:
                mask[b, T2.POS0:T2.POS0 + last_pos[b]] = True  # never backwards
        lg = lg.masked_fill(mask, float("-inf"))
        nxt = lg.argmax(-1)
        for b in range(B):
            if done[b]:
                nxt[b] = T.PAD; continue
            x = int(nxt[b])
            if x == T.EOS:
                done[b] = True; continue
            out[b].append(x)
            if x == T2.BAR:
                bars[b] += 1; last_pos[b] = -1
            elif T2.POS0 <= x < T2.CHORD0:
                last_pos[b] = x - T2.POS0
        y = torch.cat([y, nxt[:, None].to(dev)], 1)
        if all(done):
            break
    return out


def bar_lengths(grid: Sequence[int], n: int) -> List[int]:
    """Beats per bar for the first n bars of a bar grid (beat index of each bar start)."""
    return [max(1, int(grid[i + 1]) - int(grid[i])) for i in range(min(n, len(grid) - 1))]
