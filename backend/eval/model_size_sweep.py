"""What size actually fits, and what does it cost per step?

The fit check says the transcriber is capacity-bound: zero train/val gap at
20 tracks a side, so the note F1 ~0.50 plateau from epoch 19 to 40 was capacity
and not data.  The indicated fix is a bigger model, and the binding constraint
is an 8.6 GB laptop card rather than anything about the task.

Picking a size by arithmetic is how you discover at epoch 3 of a ten-hour run
that it does not fit.  So each candidate gets a real forward AND backward pass
at the real batch size and segment length, with peak VRAM read off the
allocator, before anything is committed to.

Step time matters as much as fitting.  Training is ~5 min/epoch at the current
size and 40 epochs is already three hours; a 4x slower step turns a retrain
into an overnight job, which is a real cost to weigh against the F1 it buys.

    venv/Scripts/python.exe eval/model_size_sweep.py
"""
from __future__ import annotations

import pathlib
import sys
import time

import torch
import torch.nn as nn

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.ml.transcriber import config as C          # noqa: E402
from app.ml.transcriber.model import Transcriber    # noqa: E402

# (label, conv channels, feat, lstm hidden, lstm layers)
CANDIDATES = [
    ("current  (shipped)",   (32, 32, 64),  256, 128, 1),
    ("wider conv",           (48, 48, 96),  256, 128, 1),
    ("wider + feat 512",     (48, 48, 96),  512, 256, 1),
    ("O&F-ish, 2 layers",    (48, 48, 96),  512, 256, 2),
    ("O&F-ish, 3 layers",    (48, 48, 96),  768, 256, 3),
    ("full O&F width",       (48, 48, 96),  768, 384, 3),
]


def probe(ch, feat, hidden, layers, batch, steps=3):
    dev = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    net = Transcriber(feat=feat, hidden=hidden, ch=ch, layers=layers).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    lo = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(30.0, device=dev))

    T = C.SEGMENT_FRAMES
    mel = torch.randn(batch, C.N_MELS, T, device=dev)
    tgt = (torch.rand(batch, T, C.N_PITCH, device=dev) > 0.99).float()

    # One warm-up step: the first pass pays cuDNN autotuning and allocator
    # growth, which is not what a training epoch looks like.
    for i in range(steps + 1):
        if i == 1:
            torch.cuda.synchronize(); t0 = time.time()
        with torch.amp.autocast("cuda"):
            p_on, p_fr, p_vl = net(mel)
            loss = lo(p_on, tgt) + lo(p_fr, tgt)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / steps
    peak = torch.cuda.max_memory_allocated() / 1e9
    n = net.n_params() / 1e6
    del net, opt, mel, tgt
    torch.cuda.empty_cache()
    return n, peak, dt


def main():
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device")
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"{torch.cuda.get_device_name(0)}  {total:.1f} GB, "
          f"batch {C.BATCH}, {C.SEGMENT_FRAMES} frames\n")
    print(f"{'config':<24}{'params':>9}{'peak VRAM':>12}{'s/step':>9}{'x base':>8}")
    print("-" * 62)
    base = None
    for label, ch, feat, hidden, layers in CANDIDATES:
        try:
            n, peak, dt = probe(ch, feat, hidden, layers, C.BATCH)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"{label:<24}{'':>9}{'OOM':>12}")
            continue
        base = base or dt
        flag = "" if peak < total * 0.75 else "  <- tight"
        print(f"{label:<24}{n:>8.2f}M{peak:>11.2f}G{dt:>9.3f}{dt / base:>7.1f}x{flag}")

    print(f"\nheadroom rule: peak under {total * 0.75:.1f} G leaves room for the "
          f"data loader and fragmentation over a long run")


if __name__ == "__main__":
    main()
