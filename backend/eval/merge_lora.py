"""Fold a LoRA adapter checkpoint into its base weights -> a plain checkpoint (lora 0) that loads like any
spectral model and can be the frozen base for the next adapters (the modular design: plain -> plain-base).

    python eval/merge_lora.py <adapter best.pt> <out .pt>
"""
import pathlib, sys

import torch
import torch.nn as nn

ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from app.ml.spectral.model import SpectralTranscriber        # noqa: E402
from app.ml.spectral.imagine import add_lora, LoRALinear     # noqa: E402

src, dst = sys.argv[1], sys.argv[2]
ck = torch.load(src, map_location="cpu", weights_only=False)
net = SpectralTranscriber(**ck["arch"])
if ck.get("lora"):
    add_lora(net, ck["lora"])
net.load_state_dict(ck["model"])
n = 0
for mod in list(net.modules()):
    for name, child in list(mod.named_children()):
        if isinstance(child, LoRALinear):
            lin = nn.Linear(child.in_features, child.out_features, bias=child.base.bias is not None)
            with torch.no_grad():
                lin.weight.copy_(child.base.weight + (child.B @ child.A) * child.scale)
                if child.base.bias is not None:
                    lin.bias.copy_(child.base.bias)
            setattr(mod, name, lin); n += 1
out = {k: v for k, v in ck.items() if k not in ("opt", "sched", "model")}
out.update(model=net.state_dict(), lora=0, arch=net.arch, merged_from=str(src))
torch.save(out, dst)
print(f"folded {n} LoRA layers -> {dst}")
