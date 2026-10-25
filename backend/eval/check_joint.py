"""joint_refine sanity: with ONE switch on, the side-by-side model must equal that module's own adapter;
with none on it must equal plain_base.  Also times a training step with all 22 adapters loaded.

    python eval/check_joint.py <checkpoints dir>
"""
import pathlib, sys, time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from app.ml.spectral.ft_aux import trunk                       # noqa: E402
from app.ml.spectral.imagine import add_lora                   # noqa: E402
from app.ml.spectral.joint_refine import MODULES, Gate, build  # noqa: E402
from app.ml.spectral.model import from_checkpoint              # noqa: E402

C = pathlib.Path(sys.argv[1]); dev = "cuda"
base_ck = torch.load(C / "plain_base.pt", map_location="cpu", weights_only=False)
joint = build(base_ck, C, MODULES, dev).eval()
x = torch.randn(2, 7, 264, 250, device=dev)


def out(net, g=None):
    Gate.g = g
    with torch.no_grad():
        return net.head(trunk(net, x)).float()


plain = from_checkpoint(base_ck).to(dev).eval()
print(f"no switch vs plain_base: max |diff| {(out(joint, torch.zeros(2, len(MODULES), device=dev)) - out(plain)).abs().max():.2e}")
for m in ("glides", "late", "heavydrums"):
    c = torch.load(C / f"mod_{m}" / "best.pt", map_location="cpu", weights_only=False)
    own = from_checkpoint({"arch": base_ck["arch"], "model": base_ck["model"]}); add_lora(own, c["lora"]); own.load_state_dict(c["model"])
    own = own.to(dev).eval()
    g = torch.zeros(2, len(MODULES), device=dev); g[:, MODULES.index(m)] = 1
    ref = out(own); d = (out(joint, g) - ref).abs().max()
    print(f"{m:10s} switch on vs its own adapter: max |diff| {d:.2e}  (output scale {ref.abs().max():.1f})")
# speed: a training step, 16 clips x 8 s, several switches on
joint.train(); xb = torch.randn(16, 7, 264, 500, device=dev)
opt = torch.optim.AdamW([p for p in joint.parameters() if p.requires_grad], lr=1e-4)
g = (torch.rand(16, len(MODULES), device=dev) < .15).float()
amp = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
for i in range(13):
    if i == 3:
        torch.cuda.synchronize(); t = time.time()
    Gate.g = g
    with torch.autocast("cuda", dtype=amp):
        loss = joint.head(trunk(joint, xb)).float().pow(2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
torch.cuda.synchronize()
print(f"train step, 22 adapters, batch 16: {(time.time() - t) / 10 * 1000:.0f} ms  ({10 / (time.time() - t):.2f} steps/s), "
      f"peak memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GB on {torch.cuda.get_device_name(0)}")
