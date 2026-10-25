"""Load a stage-5 joint checkpoint (app/ml/spectral/joint_refine.py) for the eval scripts.

The joint model needs each clip's adapter switches; set_clip(npz) sets them before the forward pass:
  oracle   the effects the clip was rendered with (what stage 5 trained with; stage 6's gate must learn it)
  off      no adapter (= plain_base exactly)
  all      every adapter on
init=True uses the adapters as each was trained alone (stage 5's step 0) instead of the refined ones.
"""
import json, pathlib, sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from app.ml.spectral import frontend as S                                   # noqa: E402
from app.ml.spectral.joint_refine import Gate, build                       # noqa: E402

_state = {"modules": None, "mode": "oracle", "dev": "cuda"}


def is_joint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False).get("kind") == "joint_refine"
    except Exception:                                                        # noqa: BLE001
        return False


def load(path, mode="oracle", init=False):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    base_ck = torch.load(ck["base"], map_location="cpu", weights_only=False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = build(base_ck, ck["args"]["modules_dir"], ck["modules"], dev)
    if not init:
        net.load_state_dict(ck["model"])
    net.eval()
    fe = S.SpectralFrontend().to(dev).eval(); fe.mu.copy_(ck["fe_mu"].to(dev)); fe.sd.copy_(ck["fe_sd"].to(dev))
    _state.update(modules=ck["modules"], mode=mode, dev=dev)
    return net, fe, dev


def set_clip(z):
    """Switches for this clip (an opened npz), batch of one."""
    mods = _state["modules"]; g = torch.zeros(1, len(mods), device=_state["dev"])
    if _state["mode"] == "all":
        g[:] = 1
    elif _state["mode"] == "oracle":
        used = json.loads(str(z["render"])).get("module", "plain").split("+") if "render" in z.files else []
        for m in used:
            if m in mods:
                g[0, mods.index(m)] = 1
    Gate.g = g
