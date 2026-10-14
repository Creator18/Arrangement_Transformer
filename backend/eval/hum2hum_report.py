"""After hum2hum training: score on TEST songs and render hums to listen to (CPU is enough).

Scores the best checkpoint on the held-out TEST units with the training's own
distance (lower = better) and pitch-activity F1, next to the song hum as-is
and step 2 with no differences.  Then renders, for a few test windows,
four hums side by side through the same sine renderer the hums were made with:
    <out>/<id>_<w>_1_song_hum.wav        the input
    <out>/<id>_<w>_2_imagined_piano.wav  stage 2's output (no snapping, no cleanup)
    <out>/<id>_<w>_3_real_piano.wav      the human arrangement's piano hum
    <out>/<id>_<w>_4_pair.wav            imagined (left) vs real (right)

    python eval/hum2hum_report.py --ckpt <hum2hum/best.pt> --maps <hum_maps> --units m2m_units.json --out <dir>
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from app.ml.spectral.hum2hum import HumNet, Units, closest, index, to_log, SCALE, ACTIVE  # noqa: E402

SR, HOP, FMIN = 22050, 256, 27.5


def render(m: np.ndarray) -> np.ndarray:
    """(264, T) loudness map -> sine hum, the same bank the hums were built with."""
    f = FMIN * 2 ** (np.arange(m.shape[0]) / 36)
    n = m.shape[1] * HOP; t = np.arange(n) / SR; tf = (np.arange(m.shape[1]) + 0.5) * HOP / SR
    y = np.zeros(n)
    for k in np.flatnonzero(m.max(1) > 0):
        y += np.interp(t, tf, m[k], right=0.0) * np.sin(2 * np.pi * f[k] * t)
    pk = np.abs(y).max()
    return (y / pk * 0.9 if pk > 0 else y).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--maps", required=True)
    ap.add_argument("--units", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--renders", type=int, default=6)
    a = ap.parse_args()
    import soundfile as sf
    c = torch.load(a.ckpt, map_location="cpu", weights_only=False); g = c["args"]
    s1 = HumNet(1, 2, g["d"], g["depth"]); s2 = HumNet(3, 1, g["d"], g["depth"], residual=True)
    s1.load_state_dict(c["s1"]); s2.load_state_dict(c["s2"]); s1.eval(); s2.eval()
    aw = g.get("active_weight", 9.0)
    U = json.loads(pathlib.Path(a.units).read_text())
    ds = Units(U["units"]["test"], index(a.maps), train=False)
    R = {k: [] for k in ("song as-is", "imagined", "no differences", "F1 song as-is", "F1 imagined")}
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rendered = 0
    with torch.no_grad():
        for u in ds.units:
            ws = ds.windows(u)
            for w in ws[:: max(1, len(ws) // 4)][:4]:
                A, Bm, M = ds.load(u, w)
                A, Bm, M = to_log(A[None]), to_log(Bm[None]), M[None]
                d = s1(A); p = s2(torch.cat([A, d], 1)); p0 = s2(torch.cat([A, torch.zeros_like(d)], 1))
                v, k = closest(p, Bm, M, aw); R["imagined"].append(float(v))
                R["song as-is"].append(float(closest(A, Bm, M, aw)[0]))
                R["no differences"].append(float(closest(p0, Bm, M, aw)[0]))
                b = Bm[0, k]
                f1 = lambda x, y: float(2 * ((x > ACTIVE) & (y > ACTIVE)).sum() / max(1, int((x > ACTIVE).sum() + (y > ACTIVE).sum())))
                R["F1 imagined"].append(f1(p[0, 0], b[0])); R["F1 song as-is"].append(f1(A[0, 0], b[0]))
                if rendered < a.renders and w == ws[min(1, len(ws) - 1)]:
                    lin = lambda x: np.expm1(x.numpy() * SCALE)
                    tag = f"{u[0]}_{w:03d}"
                    hs, hp, hr = render(lin(A[0, 0])), render(lin(p[0, 0])), render(lin(b[0]))
                    sf.write(out / f"{tag}_1_song_hum.wav", hs, SR)
                    sf.write(out / f"{tag}_2_imagined_piano.wav", hp, SR)
                    sf.write(out / f"{tag}_3_real_piano.wav", hr, SR)
                    n = min(len(hp), len(hr)); sf.write(out / f"{tag}_4_pair.wav", np.stack([hp[:n], hr[:n]], 1), SR)
                    rendered += 1
    print(f"TEST ({len(ds.units)} units, {len(R['imagined'])} windows; checkpoint round {c.get('round')})")
    print(f"  distance to closest real piano hum (lower = better): song hum as-is {np.mean(R['song as-is']):.4f} | "
          f"imagined {np.mean(R['imagined']):.4f} | step 2 with no differences {np.mean(R['no differences']):.4f}")
    print(f"  pitch-activity F1 vs real piano hum: song hum as-is {np.mean(R['F1 song as-is']):.3f} | "
          f"imagined {np.mean(R['F1 imagined']):.3f}")
    print(f"  rendered {rendered} test windows to {out}")
    (out / "test_scores.json").write_text(json.dumps({k: float(np.mean(v)) for k, v in R.items()}, indent=1))


if __name__ == "__main__":
    main()
