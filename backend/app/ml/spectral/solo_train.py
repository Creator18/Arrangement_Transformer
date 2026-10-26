"""Solo-voice detector: reads the full mix, says per 16 ms frame whether the mix is essentially the voice alone.

Why: the base reads a voice well inside a mix but drops notes when the voice is alone (eval/aah_test.py,
eval/aah_fill_test.py).  The fix (fill the solo bars with the song's own accompaniment, transcribe, remove what was
added) needs to know where the solo bars are, and the app should not run a separation model to find them.

Labels (the user's idea, 2026-10-26): where the mix is basically the vocal stem.  With the vocal active (within 35 dB
of the song's loudest vocal frame), a frame is SILENT when everything else sits >= 25 dB below the voice and QUIET when
it sits 8-25 dB below (both wanted: a soft pad under a voice counts too); otherwise neither.  Three classes per frame.
  * Music2MIDI songs: stems from Demucs, offline only (eval/m2m_vocal_curves.py keeps just the loudness curves)
  * MUSDB18-HQ: the true stems; natural solo voice is rare there (~1% of vocal time), so solo passages are also
    made by turning the band down in random stretches (bare, or with the "other" stem left as a quiet bed)
Real songs train this detector by the user's decision; the arranger itself stays renders-only.

Model: the base's front end (frozen, the base's calibration) -> conv over pitch -> transformer over time
(32 ms tokens, the whole 10 s clip as context) -> one logit per frame.

    python -m app.ml.spectral.solo_train --data /scratch/basu.ang/solo_det --base <base best.pt> --out <ckpt dir>
"""
import argparse
import json
import math
import pathlib
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.ml.spectral import frontend as S

CLIP_S = 10.0
CLIP = int(CLIP_S * S.SR)
T = S.n_frames(CLIP)


def frame_db(x):
    """dB per front-end frame (16 ms), aligned with S.n_frames."""
    n = len(x) // S.HOP
    e = (x[: n * S.HOP].astype(np.float32).reshape(n, S.HOP) ** 2).mean(1)
    e = np.r_[e, e[-1:]]
    return 10 * np.log10(e + 1e-12)


QUIET_DB, SILENT_DB = 8, 25
NONE, QUIET, SILENT = 0, 1, 2


def target(vdb, rdb, vmax):
    """Per-frame class: 0 neither, 1 voice over a quiet band, 2 voice over a (nearly) silent band."""
    active = (vdb > vmax - 35) & (vdb > -60)
    dom = vdb - rdb
    y = np.where(active & (dom >= SILENT_DB), SILENT, np.where(active & (dom >= QUIET_DB), QUIET, NONE))
    return y.astype(np.int64)


class Musdb:
    """Clips with the true stems; `aug` turns the band down in random stretches to make solo passages."""

    def __init__(self, root, aug):
        self.songs = sorted(p for p in pathlib.Path(root).iterdir() if (p / "bassdrums.npy").exists())
        self.st = [{k: np.load(p / f"{k}.npy", mmap_mode="r") for k in ("vocals", "other", "bassdrums")} for p in self.songs]
        self.vmax = [float(np.percentile(frame_db(np.asarray(s["vocals"])), 99.5)) for s in self.st]
        self.aug = aug

    def clip(self, i, a, rng):
        s = self.st[i]
        v = np.asarray(s["vocals"][a:a + CLIP], np.float32)
        o = np.asarray(s["other"][a:a + CLIP], np.float32)
        b = np.asarray(s["bassdrums"][a:a + CLIP], np.float32)
        if self.aug and rng.random() < 0.7:
            go, gb = np.ones(CLIP, np.float32), np.ones(CLIP, np.float32)
            for _ in range(rng.integers(1, 4)):
                L = int(rng.uniform(0.3, 6.0) * S.SR)
                st = int(rng.integers(0, CLIP))
                fade = int(0.05 * S.SR)
                env = np.ones(CLIP, np.float32)
                g = 10 ** (-rng.uniform(10, 45) / 20)
                lo, hi = max(0, st), min(CLIP, st + L)
                env[lo:hi] = g
                for j in range(fade):                       # 50 ms fades at both ends
                    t = j / fade
                    if lo + j < hi:
                        env[lo + j] = min(env[lo + j], 1 - (1 - g) * t)
                    if hi - 1 - j >= lo:
                        env[hi - 1 - j] = min(env[hi - 1 - j], 1 - (1 - g) * t)
                bed = rng.random() < 0.35                   # keep "other" as a quiet bed (pad / soft guitar)
                gb = np.minimum(gb, env)
                go = np.minimum(go, np.maximum(env, 10 ** (-rng.uniform(12, 20) / 20)) if bed else env)
            o, b = o * go, b * gb
        rest = o + b
        y = target(frame_db(v), frame_db(rest), self.vmax[i])
        return v + rest, y

    def sample(self, rng):
        i = int(rng.integers(len(self.st)))
        n = len(self.st[i]["vocals"])
        return self.clip(i, int(rng.integers(0, max(1, n - CLIP))), rng)


class M2M:
    """Real Music2MIDI mixes with Demucs-derived labels (20 ms curves, interpolated onto 16 ms frames)."""

    def __init__(self, audio_dir, curve_dir, ids):
        self.items = []
        for i in ids:
            a, c = pathlib.Path(audio_dir) / f"{i}.npy", pathlib.Path(curve_dir) / f"{i}.npz"
            if a.exists() and c.exists():
                z = np.load(c)
                hop = float(z["hop"])
                w = np.load(a, mmap_mode="r")
                tt = np.arange(S.n_frames(len(w))) * S.HOP / S.SR
                src = np.arange(len(z["vocal_db"])) * hop
                vdb = np.interp(tt, src, z["vocal_db"])
                rdb = np.interp(tt, src, z["rest_db"])
                vmax = float(np.percentile(z["vocal_db"], 99.5))
                self.items.append((w, target(vdb, rdb, vmax)))

    def clip(self, i, a):
        w, y = self.items[i]
        x = np.asarray(w[a:a + CLIP], np.float32)
        f0 = a // S.HOP
        yy = y[f0:f0 + T]
        if len(x) < CLIP:
            x = np.pad(x, (0, CLIP - len(x)))
        return x, np.pad(yy, (0, T - len(yy)))       # pads with 0 = neither

    def sample(self, rng):
        i = int(rng.integers(len(self.items)))
        n = len(self.items[i][0])
        return self.clip(i, int(rng.integers(0, max(1, n - CLIP))))


class SoloNet(nn.Module):
    def __init__(self, d=256, depth=6, heads=4):
        super().__init__()
        C = len(S.HARMONICS) + 1
        self.conv = nn.Sequential(
            nn.Conv2d(C, 32, (5, 3), stride=(3, 1), padding=(2, 1)), nn.GELU(),        # 264 -> 88 bins (1 / key)
            nn.Conv2d(32, 48, (3, 3), stride=(2, 2), padding=1), nn.GELU(),            # 44 bins, 32 ms tokens
            nn.Conv2d(48, 48, (3, 1), stride=(2, 1), padding=(1, 0)), nn.GELU())       # 22 bins
        self.proj = nn.Linear(48 * 22, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True, norm_first=True,
                                           activation="gelu")
        self.enc = nn.TransformerEncoder(layer, depth)
        self.pos = nn.Parameter(torch.zeros(1, 2048, d))
        nn.init.normal_(self.pos, std=0.02)
        self.head = nn.Linear(d, 6)                              # two 16 ms frames per token x 3 classes

    def forward(self, x):                                        # (B, C, 264, T) -> (B, T, 3) logits
        B, _, _, Tn = x.shape
        h = self.conv(x)                                         # (B, 48, 22, T/2)
        h = h.permute(0, 3, 1, 2).flatten(2)                     # (B, T/2, 48*22)
        h = self.proj(h)
        h = self.enc(h + self.pos[:, : h.shape[1]])
        return self.head(h).reshape(B, -1, 3)[:, :Tn]


class Stream(torch.utils.data.IterableDataset):
    """Endless training clips, cut in DataLoader workers (each with its own seed) so the GPU never waits."""

    def __init__(self, musdb, m2m, share, seed):
        self.musdb, self.m2m, self.share, self.seed = musdb, m2m, share, seed

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed * 1000 + (wi.id if wi else 0))
        while True:
            x, y = self.musdb.sample(rng) if rng.random() < self.share else self.m2m.sample(rng)
            yield (x * 10 ** (rng.uniform(-12, 6) / 20)).astype(np.float32), y      # level jitter


def metrics(p, t):
    """Frame scores for one binary question (p = probability, t = truth): AP, and P / R / F1 at 0.5."""
    if t.sum() == 0:
        return {"ap": None, "p": None, "r": None, "f1": None, "pos": 0}
    o = np.argsort(-p)
    tp = np.cumsum(t[o])
    prec = tp / np.arange(1, len(p) + 1)
    ap = float((prec * t[o]).sum() / t.sum())
    pr = p >= 0.5
    P = float((pr & t).sum() / max(pr.sum(), 1))
    R = float((pr & t).sum() / t.sum())
    return {"ap": ap, "p": P, "r": R, "f1": 2 * P * R / max(P + R, 1e-9), "pos": float(t.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", required=True, help="checkpoint whose front-end calibration to use")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--musdb-share", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train.log", "a")

    def say(s):
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    dev = torch.device("cuda")
    bf16 = torch.cuda.get_device_capability()[0] >= 8      # A100 / H200; a V100 only emulates bf16 (slow)
    amp = torch.bfloat16 if bf16 else torch.float16
    ck = torch.load(a.base, map_location="cpu", weights_only=False)
    fe = S.SpectralFrontend().to(dev).eval()
    fe.mu.copy_(ck["fe_mu"])
    fe.sd.copy_(ck["fe_sd"])

    root = pathlib.Path(a.data)
    sp = json.load(open("/scratch/basu.ang/m2m/m2m_splits.json"))
    m_tr = M2M(root / "m2m", root / "m2m_curves", sp["train"])
    m_va = M2M(root / "m2m", root / "m2m_curves", sp["validation"])
    mu_tr = Musdb(root / "musdb" / "train", aug=True)
    mu_te_nat = Musdb(root / "musdb" / "test", aug=False)
    mu_te_aug = Musdb(root / "musdb" / "test", aug=True)
    say(f"data: m2m train {len(m_tr.items)} / val {len(m_va.items)}, musdb train {len(mu_tr.st)} / test {len(mu_te_nat.st)}"
        f"; {'bf16' if bf16 else 'fp16'}")

    # fixed validation clips: 3 per M2M val song, 4 per MUSDB test song (natural) and 4 augmented
    vr = np.random.default_rng(0)
    vals = {
        "m2m_val": [m_va.clip(i, int(vr.integers(0, max(1, len(m_va.items[i][0]) - CLIP)))) for i in range(len(m_va.items)) for _ in range(3)],
        "musdb_nat": [mu_te_nat.clip(i, int(vr.integers(0, max(1, len(mu_te_nat.st[i]["vocals"]) - CLIP))), vr)
                      for i in range(len(mu_te_nat.st)) for _ in range(4)],
        "musdb_aug": [mu_te_aug.clip(i, int(vr.integers(0, max(1, len(mu_te_aug.st[i]["vocals"]) - CLIP))), vr)
                      for i in range(len(mu_te_aug.st)) for _ in range(4)],
    }

    net = SoloNet().to(dev)
    cls_w = torch.tensor([1.0, 4.0, 4.0], device=dev)
    say(f"SoloNet {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M params")
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, s / 1000) * 0.5 * (1 + math.cos(math.pi * min(1, s / a.steps))))
    scaler = torch.amp.GradScaler('cuda', enabled=not bf16)
    step, best = 0, -1.0
    last = out / "last.pt"
    if last.exists():
        st = torch.load(last, map_location="cpu", weights_only=False)
        net.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        step, best = st["step"], st["best"]
        say(f"resumed at step {step}")

    @torch.no_grad()
    def evaluate():
        net.eval()
        res = {}
        for k, clips in vals.items():
            P, Y = [], []
            for j in range(0, len(clips), 16):
                xb = torch.from_numpy(np.stack([c[0] for c in clips[j:j + 16]])).to(dev)
                with torch.autocast("cuda", dtype=amp):
                    lg = net(fe(xb))
                P.append(torch.softmax(lg.float(), -1).reshape(-1, 3).cpu().numpy())
                Y.append(np.stack([c[1] for c in clips[j:j + 16]]).ravel())
            P, Y = np.concatenate(P), np.concatenate(Y)
            res[k] = metrics(P[:, QUIET] + P[:, SILENT], Y >= QUIET)          # solo = quiet or silent band
            res[k + "_silent"] = metrics(P[:, SILENT], Y == SILENT)
            res[k + "_quiet"] = metrics(P[:, QUIET], Y == QUIET)
        net.train()
        return res

    random.seed(step)
    loader = iter(torch.utils.data.DataLoader(Stream(mu_tr, m_tr, a.musdb_share, step + 1), batch_size=a.batch,
                                              num_workers=a.workers, pin_memory=True, persistent_workers=True))
    net.train()
    t0, seen = time.time(), 0
    while step < a.steps:
        xb, yb = next(loader)
        xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
        with torch.no_grad():
            feat = fe(xb)
        with torch.autocast("cuda", dtype=amp):
            lg = net(feat)
        # solo frames are rare: weight them up
        loss = F.cross_entropy(lg.float().reshape(-1, 3), yb.reshape(-1), weight=cls_w)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1
        seen += 1
        if step % 100 == 0:
            sps = seen / (time.time() - t0)
            say(f"step {step} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e} {sps:.2f} steps/s")
            t0, seen = time.time(), 0
        if step % a.val_every == 0 or step == a.steps:
            r = evaluate()
            score = np.mean([r[k]["ap"] for k in vals if r[k]["ap"] is not None])
            say("VAL " + json.dumps({"step": step, "score": round(float(score), 4),
                                     **{k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                                        for k, v in r.items()}}))
            state = {"model": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                     "step": step, "best": max(best, score), "fe_mu": fe.mu.cpu(), "fe_sd": fe.sd.cpu(),
                     "kind": "solo_detector"}
            if score > best:
                best = score
                torch.save({k: v for k, v in state.items() if k not in ("opt", "sched")}, out / "best.pt")
                say(f"new best {best:.4f}")
            torch.save(state, last)
    say("(finished)")


if __name__ == "__main__":
    main()
