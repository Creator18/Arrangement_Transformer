"""Train the note transcriber.

    python -m app.ml.transcriber.train --epochs 40

Reports NOTE-level scores (mir_eval, 50 ms onset tolerance) alongside the
frame-wise ones, because note-level is what published Slakh numbers mean and
frame-wise flatters a model that smears sustain.  Onset and frame are always
shown apart: frames are dense and easy, onsets are ~0.3% of the target and are
precisely what the NMF gets wrong.  A single blended figure would hide the one
failure this model exists to fix.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import config as C
from .data import MelFront, SlakhSegments, official_splits
from .decode import decode_notes, note_scores
from .model import Transcriber


def frame_f1(pred: torch.Tensor, target: torch.Tensor, thr: float = 0.5):
    p = torch.sigmoid(pred) >= thr
    t = target >= 0.5
    tp = float((p & t).sum()); fp = float((p & ~t).sum()); fn = float((~p & t).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return (2 * pr * rc / (pr + rc) if pr + rc else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--batch", type=int, default=C.BATCH)
    ap.add_argument("--lr", type=float, default=C.LR)
    ap.add_argument("--seg-per-track", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--onset-weight", type=float, default=C.ONSET_POS_WEIGHT)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--overfit", type=int, default=0,
                    help="train AND validate on this many tracks -- a capacity "
                         "check, not a result; if it cannot fit a handful of "
                         "tracks, more data will not help")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, va, how = official_splits()
    if not tr:
        raise SystemExit("no Slakh tracks found; check config.SLAKH_DIRS")
    if args.limit:
        tr, va = tr[: args.limit], va[: max(1, args.limit // 5)]
    if args.overfit:
        tr = (tr + va)[: args.overfit]
        va = tr
        how = f"OVERFIT CHECK -- val IS train ({len(tr)} tracks)"
    print(f"splits: {how}  |  {len(tr)} train / {len(va)} val")

    dl_tr = DataLoader(SlakhSegments(tr, args.seg_per_track, seed=1),
                       batch_size=args.batch, shuffle=True, num_workers=0)
    dl_va = DataLoader(SlakhSegments(va, 4, seed=2),
                       batch_size=args.batch, shuffle=False, num_workers=0)

    front = MelFront().to(dev)
    net = Transcriber().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    start_ep, best = 1, -1.0

    C.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_p = C.CKPT_DIR / "transcriber.pt"
    if args.resume and ckpt_p.exists():
        ck = torch.load(ckpt_p, map_location=dev, weights_only=False)
        net.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        start_ep = int(ck.get("epoch", 0)) + 1
        best = float(ck.get("conp_f1", -1.0))
        # The LR schedule is not in the checkpoint, so fast-forward it.  Without
        # this a resume restarts cosine annealing at full rate, which undoes the
        # decay already earned and can knock a converging model backwards.
        for _ in range(start_ep - 1):
            sched.step()
        print(f"resumed from epoch {start_ep - 1} (note F1 {best:.3f}), "
              f"lr {opt.param_groups[0]['lr']:.2e}")
    print(f"model: {net.n_params() / 1e6:.2f}M params on {dev}")

    lo_on = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.onset_weight, device=dev))
    lo_fr = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(C.FRAME_POS_WEIGHT, device=dev))
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")

    for ep in range(start_ep, args.epochs + 1):
        net.train()
        t0, tot, nb = time.time(), 0.0, 0
        for wav, on, fr, vl in dl_tr:
            wav, on, fr, vl = wav.to(dev), on.to(dev), fr.to(dev), vl.to(dev)
            with torch.no_grad():
                mel = front(wav)
            T = min(mel.shape[-1], on.shape[-1])
            mel, on, fr, vl = mel[..., :T], on[..., :T], fr[..., :T], vl[..., :T]
            onT, frT, vlT = on.transpose(1, 2), fr.transpose(1, 2), vl.transpose(1, 2)
            with torch.amp.autocast("cuda", enabled=dev.type == "cuda"):
                p_on, p_fr, p_vl = net(mel)
                # velocity is scored ONLY where a note actually starts; a
                # regression over the silent 99.7% would be learning to output
                # zero, which is true and useless.
                m = onT > 0.5
                l_vel = (((torch.sigmoid(p_vl) - vlT) ** 2) * m).sum() / m.sum().clamp(min=1)
                loss = lo_on(p_on, onT) + lo_fr(p_fr, frT) + 0.5 * l_vel
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(net.parameters(), 3.0)
            scaler.step(opt); scaler.update()
            tot += float(loss); nb += 1
        sched.step()

        net.eval()
        f_on = f_fr = 0.0; n = 0
        agg = {"conp_f1": 0.0, "conp_p": 0.0, "conp_r": 0.0, "conpoff_f1": 0.0}
        nseg = 0
        with torch.no_grad():
            for wav, on, fr, vl in dl_va:
                wav, on, fr, vl = wav.to(dev), on.to(dev), fr.to(dev), vl.to(dev)
                mel = front(wav)
                T = min(mel.shape[-1], on.shape[-1])
                p_on, p_fr, p_vl = net(mel[..., :T])
                f_on += frame_f1(p_on, on[..., :T].transpose(1, 2))
                f_fr += frame_f1(p_fr, fr[..., :T].transpose(1, 2))
                n += 1
                po = torch.sigmoid(p_on).cpu().numpy().transpose(0, 2, 1)
                pf = torch.sigmoid(p_fr).cpu().numpy().transpose(0, 2, 1)
                pv = torch.sigmoid(p_vl).cpu().numpy().transpose(0, 2, 1)
                go = on[..., :T].cpu().numpy(); gf = fr[..., :T].cpu().numpy()
                gv = vl[..., :T].cpu().numpy()
                for b in range(po.shape[0]):
                    est = decode_notes(po[b], pf[b], pv[b])
                    ref = decode_notes(go[b], gf[b], gv[b], 0.5, 0.5)
                    s = note_scores(ref, est)
                    for k in agg:
                        agg[k] += s[k]
                    nseg += 1
        f_on /= max(n, 1); f_fr /= max(n, 1)
        a = {k: v / max(nseg, 1) for k, v in agg.items()}
        print(f"ep {ep:>3}  loss {tot / max(nb, 1):.4f}  "
              f"| NOTE F1 {a['conp_f1']:.3f} (P {a['conp_p']:.3f} R {a['conp_r']:.3f}) "
              f"+off {a['conpoff_f1']:.3f}  "
              f"| frame-wise onset {f_on:.3f} frame {f_fr:.3f}  "
              f"{time.time() - t0:.0f}s", flush=True)

        # Selected on NOTE F1, the metric the field reports and the one that
        # reflects whether the output is readable as music.
        if a["conp_f1"] > best:
            best = a["conp_f1"]
            torch.save({"model": net.state_dict(), "opt": opt.state_dict(),
                        "arch": net.arch,
                        "epoch": ep, "conp_f1": best,
                        "conpoff_f1": a["conpoff_f1"]}, ckpt_p)
            print(f"        saved (note F1 {best:.3f})")


if __name__ == "__main__":
    main()
