"""Fine-tune the transcriber into an arranger.

    python -m app.ml.transcriber.train_arranger --epochs 20

Same model, same front end, same decoder.  The only thing that changes is the
answer: instead of "every pitched note in this recording" the target becomes
"the notes a human put in their piano cover of it".

WHY THIS IS A FINE-TUNE AND NOT A NEW MODEL.  The two tasks share almost all
their difficulty — both have to pull pitches out of a dense mix — and differ
only in what to do once a pitch is found.  The transcription checkpoint already
holds the expensive half, measured at note F1 0.630 against a 0.078 chance
floor on held-out Slakh.  Music2MIDI is ~2,400 training songs, which is plenty
to teach "which notes to keep" and nowhere near enough to teach "how to hear".
Starting from scratch would spend a small dataset relearning the part we
already have.  This is AMT-APC's result (arXiv 2409.14086) arrived at for the
same reason.

WHAT TO WATCH.  The arrangement target is about FOUR TIMES SPARSER than the
transcription one — a Music2MIDI cover runs ~7 notes/sec against Slakh's ~27 —
so a model that simply keeps transcribing will show high recall and low
precision, and the interesting number is whether precision climbs.  That is the
model learning to leave things out, which is the entire point.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import config as C
from .arranger_data import AUDIO, DATA, Music2MidiSegments, available, split_ids
from .data import MelFront
from .decode import decode_notes, note_scores
from .model import Transcriber, from_checkpoint
from .train import frame_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=C.BATCH)
    # A tenth of the pretraining LR.  Fine-tuning at full rate would walk the
    # weights away from the transcription solution before the new targets had
    # taught anything, which discards the reason for starting here.
    ap.add_argument("--lr", type=float, default=C.LR / 10)
    ap.add_argument("--seg-per-song", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--onset-weight", type=float, default=C.ONSET_POS_WEIGHT)
    ap.add_argument("--audio-dir", type=str, default=str(AUDIO))
    ap.add_argument("--freeze-conv", action="store_true",
                    help="hold the convolutional front end fixed and adapt only "
                         "the recurrent stack and heads -- worth trying if the "
                         "train/val gap says we are memorising 2,400 songs")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    import pathlib
    audio_dir = pathlib.Path(args.audio_dir)

    if not available():
        raise SystemExit(f"Music2MIDI not unpacked at {DATA}")
    tr, va, te = split_ids(audio_dir)
    if not tr:
        raise SystemExit(
            f"no songs have audio in {audio_dir}.\n"
            f"The dataset ships YouTube ids, not recordings. Put 16 kHz mono\n"
            f"WAVs named <id>.wav there; ids are in {DATA}/dataset_split.npz\n"
            f"and the per-song YouTube ids are in {DATA}/youtube_csv/."
        )
    if args.limit:
        tr, va = tr[: args.limit], va[: max(1, args.limit // 5)]
    print(f"music2midi: {len(tr)} train / {len(va)} val / {len(te)} test "
          f"(with audio, from the dataset's own split)")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dl_tr = DataLoader(Music2MidiSegments(tr, audio_dir, args.seg_per_song, seed=1),
                       batch_size=args.batch, shuffle=True, num_workers=0)
    dl_va = DataLoader(Music2MidiSegments(va, audio_dir, 4, seed=2),
                       batch_size=args.batch, shuffle=False, num_workers=0)

    front = MelFront().to(dev)

    C.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    src = C.CKPT_DIR / "transcriber.pt"
    dst = C.CKPT_DIR / "arranger.pt"
    start_ep, best = 1, -1.0
    # The arranger INHERITS the transcriber's architecture rather than reading
    # current config.  It is a fine-tune: the weights only mean anything in the
    # shape they were trained in, so the checkpoint decides the size and a
    # config edit cannot silently mismatch them.
    if args.resume and dst.exists():
        ck = torch.load(dst, map_location=dev, weights_only=False)
        net = from_checkpoint(ck).to(dev)
        start_ep = int(ck.get("epoch", 0)) + 1
        best = float(ck.get("conp_f1", -1.0))
        print(f"resumed arranger from epoch {start_ep - 1} (F1 {best:.3f})")
    elif src.exists():
        ck = torch.load(src, map_location=dev, weights_only=False)
        net = from_checkpoint(ck).to(dev)
        print(f"initialised from transcriber.pt "
              f"(epoch {ck.get('epoch')}, note F1 {ck.get('conp_f1', 0):.3f}, "
              f"{net.n_params() / 1e6:.2f}M params)")
    else:
        net = Transcriber().to(dev)
        print("WARNING: no transcriber.pt -- training the arranger from "
              "scratch, which this dataset is too small to do well.")

    if args.freeze_conv:
        for m in (net.conv_on, net.conv_fr):
            for p in m.parameters():
                p.requires_grad = False
        print("conv front end frozen")

    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    for _ in range(start_ep - 1):
        sched.step()
    print(f"model: {net.n_params() / 1e6:.2f}M params on {dev}, lr {args.lr:.2e}")

    lo_on = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.onset_weight, device=dev))
    lo_fr = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(C.FRAME_POS_WEIGHT, device=dev))
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")

    for ep in range(start_ep, args.epochs + 1):
        net.train()
        t0, tot, nb = time.time(), 0.0, 0
        for wav, on, fr, vl, has_dyn in dl_tr:
            wav, on, fr, vl = wav.to(dev), on.to(dev), fr.to(dev), vl.to(dev)
            has_dyn = has_dyn.to(dev)
            with torch.no_grad():
                mel = front(wav)
            T = min(mel.shape[-1], on.shape[-1])
            mel, on, fr, vl = mel[..., :T], on[..., :T], fr[..., :T], vl[..., :T]
            onT, frT, vlT = on.transpose(1, 2), fr.transpose(1, 2), vl.transpose(1, 2)
            with torch.amp.autocast("cuda", enabled=dev.type == "cuda"):
                p_on, p_fr, p_vl = net(mel)
                # Velocity is scored only at onsets AND only for songs that
                # actually carry dynamics.  35.2% of these arrangements hold a
                # single velocity for the whole piece, which is a MuseScore
                # export default rather than quiet playing; training on it would
                # collapse the head to a constant and undo what Slakh, which has
                # real per-note velocity, already taught it.
                m = (onT > 0.5) * has_dyn.view(-1, 1, 1)
                l_vel = (((torch.sigmoid(p_vl) - vlT) ** 2) * m).sum() / m.sum().clamp(min=1)
                loss = lo_on(p_on, onT) + lo_fr(p_fr, frT) + 0.5 * l_vel
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(params, 3.0)
            scaler.step(opt); scaler.update()
            tot += float(loss); nb += 1
        sched.step()

        net.eval()
        f_on = f_fr = 0.0; n = 0
        agg = {"conp_f1": 0.0, "conp_p": 0.0, "conp_r": 0.0, "conpoff_f1": 0.0}
        nseg = 0
        with torch.no_grad():
            for wav, on, fr, vl, _ in dl_va:
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
              f"| COVER F1 {a['conp_f1']:.3f} (P {a['conp_p']:.3f} R {a['conp_r']:.3f}) "
              f"+off {a['conpoff_f1']:.3f}  "
              f"| frame-wise onset {f_on:.3f} frame {f_fr:.3f}  "
              f"{time.time() - t0:.0f}s", flush=True)

        if a["conp_f1"] > best:
            best = a["conp_f1"]
            torch.save({"model": net.state_dict(), "opt": opt.state_dict(),
                        "arch": net.arch,
                        "epoch": ep, "conp_f1": best,
                        "conpoff_f1": a["conpoff_f1"]}, dst)
            print(f"        saved (cover F1 {best:.3f})")


if __name__ == "__main__":
    main()
