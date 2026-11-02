"""Random-note curriculum for the arranger (the user's design, 2026-11-02): random notes with random qualities,
made on the fly by CPU producers (eval/random_producer.py), and the model must say exactly what was played.
Start easy; move up a level once the model is near-perfect on it; at most 10 notes at once.

    task switch   the base gets a conditioning input (SpectralTranscriber n_cond=3, zero-initialised, so step 0 IS
                  the base):  2 = ARRANGE (what the app already passes; written arrangement notes only)
                              0 = IDENTIFY EVERYTHING (the random clips: every audible note)
    batches       every step mixes --arr normal arrangement renders from disk (cond 2) with --rand random clips
                  from the spool (cond 0), so the arranger keeps its musical sense and its selective target
    inaudible     notes the mix hides (eval/random_notes.audible) are neither taught nor scored
    curriculum    every --val-every steps: note F1 on the fixed seeded test clips of the current level (control.json
                  -> producers); two checks in a row at >= --bar move up a level.  Also note F1 of the ARRANGE task
                  on base_render validation clips, against step 0, so the arranger never gets worse unnoticed
    record        <out>/records/step_<n>_level_<k>.json: every test clip's input notes (with tags) and output notes
    feeding       the spool is read by a background thread; the GPU's waiting time is logged every 200 steps

    python -m app.ml.spectral.train_random --init checkpoints/base20m/best.pt --data /scratch/basu.ang/base_render \\
        --spool DIR --control FILE --evalset DIR --out checkpoints/arr_random
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import queue
import random
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from app.ml.spectral import frontend as S
from app.ml.spectral.data import RenderedClips, clips
from app.ml.spectral.model import SpectralTranscriber
from app.ml.spectral.train import note_f1
from app.ml.transcriber.decode import decode_notes

ARRANGE, IDENTIFY = 2, 0
TAGS = ("octave_pair", "semitone_neighbour", "quick_repeat", "short", "quiet_under_loud", "soft_attack", "low", "high",
        "tremolo", "inaudible")                                       # = eval/random_notes.TAGS (the record's tag columns)
LEVEL_MAX = 6


class Spool:
    """Background reader: takes the oldest bundles, loads them, deletes them, and hands out single clips."""

    def __init__(self, folder, maxq=512):
        self.dir, self.q, self.stop = pathlib.Path(folder), queue.Queue(maxq), False
        self.loaded = 0
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self.stop:
            files = sorted((p for p in self.dir.glob("*.npz") if not p.name.endswith(".tmp.npz")),
                           key=lambda p: p.stat().st_mtime)
            if not files:
                time.sleep(0.5); continue
            for f in files[:8]:
                try:
                    z = np.load(f); d = {k: z[k] for k in ("y", "on", "fr", "vel", "teach", "level")}
                    f.unlink()
                except Exception:                              # noqa: BLE001  (a producer still renaming it)
                    continue
                for i in range(len(d["y"])):
                    self.q.put((d["y"][i], d["on"][i], d["fr"][i], d["vel"][i], d["teach"][i], int(d["level"][i])))
                self.loaded += 1

    def take(self, n, timeout):
        out = []
        t0 = time.time()
        while len(out) < n:
            try:
                out.append(self.q.get(timeout=max(0.01, timeout - (time.time() - t0))))
            except queue.Empty:
                break
        return out


def match(est, ref, tol=0.05):
    """greedy one-to-one: same pitch, onset within tol.  Returns matched flags for est and ref."""
    me, mr = np.zeros(len(est), bool), np.zeros(len(ref), bool)
    for i, (t, p) in enumerate(est):
        d = np.where((ref[:, 1] == p) & ~mr, np.abs(ref[:, 0] - t), 9) if len(ref) else np.array([])
        if len(d) and d.min() <= tol:
            j = int(np.argmin(d)); me[i] = mr[j] = True
    return me, mr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--spool", required=True); ap.add_argument("--control", required=True)
    ap.add_argument("--evalset", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--arr", type=int, default=6); ap.add_argument("--rand", type=int, default=2)
    ap.add_argument("--crop", type=float, default=8.0); ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--steps", type=int, default=60000); ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--bar", type=float, default=0.98); ap.add_argument("--arr-val", type=int, default=120)
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--max-minutes", type=float, default=0)
    ap.add_argument("--onset-neighbor-weight", type=float, default=0.3)
    ap.add_argument("--start-level", type=int, default=1, help="a fresh run (no last.pt) starts at this level")
    a = ap.parse_args()
    t_start = time.time()
    dev = "cuda"
    bf16 = torch.cuda.get_device_capability()[0] >= 8
    amp = torch.bfloat16 if bf16 else torch.float16
    out = pathlib.Path(a.out); (out / "records").mkdir(parents=True, exist_ok=True)
    log = open(out / "train.log", "a")

    def say(s):
        print(s, flush=True); log.write(s + "\n"); log.flush()

    ck0 = torch.load(pathlib.Path(a.init).expanduser(), map_location="cpu", weights_only=False)
    arch = dict(ck0["arch"]); arch["n_cond"] = 3
    net = SpectralTranscriber(**arch).to(dev)
    fe = S.SpectralFrontend().to(dev).eval()
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 500) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
    scaler = torch.amp.GradScaler("cuda", enabled=not bf16)
    st = dict(step=0, level=a.start_level, passes=0, history=[], arr_base=None)
    last = out / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=dev, weights_only=False)
        net.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        st = ck["state"]
        say(f"resumed at step {st['step']}, level {st['level']}")
    else:
        missing, unexpected = net.load_state_dict(ck0["model"], strict=False)
        say(f"from {a.init} (step {ck0['step']}); new (zero-initialised): {missing}; unexpected: {unexpected}")
    fe.mu.copy_(ck0["fe_mu"].to(dev)); fe.sd.copy_(ck0["fe_sd"].to(dev))

    def set_level(k, stop=False):
        tmp = pathlib.Path(a.control + ".tmp")
        tmp.write_text(json.dumps(dict(level=int(k), stop=stop))); tmp.replace(a.control)
    set_level(st["level"])

    def save(path, extra=None):
        ck = dict(model=net.state_dict(), arch=net.arch, opt=opt.state_dict(), sched=sched.state_dict(),
                  fe_mu=fe.mu.cpu(), fe_sd=fe.sd.cpu(), step=st["step"], state=st, best=0.0, args=vars(a))
        ck.update(extra or {})
        tmp = path.with_suffix(".tmp"); torch.save(ck, tmp); tmp.replace(path)

    arr_paths = clips(pathlib.Path(a.data), "train")
    va = clips(pathlib.Path(a.data), "validation"); random.Random(0).shuffle(va); va = sorted(va[:a.arr_val])
    arr_dl = DataLoader(RenderedClips(arr_paths, a.crop, random_crop=True), batch_size=a.arr, shuffle=True,
                        num_workers=a.workers, pin_memory=True, drop_last=True, persistent_workers=True)
    va_dl = DataLoader(RenderedClips(va, 20.0, random_crop=False), batch_size=2, num_workers=0)   # no forks after the thread
    it = iter(arr_dl)                      # fork the training workers BEFORE the spool thread exists (fork + thread = hang)
    spool = Spool(a.spool)
    evalsets = {}
    pw_on, pw_fr = torch.tensor(30.0, device=dev), torch.tensor(2.0, device=dev)
    say(f"{len(arr_paths)} arrangement clips; batch = {a.arr} arrange + {a.rand} random; level {st['level']}; "
        f"bar {a.bar}; {'bf16' if bf16 else 'fp16'}")

    @torch.no_grad()
    def run(wav, cond):
        x = fe(wav.to(dev))
        with torch.autocast("cuda", dtype=amp):
            o = net(x, torch.full((len(wav),), cond, device=dev, dtype=torch.long))
        return [torch.sigmoid(t.float()).cpu().numpy() for t in o[:3]]

    def eval_random(level):
        if level not in evalsets:
            z = np.load(pathlib.Path(a.evalset) / f"level{level}.npz")
            evalsets[level] = (z["y"], json.loads(str(z["record"])))
        Y, recs = evalsets[level]
        net.eval()
        tp = fp = fn = 0
        tag_hit, tag_n, rows = {}, {}, []
        trem_n = trem_one = trem_extra = 0
        for i in range(0, len(Y), 4):
            po, pf, pv = run(torch.from_numpy(Y[i:i + 4].astype(np.float32) / 32767.0), IDENTIFY)
            for j in range(len(po)):
                r = recs[i + j]
                est = decode_notes(po[j].T, pf[j].T, pv[j].T, 0.5, 0.5, min_frames=1, min_note_s=0.0, keep_restruck=True)
                E = np.array([[n["onset"], n["pitch"]] for n in est]).reshape(-1, 2)
                N, T = np.array(r["notes"]).reshape(-1, 5), np.array(r["tags"]).reshape(len(r["notes"]), -1)
                inaud = T[:, -1].astype(bool) if len(T) else np.zeros(0, bool)
                R = N[:, [0, 2]]
                me, mr = match(E, R)
                # a detected inaudible note is neither right nor wrong; an inaudible note missed is not a miss
                tp += int((mr & ~inaud).sum()); fn += int((~mr & ~inaud).sum())
                hit_inaud = np.zeros(len(E), bool)
                if inaud.any() and (~me).any():
                    hit_inaud[np.nonzero(~me)[0]] = match(E[~me], R[inaud])[0]
                fp += int((~me & ~hit_inaud).sum())
                # tremolo notes: the answer is ONE note; count extra strikes of that key inside it
                if "tremolo" in TAGS and len(T):
                    ti = TAGS.index("tremolo")
                    for n_ in N[T[:, ti].astype(bool) & ~inaud]:
                        x = int(((E[:, 1] == n_[2]) & (E[:, 0] > n_[0] + 0.05) & (E[:, 0] < n_[1])).sum())
                        trem_n += 1; trem_extra += x; trem_one += int(x == 0)
                for k, tg in enumerate(TAGS[:-1]):
                    m = T[:, k].astype(bool) & ~inaud if len(T) else np.zeros(0, bool)
                    tag_n[tg] = tag_n.get(tg, 0) + int(m.sum()); tag_hit[tg] = tag_hit.get(tg, 0) + int((mr & m).sum())
                rows.append(dict(input=r, output=[[round(n["onset"], 3), round(n["offset"], 3), int(n["pitch"]),
                                                   int(n["velocity"])] for n in est]))
        net.train()
        P, Rc = tp / max(1, tp + fp), tp / max(1, tp + fn)
        f1 = 2 * P * Rc / max(1e-9, P + Rc)
        res = dict(level=level, f1=round(f1, 4), p=round(P, 4), r=round(Rc, 4),
                   recall_by_tag={t: round(tag_hit[t] / tag_n[t], 3) for t in tag_n if tag_n[t]})
        if trem_n:
            res["tremolo"] = dict(notes=trem_n, held_as_one=round(trem_one / trem_n, 3), extra_per_note=round(trem_extra / trem_n, 2))
        (out / "records" / f"step_{st['step']}_level_{level}.json").write_text(json.dumps(dict(res, clips=rows), default=float))
        return res

    def eval_arrange():
        net.eval()
        po, pf, go, gf = [], [], [], []
        with torch.no_grad():
            for b in va_dl:
                o, f_, _ = run(b[0], ARRANGE)
                po.append(o); pf.append(f_); go.append(b[1].transpose(1, 2).numpy()); gf.append(b[2].transpose(1, 2).numpy())
        net.train()
        s, ff = note_f1(np.concatenate(po), np.concatenate(pf), np.concatenate(go), np.concatenate(gf))
        return round(float(s["conp_f1"]), 4)

    if st["arr_base"] is None:
        say("step-0 checks: arrange validation ...")
        st["arr_base"] = eval_arrange()
        say(f"  arrange note F1 {st['arr_base']}; identify test clips ...")
        r0 = eval_random(st["level"])
        say(f"VAL step 0: arrange note F1 {st['arr_base']} (the base) | identify level {st['level']}: {json.dumps(r0)}")
    net.train()
    wait, t_log, run_loss, n_rand = 0.0, time.time(), 0.0, 0
    while st["step"] < a.steps:
        try:
            b = next(it)
        except StopIteration:
            it = iter(arr_dl); b = next(it)
        wav, on, fr, vel = b[0], b[1], b[2], b[3]
        teach = torch.ones_like(on); cond = [ARRANGE] * len(wav)
        t0 = time.time()
        R = spool.take(a.rand, timeout=5.0)
        wait += time.time() - t0
        if R:
            wav = torch.cat([wav, torch.from_numpy(np.stack([r[0] for r in R]).astype(np.float32) / 32767.0)])
            on = torch.cat([on, torch.from_numpy(np.stack([r[1] for r in R]).astype(np.float32))])
            fr = torch.cat([fr, torch.from_numpy(np.stack([r[2] for r in R]).astype(np.float32))])
            vel = torch.cat([vel, torch.from_numpy(np.stack([r[3] for r in R]).astype(np.float32) / 127.0)])
            teach = torch.cat([teach, torch.from_numpy(np.stack([r[4] for r in R]).astype(np.float32))])
            cond += [IDENTIFY] * len(R); n_rand += len(R)
        wav = wav.to(dev, non_blocking=True)
        onT, frT, vlT = (t.to(dev, non_blocking=True).transpose(1, 2) for t in (on, fr, vel))
        tm = teach.to(dev).transpose(1, 2)
        with torch.no_grad():
            x = fe(wav)
        with torch.autocast("cuda", dtype=amp):
            o = net(x, torch.tensor(cond, device=dev))
        za, zb, zc = (t.float() for t in o[:3])
        pk = (onT > 0.5).float()
        nb = torch.zeros_like(pk); nb[:, 1:] = pk[:, :-1]; nb[:, :-1] = torch.maximum(nb[:, :-1], pk[:, 1:])
        w_on = (1 - (1 - a.onset_neighbor_weight) * nb * (1 - pk)) * tm
        l_on = (F.binary_cross_entropy_with_logits(za, onT, pos_weight=pw_on, reduction="none") * w_on).sum() / w_on.sum().clamp(min=1)
        l_fr = (F.binary_cross_entropy_with_logits(zb, frT, pos_weight=pw_fr, reduction="none") * tm).sum() / tm.sum().clamp(min=1)
        m = (onT > 0.5) * tm
        l_vel = (((torch.sigmoid(zc) - vlT) ** 2) * m).sum() / m.sum().clamp(min=1)
        loss = l_on + l_fr + 0.5 * l_vel
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        st["step"] += 1; run_loss += float(loss)
        if st["step"] % 200 == 0:
            dt = time.time() - t_log
            say(f"step {st['step']} level {st['level']} loss {run_loss / 200:.4f} {200 / dt:.2f} steps/s | "
                f"random clips {n_rand} (want {200 * a.rand}), GPU waited for them {wait:.0f} s of {dt:.0f} s | "
                f"spool bundles read {spool.loaded}")
            t_log, run_loss, wait, n_rand = time.time(), 0.0, 0.0, 0
        if st["step"] % a.val_every == 0:
            r = eval_random(st["level"]); af = eval_arrange()
            st["passes"] = st["passes"] + 1 if r["f1"] >= a.bar else 0
            st["history"].append(dict(step=st["step"], arrange_f1=af, **r))
            say(f"VAL step {st['step']}: identify {json.dumps(r)} | arrange note F1 {af} (base {st['arr_base']}, "
                f"{af - st['arr_base']:+.4f}) | passes at the bar {st['passes']}/2")
            save(out / f"level{st['level']}_latest.pt")
            if st["passes"] >= 2 and st["level"] < LEVEL_MAX:
                save(out / f"level{st['level']}_done.pt")
                st["level"] += 1; st["passes"] = 0; set_level(st["level"])
                say(f"LEVEL UP -> {st['level']} at step {st['step']}")
            save(last)
        if a.max_minutes and (time.time() - t_start) / 60 > a.max_minutes:
            save(last); say(f"time budget at step {st['step']}: saved, resubmit to resume"); return
    save(last); set_level(st["level"], stop=True)
    say(f"done at step {st['step']}, level {st['level']}")


if __name__ == "__main__":
    main()
