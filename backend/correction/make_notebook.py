"""
make_notebook.py — generate the overnight correction-model notebook.

Writes backend/correction/train_correction.ipynb, which runs the whole
track unattended: build the training pairs over all clean ASAP scores,
featurise them, train the note-level corrector, evaluate it, and print a
review summary.

Generated rather than hand-written so the JSON is always valid and the
cells stay in sync with the code they call.
"""
from __future__ import annotations

import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
C = []

C.append(nbf.v4.new_markdown_cell("""\
# Score-Informed Correction Model — overnight run

Runs unattended end to end:

1. **Build pairs** — run the real pipeline over every clean ASAP score,
   align each extraction to its score, emit (ours, truth) pairs.
2. **Featurise** — per-note local-context features.
3. **Train** — one shared encoder, three heads:
   *keep/delete*, *duration correction*, *hand*.
4. **Evaluate** — held-out pieces, split by piece (never by note).

Every cell prints its own diagnostics; nothing needs to be watched.
Long-running cells write to disk as they go, so a failure late in the run
does not destroy earlier work.

**Design note.** This is deliberately a note-level corrector with local
context, not a sequence-to-sequence re-notation model. It can fix
durations, delete spurious notes and re-assign hands — the three errors
we have actually measured. It cannot *insert* missed notes; insertion
needs generation and is a much harder problem, deferred on purpose.
"""))

C.append(nbf.v4.new_code_cell("""\
# ── environment ──────────────────────────────────────────────────────────
import os, sys, json, time, subprocess, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

REPO = Path.cwd()
while not (REPO / "backend").exists() and REPO != REPO.parent:
    REPO = REPO.parent
BACKEND = REPO / "backend"
DATA    = BACKEND / "correction" / "data"
sys.path.insert(0, str(BACKEND))

import numpy as np, torch
print("repo      :", REPO)
print("python    :", sys.version.split()[0])
print("torch     :", torch.__version__, "| cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device    :", torch.cuda.get_device_name(0))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

for p in ["score_targets.jsonl", "train_manifest.json"]:
    q = DATA / p
    print(f"{p:<22}", "OK" if q.exists() else "MISSING",
          f"({q.stat().st_size/1e6:.1f} MB)" if q.exists() else "")
"""))

C.append(nbf.v4.new_markdown_cell("""\
## 1. Build the training pairs

The long cell — roughly 85 s per score over 216 scores, so about five
hours. `build_pairs.py` appends to `pairs.jsonl` as it goes and prints a
line per piece, so progress is visible in this cell's output and partial
results survive an interruption.
"""))

C.append(nbf.v4.new_code_cell("""\
# ── build pairs over every clean score (long) ────────────────────────────
PAIRS = DATA / "pairs.jsonl"
REBUILD = True          # set False to reuse an existing pairs.jsonl

if REBUILD:
    t0 = time.time()
    proc = subprocess.Popen(
        [sys.executable, "-W", "ignore",
         str(BACKEND / "correction" / "build_pairs.py"), "--limit", "216"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", cwd=str(REPO))
    keep = ("[", "matched", "SKIPPED", "===", "our notes", "spurious",
            "missed", "duration", "too short", "onset bias", "hand labels",
            "built in", "REFUSED")
    for line in proc.stdout:
        if line.startswith(keep) or any(k in line for k in keep):
            print(line.rstrip(), flush=True)
    proc.wait()
    print(f"\\nbuild finished rc={proc.returncode} in {(time.time()-t0)/60:.1f} min")
else:
    print("reusing", PAIRS)

rows = [json.loads(l) for l in open(PAIRS, encoding="utf-8")]
print(f"pieces with pairs: {len(rows)}")
"""))

C.append(nbf.v4.new_markdown_cell("""\
## 2. Dataset sanity

Before training anything, confirm the data is not lying. The failure mode
that already bit us once was misalignment producing fabricated
"spurious"/"missed" labels, which looks exactly like a hard piece. A
bimodal precision distribution is the tell.
"""))

C.append(nbf.v4.new_code_cell("""\
# ── sanity report ────────────────────────────────────────────────────────
P = np.array([r["stats"]["precision"] for r in rows])
R = np.array([r["stats"]["recall"]    for r in rows])
tot = {k: sum(r["stats"][k] for r in rows)
       for k in ("n_ours","n_score","n_matched","n_spurious","n_missed")}
print(f"pieces {len(rows)} | our notes {tot['n_ours']:,} | score notes {tot['n_score']:,}")
print(f"matched {tot['n_matched']:,}  spurious {tot['n_spurious']:,}  missed {tot['n_missed']:,}")
print(f"precision  mean {P.mean():.3f}  median {np.median(P):.3f}  min {P.min():.3f}")
print(f"recall     mean {R.mean():.3f}  median {np.median(R):.3f}  min {R.min():.3f}")

low = (P < 0.5).sum()
print(f"\\npieces with precision < 0.5: {low} / {len(P)}")
if low > len(P) * 0.25:
    print("  WARNING: heavy low-precision tail — check for misalignment, not difficulty")
else:
    print("  distribution looks unimodal; alignment is behaving")

m = [x for r in rows for x in r["matched"]]
dr = np.array([x["dur_ratio"] for x in m]); oe = np.array([x["onset_err_b"] for x in m])
print(f"\\nmatched notes {len(m):,}")
print(f"duration ratio  median {np.median(dr):.3f} | too short {100*np.mean(dr<0.8):.1f}%"
      f" | too long {100*np.mean(dr>1.25):.1f}%")
print(f"onset err       bias {np.median(oe):+.4f} beats | spread {oe.std():.4f}")
from collections import Counter
print("hand labels    ", dict(Counter(x["tgt_hand"] for x in m)))
"""))

C.append(nbf.v4.new_markdown_cell("""\
## 3. Featurise

One row per note we produced. Features are local and relative — nothing
absolute that would tie the model to a particular key, tempo or register.

Targets:
* `keep` — 1 if the note matched a score note, 0 if spurious
* `log_dur_ratio` — how wrong our duration was (matched notes only)
* `hand` — the staff the score puts it on (matched notes only)
"""))

C.append(nbf.v4.new_code_cell("""\
# ── features ─────────────────────────────────────────────────────────────
LAYERS = ["melody","pattern","bass_pattern","bass_melody","bass",""]

def featurise_piece(r):
    ev = ([(x["ours_onset_b"], x["ours_dur_b"], x["pitch"], x["ours_vel"],
            x.get("ours_layer") or "", 1, x["dur_ratio"], x["tgt_hand"]) for x in r["matched"]]
        + [(x["onset_b"], x["dur_b"], x["pitch"], 80, "", 0, 1.0, None)
           for x in r["spurious"]])
    ev.sort(key=lambda e: (e[0], e[2]))
    if len(ev) < 5: return None
    on = np.array([e[0] for e in ev]); du = np.array([e[1] for e in ev])
    pi = np.array([e[2] for e in ev], dtype=float); ve = np.array([e[3] for e in ev], dtype=float)
    X, Yk, Yd, Yh, Md, Mh = [], [], [], [], [], []
    med_p = np.median(pi)
    for i, e in enumerate(ev):
        lo, hi = max(0, i-4), min(len(ev), i+5)
        win_p = pi[lo:hi]; win_on = on[lo:hi]
        simul = float(np.sum(np.abs(on - on[i]) < 0.06))
        prev_gap = on[i]-on[i-1] if i > 0 else 2.0
        next_gap = on[i+1]-on[i] if i < len(ev)-1 else 2.0
        dens = len(win_on)/max(1e-6, (win_on.max()-win_on.min()) or 1.0)
        lay = [0.0]*len(LAYERS); lay[LAYERS.index(e[4]) if e[4] in LAYERS else -1] = 1.0
        X.append([ (pi[i]-60)/24, (pi[i]-med_p)/12, np.log1p(du[i]),
                   ve[i]/127, np.log1p(min(prev_gap,4)), np.log1p(min(next_gap,4)),
                   simul/6, min(dens,20)/20,
                   (pi[i]-win_p.min())/12, (win_p.max()-pi[i])/12,
                   float(pi[i]==win_p.max()), float(pi[i]==win_p.min()),
                   (win_p.mean()-60)/24, win_p.std()/12,
                   on[i]-np.floor(on[i]),                       # position in beat
                 ] + lay)
        Yk.append(float(e[5]))
        Yd.append(np.log(max(1e-3, e[6])) if e[5] else 0.0); Md.append(float(e[5]))
        Yh.append(1.0 if e[7]=="RH" else 0.0)
        # bool() first: `e[5] and e[7]` returns the STRING "RH" when both are
        # truthy, and float("RH") raises.
        Mh.append(1.0 if (e[5] and e[7] is not None) else 0.0)
    return (np.array(X,dtype=np.float32), np.array(Yk,dtype=np.float32),
            np.array(Yd,dtype=np.float32), np.array(Yh,dtype=np.float32),
            np.array(Md,dtype=np.float32), np.array(Mh,dtype=np.float32))

feats, order = {}, []
for r in rows:
    f = featurise_piece(r)
    if f is not None:
        feats[r["folder"]] = f; order.append(r["folder"])
N_FEAT = feats[order[0]][0].shape[1]
print(f"featurised {len(order)} pieces | {sum(f[0].shape[0] for f in feats.values()):,} notes"
      f" | {N_FEAT} features/note")
"""))

C.append(nbf.v4.new_code_cell("""\
# ── split BY PIECE (never by note) ───────────────────────────────────────
rng = np.random.RandomState(42); idx = rng.permutation(len(order))
n_val = max(1, int(0.15*len(order)))
val_f = {order[i] for i in idx[:n_val]}; tr_f = {order[i] for i in idx[n_val:]}

def stack(fs):
    parts = [feats[f] for f in fs]
    return [np.concatenate([p[k] for p in parts]) for k in range(6)]

Xtr,Yktr,Ydtr,Yhtr,Mdtr,Mhtr = stack(tr_f)
Xva,Ykva,Ydva,Yhva,Mdva,Mhva = stack(val_f)
mu, sd = Xtr.mean(0), Xtr.std(0)+1e-6
Xtr, Xva = (Xtr-mu)/sd, (Xva-mu)/sd
print(f"train {len(tr_f)} pieces / {len(Xtr):,} notes | val {len(val_f)} pieces / {len(Xva):,} notes")
print(f"keep rate  train {Yktr.mean():.3f}  val {Ykva.mean():.3f}")
print(f"RH rate    train {Yhtr[Mhtr>0].mean():.3f}")
"""))

C.append(nbf.v4.new_markdown_cell("""\
## 4. Train

Shared trunk, three heads. Duration and hand losses are masked to matched
notes only, since spurious notes have no truth to compare against.
"""))

C.append(nbf.v4.new_code_cell("""\
# ── model + training ─────────────────────────────────────────────────────
import torch.nn as nn
torch.manual_seed(0)

class Corrector(nn.Module):
    def __init__(self, n_in, h=192):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(n_in,h), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(h,h),    nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(h,h//2), nn.ReLU())
        self.keep = nn.Linear(h//2,1)
        self.dur  = nn.Linear(h//2,1)
        self.hand = nn.Linear(h//2,1)
    def forward(self,x):
        z=self.trunk(x); return self.keep(z).squeeze(-1), self.dur(z).squeeze(-1), self.hand(z).squeeze(-1)

def T(a): return torch.tensor(a, device=DEVICE)
Xtr_t,Xva_t = T(Xtr),T(Xva)
tens = [T(v) for v in (Yktr,Ydtr,Yhtr,Mdtr,Mhtr)]
vten = [T(v) for v in (Ykva,Ydva,Yhva,Mdva,Mhva)]

model = Corrector(N_FEAT).to(DEVICE)
opt   = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)
bce, mse = nn.BCEWithLogitsLoss(), nn.MSELoss(reduction="none")
BS, EPOCHS = 4096, 40
best, best_state = 1e9, None

for ep in range(EPOCHS):
    model.train(); perm = torch.randperm(len(Xtr_t), device=DEVICE); tl=0.0; nb=0
    for i in range(0, len(perm), BS):
        b = perm[i:i+BS]
        k,d,h = model(Xtr_t[b])
        yk,yd,yh,md,mh = (t[b] for t in tens)
        loss = bce(k,yk)
        if md.sum()>0: loss = loss + ((mse(d,yd)*md).sum()/md.sum())
        if mh.sum()>0: loss = loss + (nn.functional.binary_cross_entropy_with_logits(
                                        h,yh,reduction="none")*mh).sum()/mh.sum()
        opt.zero_grad(); loss.backward(); opt.step(); tl+=float(loss); nb+=1
    sched.step()
    model.eval()
    with torch.no_grad():
        k,d,h = model(Xva_t); yk,yd,yh,md,mh = vten
        vl = float(bce(k,yk))
        if md.sum()>0: vl += float((mse(d,yd)*md).sum()/md.sum())
        if mh.sum()>0: vl += float((nn.functional.binary_cross_entropy_with_logits(
                                      h,yh,reduction="none")*mh).sum()/mh.sum())
    if vl < best: best, best_state = vl, {k_: v.detach().clone() for k_,v in model.state_dict().items()}
    if ep % 5 == 0 or ep == EPOCHS-1:
        print(f"epoch {ep:>3} train {tl/max(1,nb):.4f} val {vl:.4f}{'  *' if vl==best else ''}")
model.load_state_dict(best_state)
print(f"\\nbest val loss {best:.4f}")
"""))

C.append(nbf.v4.new_markdown_cell("""\
## 5. Evaluate

Each head against the baseline of doing nothing, which is the only
comparison that matters: the model has to beat leaving the pipeline's
output alone.
"""))

C.append(nbf.v4.new_code_cell("""\
# ── evaluation ───────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    k,d,h = model(Xva_t)
    pk = torch.sigmoid(k).cpu().numpy(); pd_ = d.cpu().numpy(); ph = torch.sigmoid(h).cpu().numpy()

print("=== HEAD 1: spurious-note detection ===")
for thr in (0.3,0.5,0.7):
    pred = (pk>=thr).astype(float)
    tp=((pred==1)&(Ykva==1)).sum(); fp=((pred==1)&(Ykva==0)).sum()
    fn=((pred==0)&(Ykva==1)).sum(); tn=((pred==0)&(Ykva==0)).sum()
    prec=tp/max(1,tp+fp); rec=tp/max(1,tp+fn)
    print(f"  keep-thr {thr}: precision {prec:.3f} recall {rec:.3f} | "
          f"correctly deleted {tn}/{tn+fp} spurious")
print(f"  baseline (keep everything): precision {Ykva.mean():.3f} recall 1.000")

print("\\n=== HEAD 2: duration correction ===")
mask = Mdva>0
before = np.abs(Ydva[mask])                      # |log ratio| as-is
after  = np.abs(Ydva[mask]-pd_[mask])            # after applying the correction
print(f"  notes {mask.sum():,}")
print(f"  mean |log dur error|  before {before.mean():.4f}  after {after.mean():.4f}"
      f"  ({100*(1-after.mean()/max(1e-9,before.mean())):+.1f}%)")
for tol,name in ((0.223,'within 25%'),(0.105,'within 11%')):
    print(f"  {name}: before {100*np.mean(before<tol):.1f}%  after {100*np.mean(after<tol):.1f}%")

print("\\n=== HEAD 3: hand assignment ===")
hm = Mhva>0
if hm.sum()>0:
    acc=((ph[hm]>=0.5).astype(float)==Yhva[hm]).mean()
    # pitch<60 -> LH is the heuristic our pipeline effectively uses
    naive=((Xva[hm][:,0]*24+60)>=60).astype(float)
    print(f"  model accuracy {acc:.3f} | pitch>=C4 heuristic {(naive==Yhva[hm]).mean():.3f}")
"""))

C.append(nbf.v4.new_code_cell("""\
# ── save + review summary ────────────────────────────────────────────────
ckpt = DATA / "corrector.pt"
torch.save({"state_dict":model.state_dict(), "mu":mu, "sd":sd,
            "n_feat":int(N_FEAT), "layers":LAYERS,
            "val_pieces":sorted(val_f)}, ckpt)
json.dump({"pieces":len(rows), "matched":int(tot['n_matched']),
           "best_val_loss":float(best)},
          open(DATA/"train_summary.json","w"), indent=2)
print("saved", ckpt, f"({ckpt.stat().st_size/1e6:.1f} MB)")
print("\\nREVIEW CHECKLIST")
print("  1. cell 2 — is the precision distribution unimodal? a heavy low tail means misalignment")
print("  2. head 1 — does deleting spurious notes cost real notes?")
print("  3. head 2 — did mean |log dur error| actually drop? this is the COnPOff lever")
print("  4. head 3 — does the model beat the pitch>=C4 heuristic? that is the LH/RH complaint")
"""))

nb["cells"] = C
nb.metadata = {"kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
               "language_info": {"name":"python"}}
out = Path(__file__).resolve().parent / "train_correction.ipynb"
nbf.write(nb, str(out))
print("wrote", out)
