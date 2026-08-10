"""
The whole pedal-aware eval chain, runnable unattended.

Same four steps as pedal_eval.ipynb, but driven by the venv interpreter
directly so it does not depend on which kernel Jupyter happens to pick — the
most likely reason the notebook failed, since the deps live in backend/venv
and a default kernel would not see torch or librosa at all.

    backend/venv/Scripts/python.exe -u backend/eval/run_pedal_eval.py
"""
import json, os, pathlib, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[2]
PY = str(ROOT / "backend" / "venv" / "Scripts" / "python.exe")
if not os.path.exists(PY):
    PY = sys.executable
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "backend" / "eval"))


def run(args, tag):
    print(f"\n{'='*64}\n{tag}\n{'='*64}", flush=True)
    t0 = time.time()
    # -u so a piped child is not silent until it exits.
    p = subprocess.run([PY, "-u", *args], cwd=str(ROOT),
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    print(f"--- {tag}: exit {p.returncode} in {(time.time()-t0)/60:.1f} min", flush=True)
    return p.returncode


def score_pedal():
    from eval.pedal_metrics import pedal_scores, aggregate, report
    from eval.synthesize_audio import _midi_to_pedal
    man = json.load(open(ROOT / "backend/eval/data/manifest.json"))
    pieces = man["pieces"] if isinstance(man, dict) and "pieces" in man else man

    rows, missing, nopedal, nopred = [], 0, 0, 0
    for p in pieces:
        tf = ROOT / "backend/eval/runs" / p["id"] / "transcription.json"
        if not tf.exists():
            missing += 1
            continue
        # Pedal travels INSIDE timing_feel, not as a top-level key.  The
        # first version of this read the top level, got [] for all 49 pieces,
        # and reported frame F1 0.0000 — which looks like a broken detector
        # and was a broken reader.  A metric that can return a confident zero
        # because of a key name is worse than no metric.
        _d = json.load(open(tf))
        pred = (_d.get("pedal")
                or (_d.get("timing_feel") or {}).get("pedal")
                or [])
        if not pred:
            nopred += 1
        truth = _midi_to_pedal(pathlib.Path(p["midi_path"]))
        if not truth:
            nopedal += 1
            continue
        r = pedal_scores(pred, truth, float(p["duration_s"]))
        r["id"] = p["id"]
        rows.append(r)

    agg = aggregate(rows)
    print(report(agg), flush=True)
    print(f"scored {len(rows)}, missing runs {missing}, no truth pedal {nopedal}, EMPTY PREDICTION {nopred}")
    if rows:
        print("\nworst 5 by frame F1:")
        for r in sorted(rows, key=lambda x: x["f1"])[:5]:
            print(f"  {r['f1']:.3f}  down pred {r['down_rate_pred']:.2f} / "
                  f"truth {r['down_rate_truth']:.2f}   {r['id'][:52]}")
    out = ROOT / "backend/eval/data/pedal_scores.json"
    json.dump({"aggregate": agg, "per_piece": rows}, open(out, "w"), indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    t0 = time.time()
    run(["backend/eval/synthesize_audio.py", "--force"], "1/4 synthesise WITH pedal")
    run(["backend/eval/run_pipeline.py", "--all", "--force"], "2/4 pipeline on all 50")
    run(["backend/eval/compare.py", "--all", "--save-baseline"], "3/4 transcription + baseline")
    print(f"\n{'='*64}\n4/4 pedal scoring\n{'='*64}", flush=True)
    score_pedal()
    print(f"\nTOTAL {(time.time()-t0)/60:.1f} min", flush=True)
