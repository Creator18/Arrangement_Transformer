"""Train ONE multi-speaker RVC voice model on the 20 VocalSet singers (CC BY 4.0), one speaker ID
each: the voices the sung-melody renders are cloned into (eval/sing_tts.py -> RVC).

Runs the RVC WebUI's own training steps without the UI (same scripts, same arguments):
preprocess -> F0 (rmvpe, GPU) -> HuBERT features (GPU) -> file list + config -> train -> index.

    cd ~/tools/rvc && ~/envs/rvc/bin/python ~/arrangement_transformer/backend/eval/rvc_vocalset.py \\
        --vocalset /scratch/basu.ang/datasets/vocalset/FULL --exp vocalset --epochs 60
"""
import argparse, copy, json, os, pathlib, random, subprocess, sys

ap = argparse.ArgumentParser()
ap.add_argument("--vocalset", required=True, help="VocalSet FULL/ folder (one sub-folder per singer)")
ap.add_argument("--exp", default="vocalset"); ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--save-every", type=int, default=10); ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--cpus", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
ap.add_argument("--skip-to", default="", help="resume from a step: f0 | feature | train | index")
a = ap.parse_args()

RVC = pathlib.Path.cwd(); PY = sys.executable
sys.path.insert(0, str(RVC))
from tools.multispeaker import build_manifest_from_root, write_manifest  # noqa: E402

exp_dir = RVC / "logs" / a.exp; exp_dir.mkdir(parents=True, exist_ok=True)
SR, SR_TAG, VER = 40000, "40k", "v2"


def run(cmd):
    print("+", " ".join(map(str, cmd)), flush=True)
    env = dict(os.environ, PYTHONPATH=str(RVC) + os.pathsep + os.environ.get("PYTHONPATH", ""), PYTHONSAFEPATH="1",
               RVC_CUDA_GRAPH="0")
    subprocess.run([str(c) for c in cmd], check=True, cwd=RVC, env=env)


steps = ["pre", "f0", "feature", "train", "index"]
start = steps.index(a.skip_to) if a.skip_to else 0

# speaker folders "<name>_<id>_<repeat>" (the WebUI's multi-speaker layout), as symlinks
root = pathlib.Path("/scratch/basu.ang/rvc_vocalset_speakers"); root.mkdir(exist_ok=True)
singers = sorted(p for p in pathlib.Path(a.vocalset).iterdir() if p.is_dir())
for i, s in enumerate(singers):
    link = root / f"{s.name}_{i}_1"
    if not link.exists():
        link.symlink_to(s.resolve(), target_is_directory=True)
print(f"{len(singers)} singers: " + ", ".join(f"{i}={s.name}" for i, s in enumerate(singers)), flush=True)

if start <= 0:
    manifest = build_manifest_from_root(str(root))
    mpath = write_manifest(str(exp_dir), manifest)
    run([PY, "train/preprocess.py", root, SR, a.cpus, exp_dir, "False", 3.7, mpath])
    n = len(list((exp_dir / "0_gt_wavs").glob("*.wav")))
    if n == 0:                                    # RVC's scripts log failures but exit 0: stop here
        sys.exit("preprocess produced no audio (see logs/%s/preprocess.log)" % a.exp)
    print(f"preprocess: {n} slices", flush=True)
if start <= 1:
    run([PY, "train/dataset/extract_f0.py", "cuda", 1, 0, 0, exp_dir, "True"])
if start <= 2:
    run([PY, "train/dataset/extract_hubert_feature.py", "cuda", 1, 0, 0, exp_dir, VER, "True"])
if start <= 3:
    manifest = json.loads((exp_dir / "multispeaker_manifest.json").read_text())
    by_key = {e["output_key"]: e for e in manifest["entries"]}
    gt, fe, f0d, f0n = (exp_dir / d for d in ("0_gt_wavs", "3_feature768", "2a_f0", "2b-f0nsf"))
    names = ({n.split(".")[0] for n in os.listdir(gt)} & {n.split(".")[0] for n in os.listdir(fe)}
             & {n.split(".")[0] for n in os.listdir(f0d)} & {n.split(".")[0] for n in os.listdir(f0n)})
    names = {n for n in names if n.rsplit("_", 1)[0] in by_key}
    lines, spk = [], {}
    for n in sorted(names):
        e = by_key[n.rsplit("_", 1)[0]]; spk[e["speaker_id"]] = e["speaker_name"]
        lines += [f"{gt}/{n}.wav|{fe}/{n}.npy|{f0d}/{n}.wav.npy|{f0n}/{n}.wav.npy|{e['speaker_id']}|{e['speaker_name']}"] * e["repeat"]
    m = RVC / "logs" / "mute"
    for sid, sname in spk.items():
        lines += [f"{m}/0_gt_wavs/mute{SR_TAG}.wav|{m}/3_feature768/mute.npy|{m}/2a_f0/mute.wav.npy|{m}/2b-f0nsf/mute.wav.npy|{sid}|{sname}"] * 2
    random.shuffle(lines)
    (exp_dir / "filelist.txt").write_text("\n".join(lines), encoding="utf8")
    cfg = copy.deepcopy(json.loads((RVC / "configs" / "v1" / f"{SR_TAG}.json").read_text()))   # as the WebUI: 40k uses v1/40k.json
    cfg["model"]["spk_embed_dim"] = 110
    cfg["speaker_info"] = [{"id": i, "name": spk[i]} for i in sorted(spk)]
    (exp_dir / "config.json").write_text(json.dumps(cfg, indent=4, sort_keys=True) + "\n", encoding="utf8")
    print(f"file list: {len(lines)} lines, {len(spk)} speakers", flush=True)
    run([PY, "train/train.py", "-e", a.exp, "-sr", SR_TAG, "-f0", 1, "-bs", a.batch, "-g", 0, "-te", a.epochs,
         "-se", a.save_every, "-pg", "assets/pretrained_v2/f0G40k.pth", "-pd", "assets/pretrained_v2/f0D40k.pth",
         "-l", 1, "-c", 0, "-sw", 0, "-v", VER])
if start <= 4:
    run([PY, "train/train_index.py", a.exp, VER, RVC / "assets" / "indices", a.cpus, "multi"])
print("(finished)", flush=True)
