"""One JSON snapshot of the modular-adapter pipeline, for the app's /progress page (light: squeue, sacct
and log tails only).

    python3 eval/progress_dump.py            print the snapshot
    python3 eval/progress_dump.py --write    also save it to ~/arrangement_transformer/progress/ (latest.json,
                                             history.jsonl) and email any NEW alert once; run every 5 min by
                                             scrontab (eval/progress_scrontab.txt)
"""
import glob, json, os, re, subprocess, sys, time

USER = os.environ.get("USER", "basu.ang")
L = os.path.expanduser("~/arrangement_transformer/logs")
CK = os.path.expanduser("~/arrangement_transformer/checkpoints")
OUT = os.path.expanduser("~/arrangement_transformer/progress")
MAIL_TO = os.environ.get("PROGRESS_MAIL", f"{USER}@northeastern.edu")
STALL_MIN = 45        # a running lane whose log hasn't changed for this long (data mixing is silent ~15 min)
BAD = ("FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL", "DEADLINE")
SHARDS = 100
LANES = [["late", "connected", "articulation", "expression"], ["glides", "noteend", "decay", "vibrato"],
         ["rolls", "unison", "levels", "exposed"], ["singer", "timbre", "reverb", "eq"],
         ["dynamics", "drums"], ["sidechain", "genre", "fast"],
         ["heavydrums"]]     # lane 6: split off lane 4 after a heavydrums shard failed (2026-10-23)
MODULES = [m for lane in LANES for m in lane]
STEP = re.compile(r"^(\S+ \S+)\s+step\s+(\d+)\s+loss ([\d.]+).*?([\d.]+) steps/s(?:.*?GPU (\d+)%)?")  # GPU % missing on some nodes
VAL = re.compile(r"^(\S+ \S+)\s+VAL step (\d+): note F1 ([\d.]+) \(P ([\d.]+) R ([\d.]+)\)\s+frame F1 ([\d.]+)")
OVERALL = re.compile(r"^\s+app 0\.9\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$")
WAIT = re.compile(r"data wait (\d+)%")
TIMING = re.compile(r"start, sub-frame centroid\s+median\s+(\S+)\s+\|err\| median\s+(\S+)\s+within 8 ms ([\d.]+)%")


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


# render arrays whose output was thrown away (two copies collided on 2026-10-23): never counted
IGNORE_ARRAYS = {"11330229", "11330334"}
SONG_LINE = re.compile(r"^\s+(train|validation) (?:pop|m2m)\d+: (\d+) renders x (\d+) windows")   # POP909 and Music2MIDI


def secs(t):
    """sacct elapsed [D-]HH:MM:SS -> seconds."""
    if not t or t in ("None", "Unknown"):
        return 0
    d, _, hms = t.rpartition("-")
    parts = [int(x) for x in hms.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return (int(d) if d else 0) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def rss_mb(v):
    if not v:
        return 0.0
    mult = {"K": 1 / 1024, "M": 1, "G": 1024}.get(v[-1], 1 / 2 ** 20)
    try:
        return float(v.rstrip("KMG")) * mult
    except ValueError:
        return 0.0


def shard_log(name, arr, task, cache):
    """(songs, clips, finished) from one shard's log; finished shards are cached."""
    key = f"{arr}_{task}"
    if key in cache:
        return cache[key]
    f = f"{L}/mr_{name}_{arr}_{task}.out"
    songs = clips = 0; fin = False
    if os.path.exists(f):
        for ln in open(f, errors="ignore"):
            m = SONG_LINE.match(ln)
            if m:
                songs += 1; clips += int(m.group(2)) * int(m.group(3))
            elif "clips written" in ln:
                fin = True
    r = (songs, clips, fin)
    if fin:
        cache[key] = r
    return r


def renders():
    """Every render array (mr-<module>, reruns merged): per shard its state, songs done (read from its log,
    so a running shard shows how far it is), clips written, run time, peak memory, node; and per module
    the totals, rate and time left.  Expected songs per shard come from the plain renders (every module
    splits the songs the same way)."""
    cpath = f"{OUT}/render_cache.json"
    try:
        cache = {k: tuple(v) for k, v in json.load(open(cpath)).items()}
    except (OSError, ValueError):
        cache = {}
    rows = [ln.split("|") for ln in sh(f"sacct -u {USER} -S now-3days -n -P "
                                      "-o jobid,jobname,state,elapsed,maxrss,nodelist,start,end").splitlines()]
    rss = {}
    for r in rows:                                     # step rows (.batch) carry the memory
        if len(r) >= 5 and ".batch" in r[0]:
            rss[r[0].split(".")[0]] = rss_mb(r[4])
    mods = {}
    for r in rows:
        if len(r) < 8 or "." in r[0] or not r[1].startswith("mr-"):
            continue
        jid, name, state, el, _rss, node, start, end = r[:8]
        arr, _, task = jid.partition("_")
        if arr in IGNORE_ARRAYS:
            continue
        key = re.sub(r"\d+$", "", name[3:])            # mr-combo2 -> combo
        M = mods.setdefault(key, {"tasks": {}, "pending": 0, "first": None})
        if task.startswith("["):                       # a pending range: [12-99%25]
            m = re.search(r"\[(\d+)-(\d+)", task)
            for i in range(int(m.group(1)), int(m.group(2)) + 1) if m else []:
                M["tasks"].setdefault(i, {"state": "pending", "arr": arr, "name": name[3:]})
            continue
        if not task.isdigit():
            continue
        i = int(task)
        st = ("done" if state == "COMPLETED" else "running" if state.startswith(("RUNNING", "COMPLETING"))
              else "pending" if state.startswith("PENDING") else "failed" if state.split()[0] in BAD else "cancelled")
        rank = {"done": 0, "running": 1, "failed": 2, "pending": 3, "cancelled": 4}
        old = M["tasks"].get(i)
        if old and rank[old["state"]] <= rank[st]:
            continue                                   # a rerun that succeeded wins over the failure
        M["tasks"][i] = {"state": st, "arr": arr, "name": name[3:], "elapsed": secs(el), "rss": round(rss.get(jid, 0.0)),
                         "node": node if node not in ("None assigned", "") else None, "start": start}
        if start not in ("None", "Unknown", "") and (M["first"] is None or start < M["first"]):
            M["first"] = start
    expect = {}                                        # songs per shard, from the plain renders
    if "plain" in mods:
        for i, t in mods["plain"]["tasks"].items():
            if t["state"] == "done":
                expect[i] = shard_log(t["name"], t["arr"], i, cache)[0]
    out = {}
    now = time.time()
    for key, M in mods.items():
        grid, songs, clips, tasks = [], 0, 0, []
        n_sh = max(SHARDS, max(M["tasks"], default=-1) + 1)                 # this job's shard count (100, 200, 400 ...)
        for i in range(n_sh):
            t = M["tasks"].get(i)
            if not t:
                grid.append("."); tasks.append(None); continue
            sg, cl, _fin = shard_log(t["name"], t["arr"], i, cache) if t["state"] in ("done", "running", "failed") else (0, 0, False)
            songs += sg; clips += cl
            grid.append({"done": "d", "running": "r", "failed": "f"}.get(t["state"], "."))
            tasks.append({"s": t["state"][0], "songs": sg, "of": expect.get(i) if n_sh == SHARDS else None, "clips": cl, "el": t.get("elapsed", 0),
                          "rss": t.get("rss", 0), "node": t.get("node")})
        total_songs = sum(expect.values()) if expect and n_sh == SHARDS else None   # the plain renders split songs 100 ways
        n = {k: sum(1 for t in tasks if t and t["s"] == k[0]) for k in ("done", "running", "failed", "pending")}
        n["pending"] += sum(1 for t in tasks if t is None)
        done_el = [t["el"] for t in tasks if t and t["s"] == "d" and t["el"]]
        first = time.mktime(time.strptime(M["first"], "%Y-%m-%dT%H:%M:%S")) if M["first"] else None
        active = n["running"] + n["pending"] > 0
        ends = [time.mktime(time.strptime(t["start"], "%Y-%m-%dT%H:%M:%S")) + t["elapsed"]
                for t in M["tasks"].values() if t.get("start") not in (None, "None", "Unknown", "") and t.get("elapsed")]
        wall = ((now if active else max(ends or [now])) - first) if first else 0      # first start -> now / last end
        rate = songs / (wall / 60) if wall > 60 and songs else None       # songs per minute over the job
        eta = ((total_songs - songs) / rate) if (rate and total_songs and active) else None
        if eta is None and active and n["done"] and wall:                   # no song count to go by: shards done so far
            eta = wall / 60 * (n_sh - n["done"]) / n["done"]               # minutes
        out[key] = {"done": n["done"], "running": n["running"], "pending": n["pending"], "failed": n["failed"],
                    "total": n_sh, "grid": "".join(grid), "songs": songs, "songs_total": total_songs, "clips": clips,
                    "first": M["first"], "wall_min": round(wall / 60, 1) if wall else None,
                    "rate": round(rate, 2) if rate else None, "eta_min": round(eta, 1) if eta else None,
                    "shard_min_mean": round(sum(done_el) / len(done_el) / 60, 1) if done_el else None,
                    "shard_min_max": round(max(done_el) / 60, 1) if done_el else None,
                    "rss_max": max([t["rss"] for t in tasks if t] or [0]),
                    "nodes": len({t["node"] for t in tasks if t and t["node"]}), "tasks": tasks, "active": active}
    try:
        json.dump({k: list(v) for k, v in cache.items()}, open(cpath, "w"))
    except OSError:
        pass
    return out


def jobs():
    rows, types = [], gpu_types()
    for line in sh(f"squeue -u {USER} -h -o '%i|%j|%T|%M|%l|%R|%P'").splitlines():
        i, j, t, el, lim, why, part = line.split("|")
        if not j.startswith("mr-"):
            rows.append({"id": i, "name": j, "state": t, "elapsed": el, "limit": lim, "reason": why, "partition": part,
                         "gpu": types.get(why) if t == "RUNNING" else None})
    return rows


def parse_train(path):
    if not path or not os.path.exists(path):
        return None
    txt = open(path, errors="ignore").read().splitlines()
    runs, cur = [], None
    for ln in txt:
        if "clips:" in ln and "train" in ln:
            cur = {"vals": [], "last": None, "stopped": None, "started": ln[:14], "trace": []}
            runs.append(cur)
        if cur is None:
            continue
        if (m := STEP.match(ln)):
            cur["last"] = {"time": m.group(1), "step": int(m.group(2)), "loss": float(m.group(3)),
                           "speed": float(m.group(4)), "gpu": int(m.group(5)) if m.group(5) else None}
            w = WAIT.search(ln)
            # [step, loss, GPU %, data wait %, steps/s] for the health charts
            cur["trace"].append([int(m.group(2)), float(m.group(3)), int(m.group(5)) if m.group(5) else None, int(w.group(1)) if w else 0, float(m.group(4))])
        elif (m := VAL.match(ln)):
            cur["vals"].append({"time": m.group(1), "step": int(m.group(2)), "f1": float(m.group(3)),
                                "p": float(m.group(4)), "r": float(m.group(5)), "frame": float(m.group(6))})
        elif "stopping at step" in ln:
            cur["stopped"] = ln[16:].strip()
    for r in runs:      # at most ~200 trace points per run
        k = max(1, len(r["trace"]) // 200)
        r["trace"] = r["trace"][::k]
    return runs


def merged_run(path):
    """All runs in one checkpoint folder's train.log (a first run plus any resumes) as one run."""
    runs = parse_train(path)
    if not runs:
        return None
    vals = {v["step"]: v for r in runs for v in r["vals"]}
    trace = {t[0]: t for r in runs for t in r["trace"]}
    return {"started": runs[0]["started"], "vals": [vals[k] for k in sorted(vals)],
            "trace": [trace[k] for k in sorted(trace)], "last": runs[-1]["last"] or runs[-2]["last"] if len(runs) > 1 else runs[-1]["last"],
            "stopped": runs[-1]["stopped"]}


JVAL = re.compile(r"^(\S+ \S+)\s+VAL step (\d+): score ([\d.]+)\s+combination F1 ([\d.]+)")
JMOD = re.compile(r"(\w+) ([\d.]+)(?:\(([+-][\d.]+)\))?")
JSTEP = re.compile(r"^(\S+ \S+)\s+step\s+(\d+)\s+loss ([\d.]+)\s+drift (\S+)\s+([\d.]+) steps/s\s+data wait (\d+)%")


def joint():
    """Stage 5 (joint refinement) from checkpoints/joint/train.log: steps, and at each validation the
    combination F1 and every module's F1 on its own clips (step 0 = as trained alone)."""
    f = f"{CK}/joint/train.log"
    if not os.path.exists(f):
        return None
    out = {"vals": [], "trace": [], "last": None, "stopped": None}
    lines = open(f, errors="ignore").read().splitlines()
    for i, ln in enumerate(lines):
        if (m := JSTEP.match(ln)):
            out["last"] = {"time": m.group(1), "step": int(m.group(2)), "loss": float(m.group(3)), "drift": float(m.group(4)),
                           "speed": float(m.group(5)), "wait": int(m.group(6))}
            out["trace"].append([int(m.group(2)), float(m.group(3)), float(m.group(4)), int(m.group(6))])
        elif (m := JVAL.match(ln)) and i + 1 < len(lines):
            mods = {k: float(v) for k, v, _d in JMOD.findall(lines[i + 1])}
            out["vals"].append({"time": m.group(1), "step": int(m.group(2)), "score": float(m.group(3)),
                                "combo": float(m.group(4)), "modules": mods})
        elif "stopping at step" in ln or "time limit" in ln:
            out["stopped"] = ln[16:].strip()
    k = max(1, len(out["trace"]) // 200); out["trace"] = out["trace"][::k]
    return out


GVAL = re.compile(r"^(\S+ \S+)\s+VAL step (\d+): detection \(balanced\) ([\d.]+)\s+\|\s+combination note F1: gate ([\d.]+)\s+known ([\d.]+)\s+none ([\d.]+)")
GSTEP = re.compile(r"^(\S+ \S+)\s+step\s+(\d+)\s+bce ([\d.]+)\s+task ([\d.]+)\s+([\d.]+) steps/s\s+data wait (\d+)%")
GMOD = re.compile(r"(\w+) ([\d.]+)/([\d.]+)")


def gate_phase(name):
    """Stage 6 phase from checkpoints/gate_<name>/train.log."""
    f = f"{CK}/gate_{name}/train.log"
    if not os.path.exists(f):
        return None
    out = {"vals": [], "trace": [], "last": None, "stopped": None}
    lines = open(f, errors="ignore").read().splitlines()
    for i, ln in enumerate(lines):
        if (m := GSTEP.match(ln)):
            out["last"] = {"time": m.group(1), "step": int(m.group(2)), "bce": float(m.group(3)), "task": float(m.group(4)),
                           "speed": float(m.group(5)), "wait": int(m.group(6))}
            out["trace"].append([int(m.group(2)), float(m.group(3)), float(m.group(4)), int(m.group(6))])
        elif (m := GVAL.match(ln)):
            det = {k: [float(a_), float(b_)] for k, a_, b_ in GMOD.findall(lines[i + 1])} if i + 1 < len(lines) else {}
            out["vals"].append({"time": m.group(1), "step": int(m.group(2)), "detect": float(m.group(3)), "gate": float(m.group(4)),
                                "known": float(m.group(5)), "none": float(m.group(6)), "modules": det})
        elif "stopping at step" in ln or "time limit" in ln:
            out["stopped"] = ln[16:].strip()
    k = max(1, len(out["trace"]) // 200); out["trace"] = out["trace"][::k]
    return out


def tests(module, tag):
    """App-threshold (0.9) overall row of ioi_check and the centroid start error of timing_check."""
    res = {}
    for f in glob.glob(f"{L}/modtest_ioi_{module}_{tag}.out") + glob.glob(f"{L}/mod_plain_*.out" if module == "plain" else ""):
        for ln in open(f, errors="ignore"):
            if (m := OVERALL.match(ln)):
                res.update(recall=float(m.group(1)), precision=float(m.group(2)), f1=float(m.group(3)), dbl=float(m.group(4)))
    for f in glob.glob(f"{L}/modtest_timing_{module}_{tag}.out") + glob.glob(f"{L}/mod_plain_*.out" if module == "plain" else ""):
        for ln in open(f, errors="ignore"):
            if (m := TIMING.search(ln)):
                res.update(start_median=float(m.group(1)), start_abs=float(m.group(2)), within8=float(m.group(3)))
                break
    return res or None


def lane_logs():
    """The log of each lane's CURRENT job: the queued/running one if there is one (None until it writes),
    else the newest finished one that wasn't cancelled (a cancelled run was replaced and is stale)."""
    queued = {}
    for line in sh(f"squeue -u {USER} -h -o '%i|%j'").splitlines():
        jid, name = line.split("|")
        if name.startswith("mod-lane"):
            queued[name[8:]] = jid
    cancelled = {l.split("|")[0] for l in sh(f"sacct -u {USER} -S now-3days -X -n -P -o jobid,state").splitlines()
                 if "CANCELLED" in l}
    out = {}
    for li in range(len(LANES)):
        if str(li) in queued:
            f = f"{L}/mod_lane{li}_{queued[str(li)]}.out"
            out[li] = f if os.path.exists(f) else None
            continue
        fs = [f for f in glob.glob(f"{L}/mod_lane{li}_*.out") if re.search(r"_(\d+)\.out$", f).group(1) not in cancelled]
        out[li] = max(fs, key=lambda f: int(re.search(r"_(\d+)\.out$", f).group(1))) if fs else None
    return out


def gpu_types():
    """node -> GPU type (v100-sxm2, a100, ...) for the GPU partitions."""
    out = {}
    for line in sh("sinfo -p gpu,courses-gpu -h -N -o '%N|%G'").splitlines():
        node, g = line.split("|")
        m = re.match(r"gpu:([\w-]+):", g)
        if m:
            out[node] = m.group(1)
    return out


def alerts():
    """Things that need a person: failed shards with no later success, failed pipeline jobs, jobs whose
    dependency can never be met, crashes in a training log, and running lanes whose log has gone quiet."""
    out = []
    done, bad = set(), {}
    rows = [(line.split("|") + [""] * 5)[:5] for line in
            sh(f"sacct -u {USER} -S now-3days -X -n -P -o jobid,jobname,state,nodelist,start").splitlines()]
    # a lane Slurm cancelled before it ever started (its dependency failed) and that no later job replaced
    for jid, name, state, node, start in rows:
        if name.startswith("mod-") and state == "CANCELLED" and start in ("None", "Unknown", ""):  # "CANCELLED by <uid>" = cancelled by hand
            later = [r for r in rows if r[1] == name and r[0].isdigit() and jid.isdigit() and int(r[0]) > int(jid)]
            if not later:
                out.append({"key": f"never:{jid}", "text": f"{name} ({jid}) was cancelled before it started: a job it waited on failed"})
    for jid, name, state, node, start in rows:
        if name.startswith("mr-"):
            key = (name, jid.split("_")[-1])
            if state == "COMPLETED":
                done.add(key)
            elif state.split()[0] in BAD:
                bad[key] = (jid, state, node)
        elif name.startswith("mod-") and state.split()[0] in BAD:
            out.append({"key": f"job:{jid}", "text": f"{name} ({jid}) ended {state} on {node}"})
    for key, (jid, state, node) in bad.items():
        if key not in done:
            out.append({"key": f"shard:{jid}", "text": f"{key[0]} shard {key[1]} ({jid}) {state} on {node}, no successful rerun yet"})
    for line in sh(f"squeue -u {USER} -h -o '%i|%j|%r'").splitlines():
        jid, name, why = line.split("|")
        if "DependencyNeverSatisfied" in why:
            out.append({"key": f"dep:{jid}", "text": f"{name} ({jid}) can never start: a job it waits on failed"})
    for li, path in lane_logs().items():
        if not path:
            continue
        txt = open(path, errors="ignore").read()
        if "Traceback" in txt:
            last = [l for l in txt.splitlines() if l.strip()][-1][:200]
            out.append({"key": f"crash:{path}", "text": f"mod-lane{li} log has a Python error: {last}"})
        jid = re.search(r"_(\d+)\.out$", path).group(1)
        running = sh(f"squeue -h -j {jid} -o %T 2>/dev/null").strip() == "RUNNING"
        idle = (time.time() - os.path.getmtime(path)) / 60
        if running and idle > STALL_MIN:
            # keyed per hour so a long stall is re-sent hourly, not every 5 min
            out.append({"key": f"stall:{jid}:{int(time.time() // 3600)}",
                        "text": f"mod-lane{li} ({jid}) running but its log hasn't changed for {idle:.0f} min"})
    return out


NOTICE_HOURS = 6      # a finished stage stays on the page this long, unless already acknowledged (in sent.json)


def acked():
    """Stage-finished notices already sent: emailed once, then off the page (idle-GPU ones stay while true)."""
    try:
        return {k for k in json.load(open(f"{OUT}/sent.json")) if k.startswith("finished:")}
    except (OSError, ValueError):
        return set()


def notices():
    """Not failures, but someone should act: a stage just finished, or no GPU work of mine is queued at all
    (stage 5 finished on 2026-10-23 and the GPUs then sat idle for 3 hours unnoticed)."""
    out = []
    f = f"{CK}/joint/train.log"
    if os.path.exists(f) and (time.time() - os.path.getmtime(f)) / 3600 < NOTICE_HOURS:
        stop = [ln for ln in open(f, errors="ignore") if "stopping at step" in ln]
        if stop:
            out.append({"key": f"finished:joint:{stop[-1][:14]}", "text": f"stage 5 (joint refinement) finished: {stop[-1][16:].strip()}"})
    for ph in ("classify", "e2e"):
        f = f"{CK}/gate_{ph}/train.log"
        if os.path.exists(f) and (time.time() - os.path.getmtime(f)) / 3600 < NOTICE_HOURS:
            stop = [ln for ln in open(f, errors="ignore") if "stopping at step" in ln]
            if stop:
                out.append({"key": f"finished:gate_{ph}:{stop[-1][:14]}", "text": f"stage 6 gate ({ph} phase) finished: {stop[-1][16:].strip()}"})
    gpu = [ln.split("|") for ln in sh(f"squeue -u {USER} -h -p gpu,courses-gpu -o '%i|%j|%T'").splitlines()]
    if not [g for g in gpu if not g[1].startswith("ALERT")]:
        rows = [ln.split("|") for ln in sh(f"sacct -u {USER} -S now-1days -X -n -P -r gpu,courses-gpu "
                                           "-o jobid,jobname,state,end").splitlines()]
        rows = [r for r in rows if len(r) == 4 and not r[1].startswith("ALERT") and r[3] not in ("Unknown", "None")]
        if rows:
            jid, name, state, end = max(rows, key=lambda r: r[3])
            out.append({"key": f"idle:{jid}", "text": f"no GPU work queued: the last GPU job, {name} ({jid}), ended {state} at {end[11:16]}"})
    return out


def notify(snap):
    """Email alerts not sent before; remember them in sent.json."""
    path = f"{OUT}/sent.json"
    try:
        sent = set(json.load(open(path)))
    except (OSError, ValueError):
        sent = set()
    new = [a for a in snap["alerts"] + snap["notices"] if a["key"] not in sent]
    if not new:
        return
    # compute nodes have no `mail` and can't reach the login node, so let Slurm send it: a 1-minute job
    # named after the alert, with a "job began" email; the name is the subject line
    more = f" (+{len(new) - 1} more, see the progress page)" if len(new) > 1 else ""
    name = f"ALERT {new[0]['text']}{more}"[:200]
    r = subprocess.run(["sbatch", "-A", "cs7170.202710", "-p", "courses", "-c", "1", "--mem=100M", "-t", "1",
                        "--mail-type=BEGIN", f"--mail-user={MAIL_TO}", "-J", name, "-o", "/dev/null", "--wrap", "true"],
                       capture_output=True, text=True)
    if r.returncode == 0:
        json.dump(sorted(sent | {a["key"] for a in new}), open(path, "w"))


# ---- the NEW BASE plan (NEW_BASE_PLAN.md): renders -> base -> instrument adapters -> combining them
BASE_RUN = "base20m"                                     # checkpoints/<BASE_RUN>, logs basetest_*
FAMILIES = ["piano_keys", "epiano", "organ", "acoustic_guitar", "electric_guitar", "bass", "strings", "brass",
            "winds", "synth_lead", "synth_pad", "voice", "tuned_perc", "world"]
TSTEP = re.compile(r"^(\S+ \S+)\s+step\s+(\d+)\s+loss ([\d.]+).*?([\d.]+) steps/s")


def simple_run(path):
    """train.py's log (no 'clips:' line, no GPU column): steps, validations, stop line, all in one run."""
    if not os.path.exists(path):
        return None
    out = {"vals": [], "trace": [], "last": None, "stopped": None, "started": None}
    for ln in open(path, errors="ignore"):
        if out["started"] is None and ln[:2].isdigit():
            out["started"] = ln[:14]
        if (m := TSTEP.match(ln)):
            out["last"] = {"time": m.group(1), "step": int(m.group(2)), "loss": float(m.group(3)),
                           "speed": float(m.group(4)), "gpu": None}
            out["trace"].append([int(m.group(2)), float(m.group(3)), None, 0, float(m.group(4))])
        elif (m := VAL.match(ln)):
            out["vals"].append({"time": m.group(1), "step": int(m.group(2)), "f1": float(m.group(3)),
                                "p": float(m.group(4)), "r": float(m.group(5)), "frame": float(m.group(6))})
        elif "stopping at step" in ln or "time limit" in ln:
            out["stopped"] = ln[16:].strip()
    k = max(1, len(out["trace"]) // 200); out["trace"] = out["trace"][::k]
    return out


def render_tests(name):
    """Test outputs for one model: name(kind, set) -> log path, kind in ioi / timing / real (set None for real).
    Returns per render set (plain / combo / fast) recall, precision, F1, double presses, quick-note recall by gap
    and start error at the app's 0.9 bar, and the real-song line."""
    res = {}
    for V in ("mod_plain", "mod_combo", "mod_fast"):
        f = name("ioi", V)
        if not os.path.exists(f):
            continue
        r = {}
        txt = open(f, errors="ignore").read().splitlines()
        for i, ln in enumerate(txt):
            if ln.startswith("overall"):
                for l2 in txt[i + 1:i + 4]:
                    if (m := OVERALL.match(l2)):
                        r.update(recall=float(m.group(1)), precision=float(m.group(2)), f1=float(m.group(3)), dbl=float(m.group(4)))
        g = [l for l in txt if re.match(r"^\s+app 0\.9\s", l)]
        if g:
            r["quick"] = [float(x) for x in g[0].split()[3:9]]               # recall by gap: 16-32 .. 96-144 ms, >=144
        t = name("timing", V)
        if os.path.exists(t):
            for ln in open(t, errors="ignore"):
                if (m := TIMING.search(ln)):
                    r["start_abs"] = float(m.group(2)); break
        res[V] = r
    f = name("real", None)
    if os.path.exists(f):
        txt = open(f, errors="ignore").read().splitlines()
        marks = [k for k, l in enumerate(txt) if l.strip() == "-- all --"]
        for l in (txt[marks[-1] + 2:marks[-1] + 12] if marks else []):
            v = l.split()
            if len(v) > 14 and v[0] not in ("piano", "human") and v[1] != "adv":
                try:
                    res["real"] = {"strict": float(v[1]), "precision": float(v[2]), "recall": float(v[3]),
                                   "tolerant": float(v[4]), "pc": float(v[5]), "notes_per_s": float(v[-3])}
                    break
                except ValueError:
                    pass
    return res


def experiments():
    """The architecture tests on arr_scratch's renders: time-attention windows and token height."""
    names = {"A_global": "A: global attention", "B_layers": "B: per-layer windows",
             "C_soft": "C: learned soft windows", "D_halfoct": "D: half-octave tokens"}
    out = {}
    for n, label in names.items():
        tr = simple_run(f"{CK}/win_{n}/train.log")
        if tr is None:
            continue
        out[n] = {"label": label,
                  "train": {"last": tr["last"], "stopped": tr["stopped"], "checks": len(tr["vals"]),
                            "best": max([v["f1"] for v in tr["vals"]], default=None),
                            "vals": [[v["step"], v["f1"], v["p"], v["r"]] for v in tr["vals"]],
                            "trace": [[t[0], t[1], t[4]] for t in tr["trace"]]},             # step, loss, steps/s
                  "tests": render_tests(lambda k, V, n=n: f"{L}/winev_real_{n}.out" if k == "real" else f"{L}/winev_{k}_{n}_{V}.out")}
    return out


def named(prefix):
    return lambda k, V: f"{L}/{prefix}_real.out" if k == "real" else f"{L}/{prefix}_{k}_{V}.out"


def newbase():
    return {"run": BASE_RUN, "base": simple_run(f"{CK}/{BASE_RUN}/train.log"), "base_tests": render_tests(named("basetest")),
            "families": FAMILIES,
            "adapters": {f: merged_run(f"{CK}/inst_{f}/train.log") for f in FAMILIES},
            "adapter_tests": {f: render_tests(named(f"insttest_{f}")) for f in FAMILIES},
            "fusion": merged_run(f"{CK}/inst_fusion/train.log"), "fusion_tests": render_tests(named("fusiontest")),
            "experiments": experiments()}


def main():
    plain_logs = sorted(glob.glob(f"{L}/mod_plain_*.out"), key=os.path.getmtime)
    plain_runs = parse_train(plain_logs[-1] if plain_logs else None) or []
    modules = {}
    for li, lane in enumerate(LANES):
        for m in lane:
            # each module's own log in its checkpoint folder: it can't be confused with a lane-mate, and
            # a resumed run appends to it, so the history stays whole
            modules[m] = {"lane": li, "train": merged_run(f"{CK}/mod_{m}/train.log"),
                          "test": tests(m, f"mod_{m}_best"), "baseline": tests(m, "checkpoints_plain_base")}
    snap = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "jobs": jobs(),
        "renders": renders(),
        "plain": {"train": plain_runs[-1] if plain_runs else None, "test": tests("plain", ""),
                  "merged": os.path.exists(os.path.expanduser("~/arrangement_transformer/checkpoints/plain_base.pt"))},
        "modules": modules,
        "lanes": LANES,
        "joint": joint(),
        "newbase": newbase(),
        "gate": {"classify": gate_phase("classify"), "e2e": gate_phase("e2e")},
        "alerts": alerts(),
        "notices": [n for n in notices() if n["key"] not in acked()],
    }
    if "--write" in sys.argv:
        os.makedirs(OUT, exist_ok=True)
        tmp = f"{OUT}/latest.json.tmp"
        json.dump(snap, open(tmp, "w"))
        os.replace(tmp, f"{OUT}/latest.json")          # readers never see a half-written file
        with open(f"{OUT}/history.jsonl", "a") as f:
            f.write(json.dumps({"time": snap["time"], "renders": {k: v["done"] for k, v in snap["renders"].items()},
                                "jobs": [[j["name"], j["state"]] for j in snap["jobs"]], "alerts": [a["key"] for a in snap["alerts"]],
                                "modules": {m: {"step": ((d["train"] or {}).get("last") or {}).get("step"),
                                                "f1": (d["test"] or {}).get("f1")} for m, d in snap["modules"].items()}}) + "\n")
        notify(snap)
    else:
        print(json.dumps(snap))


if __name__ == "__main__":
    main()
