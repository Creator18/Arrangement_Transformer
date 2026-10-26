# Arrangement Transformer: hand-off notes for Claude Code

Read this first. It says what the project is, what is running on the cluster right now, how to check it, and the
rules the user has set. Last updated **2026-10-25** (branch `feat/song-to-piano`).

## What this is

A web app that turns a song into a **piano arrangement**: FastAPI backend (`backend/`), React + Vite frontend
(`frontend/`). The core is a spectral transformer ("arranger"): audio -> HCQT features -> tokens -> per-frame,
per-key note start / held / velocity probabilities -> decoded notes. Models are trained on the Northeastern
**Explorer** cluster from **rendered** audio (arrangement MIDI played by soundfont instruments).

## The current plan (decided 2026-10-24): `NEW_BASE_PLAN.md`

renders -> **new base** (`base20m`) -> 14 **instrument adapters** -> **combining them** (attention over adapter
outputs, no bundle adjustment) -> final comparison. Details and every decision are in `NEW_BASE_PLAN.md`.

- **Data, renders only (no real songs in training):** POP909 + Music2MIDI *training* arrangements, rendered by
  `backend/eval/render_base.py` (wraps `render_arrangements.py`): arrangement + extra parts + drums, 14
  instrument families from 3 soundfonts (all banks), synth-style tone (no pitch effects), loud mastering, lead
  melodies, 25% sped up. Target = the arrangement at the **written** times. **Done:** 227,860 train + 34,262
  validation clips in `/scratch/basu.ang/base_render`.
- **Base:** ~19.9M params (d 384, depth 8, heads 6), **half-octave tokens** (`--patch-h 18`), **learned soft time
  windows** (`--time-windows s,0,s,0,s,0,s,0`), **soft onset target** (`--onset-neighbor-weight 0.3`), front end
  calibrated on 400 new renders, 120k steps.

## What is running on the cluster (as of 2026-10-25 ~17:00 EDT)

**Nothing but `progress-snap`** (scrontab, 5-minute snapshot + email alerts, see below). The base finished:
`base-fast` (A100, 11363201) trained to 120,000 steps at 14:27; `basetest` (11363261) ran the render tests and the
106 real songs (`logs/basetest_*`, `/scratch/basu.ang/eval_base20m.json`). D's tests (`winev_*D_halfoct*`) finished
too and were not worse than A, so the base was kept.

Quirk seen at the end: both V100 spares (`base-train-2/3`) still started, resumed at step 120,001, ran one
validation and exited, about a minute of GPU each. The "exit at once if finished" check in `base_fast.sh` is not
quite working.

Job IDs change if anything is resubmitted: always check `squeue -u basu.ang`.

**Reading the base's validation:** it is scored on the NEW renders, which are much harder (loud masters, more
instruments, lead lines, extra parts). Its early numbers look low, but fully trained C scores the same on those clips
(note F1 0.391 at the 0.5 bar, P 0.274 / R 0.684; 0.514 at the app's 0.9 bar). The base matched that by step 6,000.
Compare against C on the same clips, never against the old 0.8+ numbers.

## How to check status (from any machine with the `explorer` ssh alias)

```bash
ssh explorer 'squeue -u basu.ang -o "%i %j %T %M %R"'                      # what is queued / running
ssh explorer 'tail -5 ~/arrangement_transformer/checkpoints/base20m/train.log' # base training (VAL lines = validation)
ssh explorer 'cat ~/arrangement_transformer/progress/latest.json' | python3 -m json.tool | head -100   # full snapshot
ssh explorer 'ls -t ~/arrangement_transformer/logs | head'                    # newest logs
```

- **Snapshot:** `backend/eval/progress_dump.py --write` runs every 5 min (scrontab, `eval/progress_scrontab.txt`)
  and writes `~/arrangement_transformer/progress/{latest.json,history.jsonl,sent.json}`. It emails new alerts
  (failed shards / jobs, stuck lanes, a stage finished, no GPU work queued) to the user via a 1-minute Slurm job.
  Add a key to `sent.json` to stop an alert being emailed; add it to `resolved.json` once it has been dealt with
  (it then leaves the page too).
- **Progress page:** run the backend + frontend locally, open `http://localhost:5173/progress`. The backend's
  `/progress-data` reads `latest.json` over `ssh explorer` (`backend/app/routers/progress.py`).
- **Test outputs per model:** `logs/<prefix>_ioi_<set>.out` (recall / precision / F1 / double presses at the
  app's 0.9 bar, quick-note recall by gap), `logs/<prefix>_timing_<set>.out` (start error),
  `logs/<prefix>_real.out` (106 held-out Music2MIDI real songs: strict / tolerant / pitch-class F1).
  Sets `mod_plain`, `mod_combo`, `mod_fast` live in `/scratch/basu.ang/mod_*/validation`.

## Cluster facts

- `ssh explorer`; env `module load miniconda3/25.9.1; source activate ~/envs/arr`; code `~/arrangement_transformer/backend`;
  checkpoints `~/arrangement_transformer/checkpoints/`; logs `~/arrangement_transformer/logs/`; data `/scratch/basu.ang/`.
- Partitions: `gpu` (8 h max, ~4 running per user), `courses` (CPU, ~1000 cores) and `courses-gpu` (2 jobs) with
  `-A cs7170.202710`. **GPU minimum is V100**: always `-x c2184,c2185,c2186,c2187,c2188,c2193,c2194,c2195,d1025`
  (P100s + a T4). Don't run heavy Python on the login node (it gets killed); use `srun`/`sbatch`.
- **Submit with `backend/eval/sbatch_once.sh -J <name> ...`, never a blind sbatch retry:** the controller often
  replies "Unexpected message received" yet accepts the job, and retries created duplicates that corrupted data.
- **Faster GPU rule:** V100 minimum, but move to an A100 / H200 when one frees up. For train.py runs use
  `eval/gpu_train.sh` (or `eval/base_fast.sh` for an already-running chain); the trainer saves and exits on a `HANDOVER` file.
- 2026-10-25: an A100 job was killed (signal 9, 23.7 of 64 GB used, no error) 11 min after a handover; the old
  handover had cancelled the V100 spares, so the run sat idle ~7 h. `base_fast.sh` now makes the spares wait for it
  instead, and alerts cover base-/win-/inst jobs. Resumed from step 26,000.
- Node `d0135` hung 31 render shards with no output (2026-10-24); exclude it if it misbehaves again.
- Shell scripts must have LF line endings (CRLF broke them before).

## Results so far (the 0.9 app bar)

| Model | val F1 | combo renders F1 | double /1000 | real strict | real P / R | notes/s (human 8.0) |
|---|---|---|---|---|---|---|
| A global attention (`win_A_global`) | 0.813 | 0.785 | 121 | 0.248 | 0.234 / 0.279 | 9.1 |
| B per-layer windows (`win_B_layers`) | 0.823 | 0.790 | 115 | 0.252 | 0.228 / 0.296 | 10.3 |
| **C learned soft windows** (`win_C_soft`) | 0.832 | 0.793 | **102** | 0.245 | **0.241** / 0.265 | **8.3** |
| **D half-octave tokens** (`win_D_halfoct`) | **0.848** | **0.804** | 105 | 0.249 | 0.239 / 0.273 | 8.8 |
| **Base** (`base20m`, step 112k) | n/a (new renders) | **0.867** | **32** | **0.251** | 0.228 / **0.298** | 10.9 |

- **The base (2026-10-25):** the user judged it the best by ear, "bridging the gap between C and D and doing the
  good things from both". On metrics: big win on dense mixes (combo F1 +6 pts over D, a third of the double
  presses), ~2 pts behind D on plain (0.937 vs 0.953) and fast (0.912 vs 0.933) piano renders (it misses quick
  notes), level on real songs, and it over-plays (10.9 notes/s). Val note F1 on the new renders 0.568 at the 0.5 bar
  (C: 0.391). Full tables in `NEW_BASE_PLAN.md` ("Base result").
- All real-song runs above used the 0.90 threshold (tuned on validation songs by `s2p_evaluate.py`).

- By ear: **C** clearly beat A / arr_scratch (alternating attention "improved a lot", vocals much better), and
  **D beat C** (2026-10-25). D is hosted locally from `backend/app/ml/spectral/checkpoints/win_D_halfoct/best.pt`;
  D also leads on every render set (plain .953, fast .933, quick notes +3-5 pts).
- Ruled out: time stretching dense passages (hurt everywhere, `eval/stretch_oracle.py`), finer time resolution,
  8 kHz band limit (no effect). Real songs sat ~0.8 spreads louder / 1.7x wider than old renders at the front end;
  the new renders' mastering closes most of that (`eval/fe_shift.py`).
- **Centroid onset timing** is on in the app (`spectral_transcription.centroid_onsets`, halves start error on
  renders; user heard it help). `ARRANGEMENT_ONSET_CENTROID=0` turns it off.
- Earlier lines (keep for reference, not active): the modular effect adapters + joint refinement + gate
  (`render_modular.py`, `joint_refine.py`, `gate.py`; paused: weak on real songs), the attack / fast / singer
  LoRA adapters on `arr_tone_full` (`launch_*.sh`).

## The user's rules and preferences

- **Targets are always the written times**, never where the sound starts (the old piano renders shifted labels by
  the sampler latency; `render_base.py` disables that).
- **Renders only** for training; real songs (Music2MIDI test songs) are for evaluation only. POP909 has no audio;
  Music2MIDI audio on disk is 16 kHz mono.
- **Never add or remove arrangement notes in a render** except as non-target "extra parts"; no pitch effects
  (detuned / chorus *presets* are fine as instrument sounds).
- **Ear beats metric.** The user listens in the app at the **0.9** onset bar; host candidates for them.
  Host servers in a **terminal tab** (`backend/serve.ps1 -Arranger <ckpt>` on Windows; on a Mac run the same
  env vars with `python -m uvicorn app.main:app --port 8000`), not the preview pane (the app stops those).
- No 5-octave converter: present the arranger's raw 7-octave output.
- Ask before downloading anything; never handle the user's passwords or keys.
- High, efficient cluster use: parallelise, no idle waiter jobs, CPU for scoring, GPU for training / inference.
- Explain results plainly, with tables; keep the progress page and `NEW_BASE_PLAN.md` up to date.

## Hosting on the Mac (set up 2026-10-25)

`backend/venv` (Python 3.11 via `uv`), `frontend/node_modules`, ffmpeg (Homebrew), the base at
`backend/app/ml/spectral/checkpoints/base20m/best.pt` and `SalamanderGrandPiano-master/` (copied from the cluster)
are all in place and git-ignored. Gotchas: madmom needs `uv pip install Cython` then `--no-build-isolation
"madmom @ git+https://github.com/CPJKU/madmom"`; pretty_midi needs `setuptools<81` (pkg_resources); `pysfizz` gives
the Salamander piano; yt-dlp (link fetch) is now in `requirements.txt`. C and D checkpoints are not on the Mac.

```bash
cd backend && ARRANGEMENT_PIANO_MODE=spectral ARRANGEMENT_PIANO_ONLY=1 ARRANGEMENT_SONG_ARRANGER=app/ml/spectral/checkpoints/base20m/best.pt ARRANGEMENT_SONG_ARRANGER_THR=0.9 ARRANGEMENT_SONG_ARRANGER_FRAME_THR=0.5 ARRANGEMENT_FORCE_SONG=1 ARRANGEMENT_SPECTRAL_ONSET_THR=0.85 venv/bin/python -m uvicorn app.main:app --port 8000
cd frontend && npm run dev
```

## Next steps

1. **Voice** (`VOICE_PLAN.md`): build the vocal-line score on the 106 real songs (base, C, D), including the
   probability-shape diagnostic that decides whether a generative (flow-matching) melody decoder is worth it.
2. Then the **voice adapter** on the frozen base, rendered with the 1,594 cloned VocalSet singers (the base never
   heard them). Open decision for the user: allow natural vibrato / scoops / early-late entries on singer stems
   (targets still the written notes).
3. Learned note timing (`NEW_BASE_PLAN.md` section 4b): the small learned decoder (exact start times, merged double
   strikes), tested on the base and C, then a timing output inside the model.
4. Then the other instrument-adapter renders and adapters (`NEW_BASE_PLAN.md` section 3), then the combining layer.
5. Optional: the base over-plays on real songs (10.9 notes/s vs 8.0); try a higher onset bar (0.93-0.95) by ear.
