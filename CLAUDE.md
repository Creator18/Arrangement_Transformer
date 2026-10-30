# Arrangement Transformer: hand-off notes for Claude Code

Read this first. It says what the project is, what is running on the cluster right now, how to check it, and the
rules the user has set. Last updated **2026-10-30** (branch `feat/song-to-piano`).

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

## BEST so far, by ear (user, 2026-10-28: "beautiful")

Base at **0.8** + clean-up **v3** (keep 0.3, **merge 0.5**) + **sustain**, **NO dynamics / level**:
`serve.ps1 -Arranger app/ml/spectral/checkpoints/base20m/best.pt -Threshold 0.8 -Cleanup app/ml/spectral/checkpoints/cleanup3/best.pt -Keep 0.3 -Merge 0.5 -Sustain`
(Mac: the same env vars, `ARRANGEMENT_CLEANUP_MERGE=0.5`, `ARRANGEMENT_SUSTAIN=1`, no `ARRANGEMENT_DYNAMICS`; copy
`checkpoints/cleanup3/best.pt` from the cluster). Reference song: `backend/temp/f2af9154-nolevel` (`/results/f2af9154-nolevel`).
- The level stretch (`-Dynamics`) was rejected: the base's velocities follow the MIX loudness, and the 2.6x stretch turns
  whole quieter passages (fewer singers) down instead of shaping notes. User wants **per-note velocity** ("how hard
  should this note be struck") instead: a learned model on the written velocities, later.
- **Clean-up v3** (`app/ml/spectral/cleanup3_train.py`, `checkpoints/cleanup3/best.pt` step 10k; jobs `cutrain3-slow/fast`,
  V100 -> A100 handover via `eval/cleanup3_run.sh` / `cleanup3_submit.sh`): v2 + a merge output trained on
  `eval/merge_oracle.py --labels` (`/scratch/basu.ang/merge_labels`). All 3,212 held-out clips, F1 with ends: v2 keep 0.3
  0.519; v3 keep 0.3 + merge 0.5 **0.526**; keep 0.4 + merge 0.5 0.534; merge 0.3 0.512 (joins real repeats); ceiling 0.619.
  Small on metrics, but this is the setup the user called the best. Next for merging: the held-note probabilities
  across the gap between two strikes as input.
- Link fetch: `app/routers/upload.py` now moves a dead googlevideo server onto a backup server from the link's `mn`
  list (2026-10-28: two of the ISP's YouTube cache nodes timed out).

## Slides, and "which of our notes do humans never keep" (2026-10-28, running)

- User: singer / instrument **slides** show up as quick out-of-key notes next to the real one ("flickers"); renders have
  no pitch bends, so the base and clean-up never learned to ignore them. (Not the main cause of double presses: those
  happen on renders too, and in song f2af9154 they sit mostly in the bass.)
- **User's idea:** compare our hosted output with ALL human arrangements of a song and learn only which notes humans
  NEVER keep (drop-only; the clean-up may learn from real songs, like the solo detector; the base stays renders-only).
- **Measurement first** `eval/never_kept.py` (`never_kept_run.sh` GPU jobs `nevkept` / `nevkept-c`, then
  `never_kept_score.sh` on courses, then `--report`; dir `/scratch/basu.ang/never_kept`, frozen script `frozen.py`):
  925 Music2MIDI songs with audio, 1,520 arrangements (271 songs have 2-17). Kept = any arrangement has the pitch
  class within 150 ms; arrangements gated at |lag| <= 0.5 s. Control: one human arrangement vs the others.
  Smoke test (2 songs): notes/s ours 12.6 raw / 9.2 after clean-up vs humans 7.5; slide-shaped + out-of-key notes are
  1.7% of ours (0.4% of a human's), ~72% never kept; humans disagree a lot (~40% of one arranger's notes not in the others).
- **Beats split off the GPU jobs (2026-10-28):** the cluster cancelled shards 0, 1, 3, 6 for idling. Profiled: 79 of 82 s
  a song is madmom (one CPU core) for clean-up v3; the GPU sat at ~0%. Now `--beats` (`never_kept_beats.sh`, CPU on
  courses, job `nevbeats` 11452752, 40 tasks) caches madmom per song in `/scratch/basu.ang/never_kept/beats`, and
  `--run --cached-only` (`never_kept_run.sh`) reads it: one song 32 s incl. model loading (was ~200 s). Shards 0, 1, 3, 6
  queued after the beats (`nevkept-c` 11452778, courses-gpu). Shards 2, 4, 5, 7 still run the old way (code loaded
  before the change); if they get cancelled too, resubmit them with the new script. **Any GPU job that runs the
  clean-up v2/v3 chain should take cached beats, never madmom inside the GPU job.**
- **Result (2026-10-29, all 925 songs; 1,318 of 1,520 arrangements passed the lag gate, 85 songs had none):**
  notes/s ours 14.6 raw / 10.9 after clean-up vs humans 6.9 median. Fair test (songs with 2+ arrangements, all but one
  as reference): our notes never kept 32.3% vs the left-out human's 25.2% (the arrangers' own disagreement floor).
  The excess sits where the voice sounds (35.6% vs human 26.3%; voice silent 25.0% vs 23.1%). Slide-shaped + out of
  key: 1.7% of our notes (human 0.3%), **84.5% never kept** (human 46%); all slide-shaped 8.8% (human 2.6%), 54% never
  kept; out of key 52% vs 32%; short < 100 ms 17% of ours vs 5% of a human's. Double presses are NOT visible here
  (pitch class within 150 ms lets either strike match). Clean-up already drops the foreign ones more (45% vs 34%
  never kept). Labels get cleaner with more arrangements: never kept 37% (1 arrangement) -> 19% (3+) -> 9% (9+).
- If it holds up: a "humans never keep" output on clean-up v3 (split by song audio). Alternative / complement: renders
  with scoops, slides and vibrato on voice and lead instruments, for clean-up data only (needs the user's OK: exception
  to "no pitch effects").
- Slide rule (`app/services/slides.py`, `serve.ps1 -Slides`): drop-only, short + out of local key + next to a longer note
  1-2 keys away. A/B `/results/f2af9154-nolevel` vs `/results/f2af9154-noslide` (57 notes). Off in the current hosting.

## CURRENT hosted setup (2026-10-30): clean-up v4, "not too bad after testing multiple times" (user)

`serve.ps1 -Arranger app/ml/spectral/checkpoints/base20m/best.pt -Threshold 0.8 -Cleanup app/ml/spectral/checkpoints/cleanup4/best.pt -Keep 0.3 -Merge 0.5 -NK 0.7 -Sustain`
**User (2026-10-30): keep this as it is; next step not decided yet; draft a paper of everything so far first.**
- **v4** (`app/ml/spectral/cleanup4_train.py`, `checkpoints/cleanup4/best.pt` = step 6k, also on the laptop): v3 + a
  "no human keeps this" output (nk), drop-only after keep / merge when p_nk >= `ARRANGEMENT_CLEANUP_NK` (0.7). Each step
  = a render batch (v3 losses) + a real-song batch (nk BCE, weight min(3, nref)/2). Data `eval/human_data.py` ->
  `/scratch/basu.ang/human_data` (840 songs: our raw 0.8 notes + v3 features + labels from never_kept; 89 songs held out
  = split "test"). Job `cutrain4` (courses-gpu, `eval/cleanup4_run.sh`, BUDGET=1380 runs it in one piece).
- Held-out 89 songs (AUC 0.72; never kept 34% of our notes): bar 0.9 drops 1.6% (86% of them never kept), 0.8 3.8%
  (83%, 1.0% of kept notes lost), **0.7 7.8% (75%; 3.0% lost), 11.4 -> 10.5 notes/s**, 0.6 14.5% (66%; 7.4% lost),
  0.5 23.7% (61%; 14.6% lost). Renders unchanged (keep 0.3 merge 0.5 F1 with ends 0.532 vs v3 0.534).

### Humans on merges and density (2026-10-30, `eval/human_merge.py`, `/scratch/basu.ang/human_merge{,/samekey}`)

Our same-key pairs (B starts <= 150 ms after A ends; 1.25M pairs, 840 songs) vs the arrangements that play A; a pair
is "single" if no such arrangement re-strikes at B; chance = the same test 370 ms later. `--samekey` (re-strike must
be the same key as the human's A note) is the readable version:

| start-to-start | share of pairs | v3 merges | humans single | chance | excess |
|---|---|---|---|---|---|
| 0-100 ms | 8% | 29% | 84% | 62% | +22 |
| 100-150 | 16% | 21% | 73% | 57% | +16 |
| 150-200 | 20% | 20% | 67% | 58% | +9 |
| 200-300 | 25% | 20% | 60% | 60% | 0 |
| 300-600 | 23% | 18% | 54% | 61% | -8 (real repeats) |
| 600+ | 8% | 18% | 52% | 79% | -27 (real repeats) |

Humans' own pairs vs the other arrangers: single 33% vs chance 56% (the test works on human notes). Loudness, length,
register, voice barely move the excess; v3's merge prob vs "single" AUC 0.62. Pitch-class version (any octave) is
saturated (chance ~60% of random moments have the pitch class within 150 ms).
Density per 10 s window (18,323 windows): notes/s ours raw 12.8 / after clean-up 9.9 / human 7.0; human / ours after
clean-up by our density quartile 0.98 / 0.78 / 0.67 / 0.60; shape across a song (corr. with the other arrangements)
human 0.77, ours 0.76: we put density in the right places, we only keep too much in busy parts.

**Solo piano through the song path (2026-10-30):** `serve.ps1` forces every upload onto the song path
(`ARRANGEMENT_FORCE_SONG=1`), piano included. On a piano upload (session e0043c18, detector said piano_only) the chain
kept 1,624 of 1,747 notes (57 dropped, 64 merged, 2 never-kept drops). User: "the output felt exactly the same as the
piano song". So no routing change is needed.

**Paper (2026-10-30):** draft in `paper/draft.md` (HTML: `python paper/build_html.py` -> `paper/draft.html`, figures in
`paper/figures.py`). **User: ablations are still remaining** (list in the draft, section "10b Ablations still to run"):
pipeline stage removal, clean-up inputs (incl. madmom), never-kept label variants, render-recipe parts, soft onset
target, and baselines vs published song-to-piano systems (downloads need the user's OK). None started.

**Parked ideas (NOT decided; user wants to think first):** (1) merge by distance: easier merge < 150 ms, v3 bar
150-300 ms, never > 300 ms (check on held-out renders with F1 with ends, then by ear); (2) density-aware never-kept bar
that tightens in windows much busier than the song's usual; (3) per-note velocity; (4) beat heads on the base (paused);
(5) slide renders for clean-up data (needs OK on pitch effects); (6) user's idea (2026-10-30, "keep in mind, don't
change anything now"): let the user pick a start / end time to transcribe on link fetch (yt-dlp `--download-sections`)
and on file upload (waveform with two handles); pad a few seconds each side then trim, start notation at the first
downbeat, show original-song times, export MIDI from 0.
**Hosting (2026-10-30, not started, user wants it):** public demo, free, not from the laptop -> HF Space free CPU (2 vCPU)
+ weights in a PRIVATE HF model repo read via a secret token + frontend on Vercel; no link fetch on the server, length
cap, one job at a time, sessions expire. **True CPU timing** (laptop, 2 threads, GPU hidden with
`CUDA_VISIBLE_DEVICES=-1`; an EMPTY value does not hide it on Windows, so an earlier "56 s CPU" figure was the GPU):
3:08 song v4 156 s, v4 no-beats 137 s (GPU: 56 s vs 12 s). Expect ~3-5 min per 3-min song on HF's free 2 vCPU:
the start/end range picker matters for the demo.
**v4 without beat inputs** (`--no-beats`, `checkpoints/cleanup4_nobeat/best.pt` = step 8k, job `cutrain4nb`, init v4):
step 0 (beats just removed) renders F1 0.775 -> 0.753, merging stops, AUC 0.717 -> 0.674; after 2k steps everything
back (final: F1 0.777 / ends 0.534, AUC 0.721; bar 0.7 drops a bit more: 9.9 notes/s, 4.3% kept lost vs 3.0%).
madmom not needed. **Hosted 2026-10-30 for the user's ear** (tab "backend v4 no-beats"); A/B on the user's latest song:
`/results/f4907c52-cf7c-4634-a613-8b59cfc970a4` (v4, 1,959 notes) vs `/results/f4907c52-nobeat` (1,842).

## Our own beat / downbeat / tempo model (started 2026-10-28; replaces madmom, the user's priority)

- **Data** `eval/beat_labels.py` (jobs `beatlab` 0-89, courses) -> `/scratch/basu.ang/beat_data` (31 GB): POP909 render
  clips only (exact beats + downbeats from beat_midi.txt; the renders' drums sit on the same grid; Music2MIDI beats were
  only estimated, so left out). Speed recovered per song render (1.0 / 1.2 seen). 115,147 train clips (728 songs),
  14,089 held-out (90 songs); 98.5% of clips labelled. 64-band log-mel, 100 fps, float16 memmaps.
- **Model** `app/ml/spectral/beat_train.py`: multi-task TCN (2.3M params; beat, downbeat per frame, tempo 30-300 BPM),
  time-scale 0.8-1.25x augmentation. **Decoder ours** `app/ml/spectral/beat_decode.py` (bar-pointer Viterbi for
  beats, meter 3/4 Viterbi for downbeats). Job `beattrain` (`eval/beat_run.sh`: < 2 h pieces on gpu / gpu-short,
  resubmits itself) -> `checkpoints/beat1`.
- **Baseline** `eval/beat_compare.py` (jobs `beatcmp`): madmom on the same 600 held-out clips -> `/scratch/basu.ang/beat_compare`;
  merge with `--merge madmom_*.json`; ours with `--ckpt`.
- Then: swap madmom out of `app/services/cleanup.py` / `eval/clip_beats.py`, recompute the clip beats, retrain clean-up v3.
- **PAUSED by the user (2026-10-28), pick up later.** The standalone TCN above was NOT the agreed plan: on 2026-10-27
  we agreed a beat output trained on the renders is added to OUR model. Proposed: beat / downbeat / tempo heads on
  the frozen base (base20m), trained on `beat_data`'s POP909 labels; alternative: inside the clean-up (notes only, no
  audio: likely weaker). The user hasn't chosen yet. The TCN run (`beattrain`) was cancelled: its per-clip CPU data
  pipeline starved the GPU (< 200 steps in 12 min). madmom's numbers on the 600 held-out clips are done:
  `/scratch/basu.ang/beat_compare/madmom_*.json` (merge with `eval/beat_compare.py --merge`).

## What is running on the cluster (as of 2026-10-25 ~17:00 EDT)

**Nothing but `progress-snap`** (the solo detector finished 2026-10-26 04:30; scrontab, 5-minute snapshot + email alerts, see below). The base finished:
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

## Voice: the solo-voice fix (2026-10-26)

The user noticed the base reads a voice well inside a mix but drops notes when the voice is alone. Tests on POP909
melodies sung as a synthetic neutral "aah" (`eval/aah_test.py`, `eval/aah_fill_test.py`, results in
`backend/eval/voice_tests.json`, charted on the progress page's "Voice" section):
- Solo "aah": base F1 0.883 (C 0.998, D 0.997: it is the base's weakness). **Fill and remove** (loop the song's own
  accompaniment bars, beat-cut and chord-matched, kept *below* the voice; transcribe; delete the added notes; never
  delete notes the solo pass found) -> 0.974. Singer-like voice 0.567 -> 0.929; with separation bleed + monophony 0.871.
- Rules can't find solo bars once anything sounds under the voice, so a **solo detector** was trained
  (`app/ml/spectral/solo_train.py`, ckpt `checkpoints/solo_det/best.pt`, step 24k): mix in, per-frame "the mix is
  basically the voice" out. Labels: Demucs on Music2MIDI (`eval/m2m_vocal_curves.py`, curves only, cluster) + MUSDB18-HQ
  true stems with the band turned down. Real songs train this detector by the user's decision; the arranger stays
  renders-only. Music2MIDI val AP 0.849 / F1 0.77; MUSDB natural AP 0.768.
- v1 counted a quiet band only half (soft target 6-18 dB). The user wants **both quiet and silent**: v2
  (`checkpoints/solo_det2`, job `solo-train3`) has three classes per frame: silent band (25 dB+ below the voice),
  quiet band (8-25 dB), neither.
- Demucs 4.1.0 is installed in `~/envs/arr` on the cluster (weights copied from the laptop). Don't run heavy jobs on the laptop.
- v2 finished (30k steps, `checkpoints/solo_det2/best.pt`): Music2MIDI val AP 0.893 (silent 0.629, quiet 0.839).

### Fill-and-remove on the 106 REAL test songs (`eval/solo_fill_real.py`, 2026-10-26)

Detector -> beats (librosa, 4/4) -> solo runs (>= 2 beats) -> plain pass of the base on the whole song -> fill notes
taken from the base's own accompaniment outside the solo runs (chord-matched bar, kept below the voice) -> played
under the solo runs -> second pass -> remove the added notes. Scored inside the TRUE solo parts (Demucs curves, 12 dB
line; 15.5 min, 2,087 sung notes in 75 songs) against the human arrangement's top line, **by pitch class within
150 ms**: human arrangers write the sung line an octave UP in solo passages (+12 in most matches), so exact pitch
reads ~0.

| run (dir in /scratch/basu.ang) | what | sung notes recovered |
|---|---|---|
| plain | the base alone | 26.5% |
| `solo_fill_real`, `_union` | pluck fill; replace / **union** (plain notes always kept: never worse) | 29.4% / 32.7% |
| `solo_fill_band` | user: the fill must sound like an instrumental mix. band (bass + keys/guitar + pad) / band+drums, union | 33.8% / **36.8%** |
| `solo_fill_rules` | band+drums union, v1 vs v2 regions, six removal rules | v1 33.8-36.8%, **v2 35.6-39.5%** |

Detector frames vs the true solo parts: v1 recall 0.81 / precision 0.74 (17 min flagged); v2 0.95 / 0.43 (34 min).
**Added notes are mostly not the voice:** only 27-37% match a sung note; fill leftovers 3-29% (octave-aware removal
clears them but also deletes sung notes, -3 pts); "other" (neither voice nor fill: drums, bleed, overtones) 43-64%.
Voice-floor removal is the cheapest cleanup (-1/4 of added notes for -1 pt).

**Pitch-track voice filter** (`/scratch/basu.ang/solo_fill_voice`, finished 2026-10-26 17:53; band+drums union):
pYIN on the mix inside the solo runs (there the mix is mostly the voice); keep an added note only if the sung line
sits within 1 semitone for >= 40% of 50-200 ms after its start (`-pc`: any octave); `voice all` also filters the
plain pass's notes in the solo runs (the only variant that can lose a plain note).

| | recovered | notes / sung note | added | voice | leftover | other |
|---|---|---|---|---|---|---|
| plain | 26.5% | 1.37 | | | | |
| v2 strict | **39.4%** | 2.04 | 1,404 | 27% | 29% | 44% |
| v2 strict + voice | 37.5% | 1.68 | 655 | 43% | 14% | 43% |
| v2 floor + voice-pc | 37.7% | 1.70 | 699 | 43% | 17% | 41% |
| **v2 floor + voice all** | 33.7% | **1.27** | 631 | 43% | 14% | 42% |
| v1 floor + voice | 34.4% | 1.57 | 429 | **47%** | 13% | 40% |

The voice filter halves the added notes for ~2 pts; `voice all` is the first variant better than plain on BOTH
recovery (+7 pts) and note count (1.27 vs 1.37). ~40% of added notes are still "other" (neither voice nor fill).
Not yet compared by ear, nor against the stem split below. Shard 2 was killed twice by signal 9 on node **c2205**
(1.4 GB used, not memory): exclude c2205. Each run uses a frozen copy of the script in its output dir (`frozen.py`), run with
`PYTHONPATH=~/arrangement_transformer/backend:~/arrangement_transformer/backend/eval`.
**sbatch_once.sh still makes duplicates** when the controller is slow (its 20 s check is too short): after every
submit, `squeue -u basu.ang` and cancel extra copies of a job name before they start.

## Onset bar 0.8 and the clean-up model (2026-10-27)

The user hears that **0.8 has the notes they want, voice included**; the app is hosted at 0.8 with no solo detector
for now (`serve.ps1 -Threshold 0.8`; the script now clears the solo env vars when `-Solo` is not given).
`eval/thr_review.py` (logs `thrrev_*`):

| bar | combo F1 | new renders F1 / sung recall | plain F1 | real pc-F1 / notes/s / sung line |
|---|---|---|---|---|
| 0.8 | 0.856 | 0.692 / 0.820 | 0.932 | 0.477 / 15.1 / 0.787 |
| 0.85 | 0.869 | 0.714 / 0.791 | 0.937 | 0.496 / 13.1 / 0.747 |
| 0.9 | 0.871 | 0.719 / 0.722 | 0.935 | 0.510 / 10.7 / 0.671 |

Notes 0.8 adds over 0.9 on renders: 28-44% right, the rest double presses, octave / overtone ghosts, neighbours.
Peak probability, length, loudness and "key held" barely separate right from wrong on real songs, so no rule cleanup.
**Clean-up model** (user's design): messy 0.8 output in, written arrangement as the answer; keep / drop per note
(never writes). Data = the base at 0.8 on the 34,262 base_render VALIDATION clips (228 songs, never trained on):
`eval/cleanup_data.py` -> `/scratch/basu.ang/cleanup_data` (jobs `cudata-0..7`). Model `app/ml/spectral/cleanup_train.py`
(note tokens + a +-3-frame window of the base's onset / frame probabilities at the key, neighbours, octaves, twelfths;
6-layer transformer; split by song) -> `checkpoints/cleanup1`, job `cutrain` (VAL lines compare with base 0.8 / 0.9).
Result: best at step 4,000 (then overfits: 228 songs): keep 0.4 F1 0.769 (R 0.711 / P 0.838) vs raw 0.9 0.751, raw 0.8
0.722. **By ear: "0.4 cleanup sounded amazing"** (song 134777de: 15.5 -> 9.7 notes/s). In the app:
`serve.ps1 -Threshold 0.8 -Cleanup app/ml/spectral/checkpoints/cleanup1/best.pt -Keep 0.4` (`app/services/cleanup.py`,
20 s windows every 10 s, centre-weighted). Progress page has its training-health panel (`/cleanup-data`).

### Clean-up v2: more data + learned quantizing (2026-10-27)

The user still hears excess double presses on some songs and wants beats/tempo so notes can be quantized, but **not
naive snapping** (notes sit at fractions / multiples of the beat, and some are deliberately off-grid). Checks first:
- `eval/grid_check.py`: POP909 written notes are 94% within 2% of a 16th or triplet slot (92% exact; median song 98%).
  Music2MIDI only 73% within 2% (23% exact: human-played timing) -> **position labels from POP909 clips only**.
- `eval/beat_check.py` (300 clips, true beats recovered from the source song): madmom (built from source in `~/envs/arr`,
  `~/src/madmom`) beat model puts 82% of true beats on its grid, 87% of its grid on true half-beats, usable on 67%;
  tempo x1 57% / x2 32% / x0.5 6%. Downbeat model similar (0.830 / 0.888 / 0.690). So madmom's beats are
  **features**, not a hard grid, and an OFF class covers notes (or clips) where the grid is wrong.
- Design: `app/ml/spectral/cleanup2_train.py` = v1 features + same-key context + madmom beat features (phase in the
  beat at 1, 1/2, 1/3, 1/4, tempo, distance to each slot, activations, bar phase) + a **position head**: the written
  start's slot in the madmom beat holding the base start, slots 0, 1/4, 1/3, 1/2, 2/3, 3/4, 1, or OFF (tolerance
  min(30 ms, 8% of the beat)). Loss = keep BCE + 0.5 x position CE.
- Data: v1's validation clips + 40 clips of each of the 1,522 TRAINING songs (`cleanup_data_tr`, 60,878 clips,
  jobs `cudtr-0..7`), madmom beats for all 95k clips in `/scratch/basu.ang/clip_beats` (`eval/clip_beats.py`).
- Smoke tests passed (`/scratch/basu.ang/cu2_smoke*`). Note: on renders the base's start is already within 3 ms of
  the written time (median), better than nearest-slot snapping (8 ms); the position head must earn its keep on
  real songs and by ear.
- **Result** (`cutrain2`, 60k steps, best step 38k, `checkpoints/cleanup2/best.pt`, also on the laptop): no overfitting
  but keep/drop barely moves (keep 0.4 F1 0.771 vs v1 0.769; P 0.852). Position head 89.5% right slot; start error
  vs written: base 3.7 ms, learned 8.0 ms, nearest-slot 10.0 ms (renders are already exact; real songs untested).
- **Hosted (2026-10-27):** `serve.ps1 -Threshold 0.8 -Cleanup app/ml/spectral/checkpoints/cleanup2/best.pt -Keep 0.4
  -Quantize` (`-QuantP` = min slot probability, default 0.5; never moves a note > 60 ms; same-key strikes landing
  together are merged). madmom on the whole song (laptop venv has pip madmom 0.16: `clip_beats._compat()` patches it
  for Python 3.11 / numpy 1.26). Song 134777de: 5,394 -> 3,454 notes, 3,041 moved, ~93 s.
- **User (2026-10-27): madmom is a stopgap; build our OWN learned beat / downbeat / tempo model next.** Renders give
  exact beat labels (known MIDI tempo, incl. the 25% sped-up ones); score it with `eval/beat_check.py` vs madmom, then
  retrain v2 on our beats.

### The user's stem split (2026-10-27, hosted)

User's idea: Demucs (`htdemucs_6s`, `separation._get_separator`) splits voice / rest; **voice at 0.8 + clean-up v2
keep 0.5**, **rest at 0.9**; voice notes on a key the rest strikes within 50 ms are dropped as duplicates; join;
quantize with v2's position head (madmom on the mix). `app/services/stem_split.py`, hooked at the top of
`spectral_transcription.transcribe`; `serve.ps1 ... -Cleanup app/ml/spectral/checkpoints/cleanup2/best.pt -Keep 0.4
-Quantize -Stems` (env `ARRANGEMENT_STEM_SPLIT=1`, `_VOCAL_THR` / `_VOCAL_KEEP` / `_REST_THR`). Stems are RMS-matched
to the mix (the base trained on loud masters). About +75 s per song on the laptop GPU.
- Song 57439b3b: 2,221 notes (mix 0.8 + v2 + quantize) -> 2,856 (voice 657 -> kept 560, 496 new; rest 2,373 at 0.9).
  Session copy `backend/temp/57439b3b-stems` (`/results/57439b3b-stems`). Waiting on the user's ear.
- My worry (base weak on a voice alone) was mostly the 0.9 bar: solo "aah" recall 0.81 at 0.9, 0.95 at 0.5.
- **Running (2026-10-27):** `stemsplit-0..3` (jobs 11409506 / 11409507 / 11409512 / 11409516, gpu) =
  `eval/stem_split_real.py` on the 106 real songs: mix 0.9 / mix 0.8 + v2 / stems / stems keep 0.4; strict + pitch-class
  F1, notes/s, sung line, re-strikes per 1000 notes. Parts in `/scratch/basu.ang/stem_split_real/part*.json`, logs
  `logs/stemsplit_*`. Merge: `python eval/stem_split_real.py --merge /scratch/basu.ang/stem_split_real/part*.json`.
  The cluster now has the 6-stem weights (`~/.cache/torch/hub/checkpoints/5c90dfd2-34c22ccb.th`, copied from the laptop).
- On the Mac: copy `checkpoints/cleanup2/best.pt` from the cluster to `backend/app/ml/spectral/checkpoints/cleanup2/`;
  the Mac's git madmom needs no patch; Demucs 6-stem weights download on first use unless copied
  (`~/.cache/torch/hub/checkpoints/5c90dfd2-34c22ccb.th` on the cluster) — ask the user before any download.

## Expression: dynamics and sustain (2026-10-27)

The user: "everything feels numerically accurate but it is not the song I remember". Two measured causes, both fixed
behind flags in `app/services/expression.py` (hooked in `transcription.py`'s song-arranger branch):

- **Dynamics (`ARRANGEMENT_DYNAMICS=1`).** The base's velocities were squeezed (one real song: p10 78 / median 83 /
  p90 93): squared-error velocity head, 40% of Music2MIDI targets a flat default 80, and stem gains / soundfonts /
  mastering in the renders break velocity vs loudness. Its ORDER is right, so each song is stretched to a pianist's
  spread: `64 + 14 x (v - median) / std`. Playback already honours velocity (Salamander velocity layers).
- **Sustain (`ARRANGEMENT_SUSTAIN=1`).** The song path sent no pedal (`_last_pedal = []`), so every sound stopped at
  its key release. Now: pedal down where the song's sound rings past its keys (CQT level of a released key stays
  within 40 dB of its peak), lifted when the bass moves. The sampler / previews already play a pedal track (CC64).
- **Bench** `eval/expression_bench.py` (`/scratch/basu.ang/expression_bench`): MAESTRO windows (30 s) rendered through
  sfizz + Salamander WITH the pianist's pedal, read by the hosted setup (0.8 + clean-up v2 keep 0.4); 29 validation
  clips to fit, 40 test clips to score (`eval/velocity_metrics.py`, `eval/pedal_metrics.py`, plus pedal-lift
  recall / precision within 100 ms).

| velocity (40 test clips) | spread ratio | Pearson | MAE | soft-note bias |
|---|---|---|---|---|
| base as it was | 0.73 | 0.808 | 11.9 | +17.1 |
| fixed x1.36 (MAESTRO only: too narrow on pop) | 0.98 | 0.808 | 10.4 | +0.7 |
| **per-song stretch to std 14 (on)** | **1.03** | 0.809 | **10.3** | **+1.2** |

  The key's CQT level and the song's loudness curve were fitted too and added nothing (the base hears loudness; it
  only squeezes it). Sustain: no rule beats "always down" on frame F1 (0.697; pianists hold it 61% of the time); ours
  0.62 (down 79%), lifts recall 0.45 / precision 0.24; lifting at pitch-class changes was worse (F1 0.5-0.6).
- Test song 10061056 in the app: velocities 52 / 64 / 89 (p10 / median / p90, range 12-109), 486 pedal presses.
  A/B renders (same notes, 40-100 s: flat dry / dynamics / sustain / both) in `backend/temp/expression_ab/` on the
  Mac. **Waiting on the user's ear.** Hosted on the Mac with both flags (localhost:5173).
- The cluster's `transcription.py` differs from the repo's, so the hook is NOT on the cluster yet (the bench calls
  `expression.py` directly).

## Clean-up v2: keep 0.3, and the merge plan (2026-10-27)

- **By ear the user prefers keep 0.3** over 0.4 and 0.5 (v2, base at 0.8, quantize OFF). The Mac backend now runs
  `...THR=0.8 ARRANGEMENT_CLEANUP=app/ml/spectral/checkpoints/cleanup2/best.pt ARRANGEMENT_CLEANUP_KEEP=0.3
  ARRANGEMENT_DYNAMICS=1 ARRANGEMENT_SUSTAIN=1`. The user's last song (link upload, session 6b814090 at keep 0.4:
  4,754 notes, 13.1/s) rerun at 0.3 as session 07842b8d: 5,837 notes, 16.1/s.
- **Quantizing stays OFF** (user, 2026-10-27): v2's position head snaps confident notes 100% onto madmom's grid
  (unreliable: tempo right 57%), flattens rolled chords and pulls the song's push / lay-back / tempo drift onto
  POP909's written grid; on renders the base's starts are 3.7 ms from the written time, quantized 8.0 ms. A
  groove-preserving "tighten, don't snap" (song's own tempo curve and groove template, outliers only, partial
  moves, chords as groups) was designed; a render test bench (known swing / rolls / tempo curves + added sloppiness)
  was started then paused for the "it is not the song I remember" work.
- **Clean-up v2 is keep / drop only**: per note, from its pitch, start, length, loudness, the base's onset / frame
  probabilities +-3 frames on its key, neighbours, octaves and twelfths, what the same key just did, and madmom beat
  position (a transformer over each 20 s clip). Labels: a base note is "keep" if a written note starts on its key
  within 48 ms (`eval/cleanup_data.py`).

### PLAN (user's idea, not started): let the clean-up MERGE double presses

Keep / drop cannot repair a double press: one written note the base wrote as two same-key strikes A, B. Dropping B
leaves A cut short (the base ended it at B); dropping A starts the note late. The decoder's own merge
(`decode.py` `merge_fragment_s`) only joins a piece < 60 ms with a gap <= 30 ms, so real double presses stay.
- **New action per note:** "merge with the previous strike of the same key", only when that strike is on the same
  key and this one starts close behind it (within ~150 ms of its end). Merged note: start = one of the base's two
  starts (never a new time), end = the later end (never past the base's sound), velocity = the louder. Only the short
  gap between the strikes is filled. **It never adds a note or extends one beyond the base's own notes** (the
  user's constraint). Example: written 2 notes -> base 4 (double presses) -> cleaner 2 again.
- **Labels from data already on the cluster** (`/scratch/basu.ang/cleanup_data`, `cleanup_data_tr`: the base's 0.8
  notes AND the written notes with lengths): a same-key pair of base notes inside one written note = merge (the
  start label = whichever base start is nearer the written one).
- **Model:** v2 + a merge output, fine-tuned from `checkpoints/cleanup2/best.pt` (cheap; v2 already reads the
  same-key context).
- **Measure FIRST (CPU, before any training)** on the held-out clips (`checkpoints/cleanup1/split.json`): how often
  the base splits one written note into >= 2 strikes; what v2 at keep 0.3 does with those today; the ceiling of a
  perfect merge on note F1 and on F1 WITH note ends (COnPOff, which keep / drop barely moves). Train only if the
  ceiling is worth it; then the numbers, then the user's ear at keep 0.3.

### Merge ceiling, measured (2026-10-27, `eval/merge_oracle.py`, `/scratch/basu.ang/merge_oracle`, 40 `courses` tasks)

User's refinement: merges are whatever brings the output closest to the clean written arrangement (no hand rule).
Per clip and key, an exact search splits the base's strikes into runs; each run is dropped or played as one note (first
base start, last base end); score +1 start match (50 ms), +1 end match too (mir_eval rule), -1 unmatched. 3,212
held-out clips (22 songs), 524k written notes, mir_eval note F1:

| | notes | starts R / P / F1 | starts + ends R / P / F1 |
|---|---|---|---|
| base 0.8 | 614k | 0.784 / 0.669 / 0.722 | 0.507 / 0.433 / 0.467 |
| v2 keep 0.3 (hosted) | 477k | 0.734 / 0.807 / 0.768 | 0.495 / 0.544 / 0.519 |
| best keep / drop | 411k | 0.784 / 1.000 / 0.879 | 0.506 / 0.645 / 0.567 |
| **best keep / drop / merge** | 410k | 0.782 / 1.000 / 0.877 | **0.552 / 0.706 / 0.619** |

Merging adds nothing on starts but **+5.2 pts with note ends** over perfect keep / drop (+10 over v2 today). The
base strikes 8.1% of written notes 2+ times (3+: 1.1%); 3.1% of written notes are same-key repeats < 150 ms apart
(the search keeps those apart). Best plans: 53k merges of 127k strikes (2.4 per merge): many merges use a later
ghost strike to carry a note the base cut short. Even the ceiling misses ends on ~30% of right-start notes: the
base's note ends are the next limit.

## Next steps

0. Clean-up merge (section above): ceiling measured (+5 pts with ends); next: v2 + merge output trained on the
   searched plans (fine-tune from `checkpoints/cleanup2/best.pt`), on `gpu-short` (H200 / A100) in 2 h pieces.
0. Voice: fill-and-remove on the 106 real songs is measured, voice filter included (section above). Then
   pick detector + removal rule by ear in the app, and put the chosen chain into the app.

1. **Voice** (`VOICE_PLAN.md`): build the vocal-line score on the 106 real songs (base, C, D), including the
   probability-shape diagnostic that decides whether a generative (flow-matching) melody decoder is worth it.
2. Then the **voice adapter** on the frozen base, rendered with the 1,594 cloned VocalSet singers (the base never
   heard them). Open decision for the user: allow natural vibrato / scoops / early-late entries on singer stems
   (targets still the written notes).
3. Learned note timing (`NEW_BASE_PLAN.md` section 4b): the small learned decoder (exact start times, merged double
   strikes), tested on the base and C, then a timing output inside the model.
4. Then the other instrument-adapter renders and adapters (`NEW_BASE_PLAN.md` section 3), then the combining layer.
5. Optional: the base over-plays on real songs (10.9 notes/s vs 8.0); try a higher onset bar (0.93-0.95) by ear.
