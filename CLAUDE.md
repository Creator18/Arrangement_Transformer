# Arrangement Transformer: hand-off notes for Claude Code

Read this first. It says what the project is, what is running on the cluster right now, how to check it, and the
rules the user has set. Last updated **2026-10-26** (branch `feat/song-to-piano`).

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

## Next steps

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
