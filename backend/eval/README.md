# Evaluation Harness

Measures the transcription half of the pipeline — audio → 88-key MIDI — against ground-truth MIDI from a held-out slice of MAESTRO. This is what we benchmark against when changing the model, the separation strategy, the cleanup rules, or anything else upstream of `/transform`.

The transformed (88 → 61) half of the pipeline is measured separately by `eval_arrangement.py` — see **Arrangement eval** below.

## What lives where

```
backend/eval/
├── select_test_set.py        # build a 50-piece deterministic test set
├── synthesize_audio.py       # render each ground-truth MIDI to WAV via sfizz
├── run_pipeline.py           # run separate → transcribe → segment on one piece
├── compare.py                # mir_eval F1/P/R vs ground truth, aggregate
├── data/
│   ├── manifest.json         # the 50 picks (composer, title, paths)
│   ├── audio/                # synthesised WAVs (gitignored)
│   └── synth_log.txt         # synthesis output log
├── runs/                     # per-piece run dirs with transcription.json
└── baselines/                # snapshot F1 numbers, named by git SHA + ts
```

## How to run a full evaluation cycle

```bash
# 1. Build the test set (run once; idempotent)
python backend/eval/select_test_set.py

# 2. Synthesise audio for all 50 pieces (~45-60 min, one-time)
python backend/eval/synthesize_audio.py

# 3. Run the pipeline on all 50 (~40 min)
python backend/eval/run_pipeline.py --all

# 4. Compare against ground truth
python backend/eval/compare.py --all

# 5. Save this as a baseline (do this after every meaningful change)
python backend/eval/compare.py --all --save-baseline
```

After making a pipeline change, re-run steps 3-4 and compare against the previous baseline:

```bash
python backend/eval/run_pipeline.py --all --force
python backend/eval/compare.py --all --compare-to backend/eval/baselines/baseline_<sha>_<ts>.json
```

The output lists pieces whose F1 moved by more than 0.01 in either direction. That's the regression-test surface.

## How the test set was built

50 pieces from MAESTRO v3's official `test` split, never used for any training. Selection rules:

- Duration in [30s, 8min] — long enough to be meaningful, short enough that one synthesis + run cycle fits in a session
- Stratified by canonical composer — each composer with at least one eligible piece gets at least one slot; remaining slots distributed proportionally to that composer's share of the eligible pool (largest-remainder method)
- Fixed random seed (42) — re-running `select_test_set.py` always produces the same manifest

The current selection covers 14 composers from Bach through Rachmaninoff. See `data/manifest.json` for the full list and per-composer counts.

## What gets measured

The harness compares the **ORIGINAL** transcription view (after the `/segment`-side cleanup pass) against the ground-truth MIDI using `mir_eval.transcription.precision_recall_f1_overlap`. Two tiers:

| Metric | Onset tolerance | Pitch tolerance | Offset tolerance |
|--------|-----------------|------------------|-------------------|
| **COnP** (Correct Onset + Pitch) | ±50 ms | ±50 cents | — |
| **COnPOff** (also Correct Offset) | ±50 ms | ±50 cents | ±50 ms or 20% of duration |

COnP is the standard piano transcription benchmark; COnPOff is the stricter version that also requires the duration to be roughly right.

Interpreting the two:

- **High COnP, low COnPOff** → onset detection is solid, duration estimation is the weak point. Fix: better duration handling (snap to musical values, smarter offset detection).
- **Low COnP** → the model is missing or hallucinating notes. Fix: model improvement, threshold tuning, or separation issues upstream.
- **Predicted >> truth note count** → duplication across REPET stems, or threshold too loose.
- **Predicted << truth note count** → threshold too strict, or model under-recall.

## Audio source: synthesised vs real

The harness synthesises audio from MAESTRO's MIDI files using our own sfizz + Salamander sampler. This is **deliberately optimistic** — synthesised audio has no mic preamp noise, no room reverb, no performance imperfections beyond what's encoded in the MIDI. F1 numbers on this set will be higher than on real recordings.

Trade-offs we accepted:

- **Pro**: free, instant, deterministic, no 118 GB download
- **Pro**: every run measures the *pipeline*, not the audio source. Removes a confound
- **Con**: numbers don't translate directly to real-audio quality

Mitigation for the future: hand-pick 5–10 real MAESTRO recordings to download individually as a "reality anchor" subset. The bulk of the eval stays on synthesised audio for reproducibility; the anchor catches if the synthesised numbers and real numbers diverge.

## Baseline workflow

Baselines are JSON snapshots of `compare.py --all` output, named by git short-SHA and timestamp. Save one whenever you make a change you want to remember the impact of:

```bash
python backend/eval/compare.py --all --save-baseline
# writes baselines/baseline_<sha>_<timestamp>.json
```

Then any future run can diff against it:

```bash
python backend/eval/compare.py --all --compare-to backend/eval/baselines/baseline_4520d49_20260513_220000.json
```

This is the regression-test surface for the entire pipeline. If a pipeline change passes its own smoke tests but the eval shows ≥3 pieces regressing by F1 ≥ 0.01, the change is probably worse than it looks.

## What's *not* in this harness

- **Subjective arrangement quality.** `eval_arrangement.py` now covers the structural side (see below), but it only catches what is provably wrong — out-of-range notes, doubled keys, unplayable chords. Whether an arrangement is *musical* still needs blind A/B listening against a trivial baseline.
- **Real-world audio evaluation.** Synthesised audio only for now. Add a hand-curated real-audio subset later.
- **Klangio comparison numbers.** Manual workflow: upload the 50 audio files to Klangio's free trial, download their MIDI outputs, run them through `compare.py` against the same ground truth. Not automated.
- **Per-layer breakdown.** Currently we compare full transcriptions. If we want per-layer accuracy (bass_pattern vs bass_melody vs pattern vs melody), we'd need ground-truth labels for those layers, which MAESTRO doesn't provide.

## Tuning the harness

The mir_eval tolerances are tuned to standard piano AMT defaults. To change them, edit constants at the top of `compare.py`:

- `ONSET_TOL_S` (default 50 ms) — how close the onset has to be
- `PITCH_TOL_C` (default 50 cents) — pitch tolerance; 50 cents = ±½ semitone, i.e. semitone-exact
- `OFFSET_TOL_S` (default 50 ms) — minimum offset tolerance
- `OFFSET_RATIO` (default 0.2) — offset tolerance also includes 20% of note duration

These are the values used in the Onsets & Frames paper and most subsequent piano AMT work; comparable numbers across the literature use these defaults.

## Arrangement eval (88 → 61)

`eval_arrangement.py` measures the *arrangement* half. It cannot work like the
transcription harness: there is no ground truth, because a good 5-octave
arrangement legitimately differs from its 7-octave source. Distance from the
source is not a quality signal — a perfect arrangement would score badly on it.

So the metrics are reference-free, and of two kinds:

- **Absolute** — facts about a 61-key piano, not opinions about arranging.
  A note outside the keyboard is wrong. Two note-ons for one key at the same
  instant is wrong. A chord no two hands can take is wrong.
- **Relative** — what changed in the two voices a listener actually tracks
  (top line, bass line). Deliberately not a note-by-note diff, so an arm that
  legitimately rewrites inner figuration is not punished for doing its job.

Lower is better on every column; 0 is ideal.

```bash
python backend/eval/eval_arrangement.py                  # 5 octaves, all arms
python backend/eval/eval_arrangement.py --octaves 6
python backend/eval/eval_arrangement.py --arms fold --limit 30
```

`passthrough` is the control — and it is *literally* what the 7-octave preset
does, since `compress_notes_to_range` returns untouched notes for `(21, 108)`.
That is worth stating plainly: **all of the 7-octave tuning work bypassed range
compression entirely**, so 5-octave was running through a stage no listening
test had ever covered.

Result that motivated `fit_notes_to_range` (120 pieces, 218k notes, 5 octaves):

| arm | out-of-range | doubled keys/piece | worst | crossings | note loss |
|---|---|---|---|---|---|
| passthrough | 3078 | 0.2 | 4 | 0.0 | — |
| fold (old) | 0 | 12.6 | 96 | 1.6 | 0% |
| fold_nodouble | 0 | **0.0** | **0** | 0.2 | 0.006% |

Doubled keys are the metric that matters most here: a second note-on for a key
already held double-triggers the sampler, and in MIDI export the first note-off
silences the key while the second note is still notated as held. That is the
mechanism behind repeated notes losing their sustain.

**Corpus caveat.** These numbers come from ASAP solo classical piano
(`correction/data/work/`). Song and pop material typically carries *more*
sub-C2 content, so they probably understate the problem rather than overstate it.
