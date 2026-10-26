# Voice: how to transcribe real singing as piano

Opened 2026-10-25, after the base (`base20m`) was judged by ear to combine the good things of C and D. Voice is the
one open problem. This file holds what we found, the literature, the decisions so far and what is still open.

## 1. Where the base stands on voice

**What the base heard in training** (sample of 3,000 `base_render` train clips):

| | share |
|---|---|
| renders with a sung melody | ~13% (398 / 3,000) |
| ... sung by the soundfont "vowel" voice (`vocal_stem`, syllables like "pu o oo bu") | **all of them** |
| ... sung by a cloned VocalSet singer (`/scratch/basu.ang/voices`, 1,594 RVC clones) | **none** |

`render_base.py` never passed the cloned voices through, so the base has never heard a realistic singer.

**Evidence the cloned singers help** (earlier singer LoRA `arr_sing_lora` on `arr_tone_full`, 106 real songs,
`logs/eval_sing_11143615.out`):

| | strict | P | R | in-range | notes/s (human 8.0) |
|---|---|---|---|---|---|
| tone-full (its base) | 0.246 | 0.211 | 0.310 | 0.63 | 11.8 |
| + singer LoRA | 0.249 | **0.240** | 0.276 | **0.74** | **8.4** |

Precision +3 points and over-playing gone, but strict F1 barely moves: the all-notes score is dominated by the
accompaniment, so it cannot see the vocal line. Hence a vocal-line score (section 4).

## 2. What the literature says

No paper treats "sung line -> piano notes" directly; it is covered piecewise by singing transcription, piano-cover
generation and perception studies.

1. **Target = the intended (written) tune, not the performed sound.** Every piano-cover system (Pop2Piano,
   PiCoGen / PiCoGen2, AMT-APC) trains toward a written arrangement; PVG sheet music transcribes the melody, then
   arranges the accompaniment. Tagg: pop singing is "neither conceived nor designed" for notation; the lead sheet is
   the compromise. Confirms our "written times" rule.
2. **Listeners hear the intended note through vibrato and scoops.** Perceived pitch of a long vibrato tone ~ the
   mean of its extremes; scoops are integrated partly by averaging over the tone (Larrouy-Maestri & Pfordresher
   2018). Singing datasets (N20EMv2) do not count vibrato or an onset bend as separate notes, and put the onset at
   the **vowel**, not the consonant. Known failure of singing transcribers: vibrato read as trills / repeated notes
   (our "double presses").
3. **Real singers are off the grid.** 88k pop vocal tracks (The Vocal Eras Tour, TISMIR): singers anticipate the
   beat, most on beats 1 and 3; rap lags. Arrangers still write on the grid, so strict 50 ms matching undercounts
   sung notes; report 150 ms too.
4. **Two architectures.** End-to-end audio -> piano (Pop2Piano, AMT-APC; ours) vs melody-first via a lead sheet
   (PiCoGen; Sheet Sage / "Melody transcription via generative pre-training", Donahue et al. ISMIR 2022, onsets on a
   beat-quantised 16th grid). Singing-only references: ROSVOT, VOCANO, MIR-ST500; T3MS (2025) predicts note values
   directly from singing.
5. **Scoring.** Molina et al. (ISMIR 2014): score singing transcription by error type (split, merged, onset,
   offset), not one F1. No study quantifies how human pianists treat the melody in pop covers; our own POP909 check:
   the human melody is the top voice 78% of the time (skyline F1 0.781).

Sources: Pop2Piano arXiv 2211.00895; PiCoGen 2407.20883; PiCoGen2 2408.01551; AMT-APC 2409.14086; style
bootstrapping 2608.03050; Sheet Sage 2212.01884; ROSVOT 2405.09940; VOCANO (ISMIR 2021); T3MS 2502.12438; Molina et
al. ISMIR 2014; N20EMv2 guidelines 2304.12082; vocadito 2110.05580; Larrouy-Maestri & Pfordresher, JEP:HPP 2018;
The Vocal Eras Tour, TISMIR 10.5334/tismir.278; Tagg, Analysing Popular Music (1982).

## 3. Flow matching? (discussed 2026-10-25)

Not as the next step. It is a generative method: it helps where the right answer is genuinely ambiguous (octave of
the melody, how to put an off-beat sung rhythm onto the grid, scoop / melisma as one note or several) and an
averaging model smears its probabilities. But:

- the main gap is **data** (the base never heard real vibrato / scoops / timing); a generative decoder trained on the
  same renders inherits it;
- piano rolls are binary (needs discrete flow matching or a note embedding); diffusion transcription
  (e.g. DiffRoll) gained little over plain prediction;
- it costs inference passes (~50 s per song already on a laptop CPU) and removes the onset-cutoff knob the user
  tunes by ear (0.9), and makes metric comparisons noisier.

Where it could fit later: a small melody-only decoder on the frozen base (probabilities + audio -> clean written
melody notes). Plan 4b's learned timing decoder attacks the same split / merged strikes more cheaply. **Decide with
data:** the vocal-line score reports the probability shape on missed / split melody notes. Confidently wrong or
silent -> renders + voice adapter. Spread / doubled peaks -> a generative melody decoder may earn its place.

## 4. Next: the vocal-line score (agreed 2026-10-25, not built yet)

On the 106 Music2MIDI test songs, for base, C and D at the 0.9 bar (all three real-song runs used thr 0.90):

- **Reference melody:** skyline of the human arrangement (top note of each onset event, >= C4; `rhythm_groups.top_voice`).
- **Scores:** melody recall / precision at 50 ms and 150 ms; octave-tolerant match; skyline-vs-skyline F1 (is the
  melody on top); **split-note count** (one melody note answered by >= 2 starts); start-time error on matched notes.
- **Sung vs instrumental sections:** vocal activity from Demucs (htdemucs) vocals energy, so the voice is scored
  apart from lead instruments.
- **Diagnostic for section 3:** onset-probability shape on missed and split melody notes (silent / confident-wrong /
  spread).
- CPU for scoring (courses partition), one GPU job for the rolls and Demucs.

## 5. Then: the voice adapter (plan section 3, voice family)

LoRA on the frozen base, rendered with the **cloned VocalSet singers** (not soundfont vowels), same tone variation and
mastering. **Open decision for the user:** let the singer stems carry natural vibrato, scoops and slightly early /
late entries, with targets still at the written notes. That relaxes the current rule (+-3-5 cents drift, no vibrato)
for singer stems only; the literature says that is exactly what the model must learn to see through.

Fallback if the adapter cannot close the gap: a separate vocal path (Demucs vocals -> dedicated melody transcriber,
starting from the POP909-prior decoder, commit 480967c) merged with the base's output.
