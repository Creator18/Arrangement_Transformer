# Roadmap

Active queue of improvements with concrete reasoning.  Tickets are ordered
by ROI (impact / effort) given the current state of the pipeline.  Lessons
learned from past iterations are captured at the bottom so the *reasons*
behind the current shape don't get lost across sessions.

---

## Current pipeline (as shipping)

```
audio
  ↓ Demucs (htdemucs) — drumless harmonic mix, GPU
  ↓ madmom RNN + DBN  — beat + downbeat + time-signature
  ↓ bandpass split at 240 Hz
bass.wav            upper.wav
  ↓ pYIN              ↓ Basic Pitch (NOT Onsets & Frames — see lessons)
bass notes          upper notes (no layer label)
        └─── merge → 88-key rule-based transcription ────┘
                              ↓
            cycle-mining note-level pattern classification
                       (pattern_extraction.py)
                              ↓
                    trained ML Transformer (88 → 61)
                              ↓
      compress + walking bass + voice leading + tempo enforcement
                              ↓
                       61-key MIDI arrangement
```

Verified-good as of commit `460072f`.  User confirmation: "best arrangements
so far" with the 88-key on tempo and the 5-octave coherent.

---

## Active improvement queue

### 1. Audio-level pattern classifier + Onsets & Frames retry

**Status:** Highest-leverage open item.  Not yet attempted in this shape.

**The problem to solve.**  Basic Pitch caps out at ~0.87 onset F1.  O&F
reaches ~0.95.  We tried O&F in phase 4b and it produced noisier
arrangements — but the diagnosis we eventually landed on is that O&F itself
isn't the issue.  The downstream `pattern_extraction.py` cycle-miner
classifier degrades with note density (more notes diluting the repetition
signal), so a higher-recall transcriber overwhelms it.  Basic Pitch happens
to match cycle-mining's bandwidth.

**The right fix.**  Restore audio-level pattern/melody labels so
`pattern_extraction.py` doesn't have to infer the layer from note repetition
alone.  Then switch the upper transcriber back to O&F and benefit from the
higher recall.

**Implementation options** (ranked by likely fit):

1. **Tiny learned classifier on (note, audio context)** — for each
   transcribed upper-register note, look up a few-hundred-ms audio window
   around its onset, run a small CNN/Transformer that outputs
   P(pattern|melody).  Train on MAESTRO with synthetic labels derived from
   the cycle-miner's confident decisions, or hand-labeled segments.
   Training cost: low (small model, light data).  Inference cost:
   negligible (~ms per note).  Most flexible long-term.

2. **REPET-SIM** — REPET extended to use spectral similarity rather than
   strict periodicity.  Handles multi-period and structurally-varying
   patterns better than vanilla REPET, which we tried and found brittle in
   phase 5c.

3. **Downbeat-aligned median template** — REPET-style but with the median
   computed per downbeat-aligned bar from madmom, recomputed per section
   from segmentation's section boundaries.  More musically structured than
   plain REPET.

4. **NMF on Demucs's "other" stem** — uses the algorithm we explicitly
   rejected, but restricted to drum-less / vocal-less input where it might
   behave better.  Lowest priority because of past failure mode evidence.

### 2. Hand-aware playability enforcement

**Status:** Identified gap.  Currently no per-hand fingerspan or note-count
enforcement; only a total simultaneous-key budget of 10 across both hands.

**The problem.**  The trained model can output a frame with 7 notes in the
bass clef spanning 2 octaves while only 3 active in the treble — totally
unplayable on actual piano even though the total polyphony budget is
satisfied.

**Implementation plan.**  A post-processing pass after the ML output (or
the rule-based output — applies to both paths) that:

- Assigns each note to LH or RH based on pitch, with the C4–E4 ambiguous
  zone resolved by which hand is less loaded
- Caps per-hand simultaneous notes at 5 (one per finger) — drops
  non-chord-tones first when over the limit
- Caps per-hand fingerspan at ~12 semitones — drops the outermost note
  (highest if RH, lowest if LH) when exceeded
- Stamps each note with its `hand` field so the frontend can colour them

Strict mode (drop notes) prioritises playability; permissive mode (warn but
keep) prioritises original-arrangement fidelity.  Default should be strict
for arrangement output.

### 3. Diversify training data

**Status:** MAESTRO-only training works well on classical-piano-shaped
inputs and produces "different" output on pop/film-score inputs.

**Options:**

- **POP909** (~900 popular-music piano arrangements, ~50 MB MIDI)
- **GiantMIDI-Piano** (~10k classical transcriptions, ~500 MB MIDI)
- **Stylistic conditioning** — let the user select "classical reduction"
  vs "pop reduction" via a one-hot in the cond tensor

Likely a 1–2 hour data-pipeline change plus a retraining run.

### 4. Multi-instrument transcription via MT3

**Status:** Currently we collapse everything to piano-band audio before
transcription.  Real instruments (guitar, vocals, electronic timbres) get
flattened into "what does a piano-trained transcriber make of this?"

MT3 (Magenta, 2021) is the natural answer.  Heavyweight (JAX), would add a
third ML framework alongside PyTorch + TF, but unlocks per-instrument
output that we could route differently.  Probably a phase-7+ project.

---

## Lessons learned

Captured here so the reasoning behind past decisions doesn't get lost when
revisiting the architecture months from now.

### Why Basic Pitch (and not Onsets & Frames) for now

We tried O&F in phase 4b.  The arrangements got noisier.  Initial diagnosis:
"O&F is too sensitive for our inputs."  Later (correct) diagnosis: O&F is
fine; the failure point is that `pattern_extraction.py` falls back to
cycle-mining when notes are unlabeled, and cycle-mining's repetition signal
degrades with note density.  Basic Pitch's lower recall happens to keep
note density inside cycle-mining's working range.  The principled fix is
item #1 in the active queue — get a real per-note classifier so density
stops being a problem.

### Why NMF was removed

NMF tries to decompose a spectrogram into K spectral bases × K temporal
activations.  For piano-only input the decomposition is roughly musically
meaningful (one component per voice / pattern).  For mixed audio it isn't;
the components mix drums and harmonic content arbitrarily.  We replaced it
with Demucs (instrument-based separation) to handle real songs.  The cost
was losing the per-component pattern/melody labels NMF gave us "for free"
on piano-only input.  REPET (phase 5c) was an attempt to recover that
distinction without NMF; the implementation was too brittle and got
reverted.  See item #1 for the right path forward.

### Why the hybrid ML + rule-based shape

End-to-end ML systems are common but conflate two different problems:
transcription (a well-defined inverse problem) and arrangement
transformation (a combinatorial mapping with no closed form).  The
former benefits from carefully-tuned classical methods; the latter
benefits from learning.  Splitting at the 88-key boundary gives each
problem the tool that fits it.  The user's testing confirmed this:
ML-on-everything was noisy, rules-on-everything missed dynamics, hybrid
was the best.

### Why madmom

`librosa.beat.beat_track` is a fine classical algorithm.  madmom is a
trained RNN + DBN tracker that's about 2× more accurate on real music
(per ISMIR benchmarks).  It also returns downbeats and an inferred time
signature, neither of which librosa provides.  The data is consumed by
four downstream stages so the accuracy gain compounds.  Install was
tricky (three compatibility shims for Python 3.11 + numpy 1.26) but the
quality difference was visible in every test since.

### Why the spectrogram-crop training trick

The arrangement-reduction problem has no labeled training pairs at scale.
Self-supervision: a good 5-octave arrangement of a piece is one whose
sound, *constrained to the 5-octave keyboard's frequency band*, matches
the original's sound in that same band.  Frequency-cropped mel
spectrogram → JS divergence loss → backprop through differentiable
piano-roll-to-mel synthesis.  This is the single most paper-worthy idea
in the project and the one that lets us train on unlabeled MAESTRO MIDIs
without ever needing a hand-annotated "good reduction" dataset.
