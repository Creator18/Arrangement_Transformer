# New base + instrument adapters: training plan

Decided 2026-10-24. **Renders only, no real songs in training** (real songs are used only to evaluate).

## 1. Training data (renders)

| | |
|---|---|
| Arrangements | POP909 final arrangements (909 songs) + Music2MIDI arrangements of the **training** songs (by-song split; validation and test songs never rendered) |
| Target rule | **Keep the arrangement.** Audio = the arrangement + extra parts not in the target (octave doublings, held chords and pads, counter-lines, arpeggios) + drums. Target = the arrangement's notes at their **written** times. |
| Instruments | 14 families: acoustic piano and keys, electric pianos, organs, acoustic guitars, electric guitars, bass, bowed strings, brass, woodwinds and reeds, synth leads, synth pads, voice (choirs + singer renders), tuned percussion, world plucked and bowed. All suitable presets from **3 soundfonts** (OmegaGMGS2, FluidR3 GM2, MuseScore_General), all banks. Detuned and chorus presets kept as instrument sounds. Sound effects, unpitched percussion and noise presets left out. |
| Lead lines | In many renders the melody is played by a lead instrument (synth lead, electric guitar, violin, flute, sax, trumpet…), so fast instrumental melodies are common. |
| Tone variation | Synthesizer-style, **nothing that shifts pitch**: amplitude envelope (attack, decay, sustain, release), filter cutoff and resonance (static and enveloped), EQ (low/high shelves, mid bands), drive and distortion, compression, reverb, delay, tremolo (loudness LFO). |
| Mastering | Bus compression, limiting and loudness raised to real-master levels (with heavy bass in some), so render loudness covers the range real songs sit in. Measured: real songs sit ~0.8 spreads above our current renders and 1.7x wider, up to 3x in the bass. |
| Tempo | ~25% of renders sped up 1.2-2x (targets scaled exactly). |
| Drums | Many kits (standard, 808, 909, jazz, brush, house, techno, hip-hop…), never in the target. |
| Size | ~250k clips of 20 s (arr_scratch had 116k). Rendered on the courses CPU partition. |

## 2. Base model

| | |
|---|---|
| Tokenizer | HCQT, 7 channels (harmonics x0.5-x5 + onset stream), 3 bins per semitone, 16 ms frames; **half-octave x 32 ms tokens** (changed 2026-10-24). |
| Front-end calibration | Per-band statistics measured on the **new renders** (which include mastering), not on 64 clean clips. |
| Size | Larger than 7.5M (the transcriber looked capacity-bound): width 384, depth 8 (~20M parameters). |
| Time attention | Learned soft windows alternating with global layers (test C: ~30% fewer double presses, best precision). |
| Onset target | Soft: full credit at the written frame, partial credit one frame either side. |
| Training | From scratch, ~120k steps, early stop on validation; prefer A100/H200, V100 minimum. |
| Decoding | Note starts at the centroid of the onset bump (already in the app). |

## 3. Instrument adapters

One LoRA adapter per instrument family (14), on the frozen base.

| Clips | Audio | Target |
|---|---|---|
| (a) single-family | the whole arrangement + extra parts, all on that family's instruments | **everything played**, extra parts included |
| (b) mixed | the arrangement spread over several families | only the notes **this family** plays, extra parts included |

Same tone variation and mastering as the base.

## 4. Combining the adapters

An attention layer over the adapters' outputs (AdapterFusion-style), trained on mixed renders with the adapters frozen. **No bundle adjustment.**

## 4b. Learned note timing (after the base is ready)

The app places each note start at the centroid of its onset bump (2.2 ms median error vs 4.5 ms for the old
first-frame rule), but merged double strikes, flat-topped bumps and uneven bumps still misplace some notes. Plan:

1. **Learned decoder first** (~4-6 h end to end, small GPU cost): a tiny network that reads each detected note's
   onset / frame probability curves (+-4 frames, neighbouring keys) and a small audio patch, and outputs the exact
   start time, whether it is one note or two merged strikes, and optionally a corrected end. Trained on renders
   (true times known), works with any transcriber (the base, C). Judged with eval/timing_check.py and by ear.
2. **Then inside the model:** a sub-frame timing output trained against the written times, added at the
   instrument-adapter stage; keep the decoder only if it still adds something.

## 5. Checks at every stage

- Held-out renders: recall / precision / F1, quick notes by gap, double presses, timing (centroid), per family.
- A fast-instrumental-melody timing test (dense lead figures with known start times).
- The 106 Music2MIDI test songs (evaluation only).
- Listening: "Answer" and the current reference song, against C, arr_scratch and the attack adapter.

## Decided (2026-10-24)

- Base: ~20M parameters (width 384, depth 8).
- Sped-up renders: 25%.
- Tokens: **half an octave** tall (29 pitch rows). Half-octave test (D) led its octave twin (A) at every validation
  check by 4-6 points; base switched before D's real-song tests (they run alongside; stop the base early if they flop).
- Time attention: learned soft windows (C), so the base combines both improvements (untested together).
