"""
Arrangement Transformer — ML transformation core.

This package contains the learned 88-key → 61-key arrangement transformation:

    audio → harmonic extraction → scale/chord-conditioned noise
        → Transformer encoder (88 + cond → 61) → composite loss
        → post-processing rules → MIDI

NMF is not used anywhere here.  Source separation, when needed at inference
time, runs through Demucs in `services/`.

Layout
------
    config.py           — hyperparameters, paths, loss-weight schedules
    models/             — Transformer generator, differentiable mel-spectrogram
    losses/             — five loss terms + composite assembly
    training/           — training loop, checkpoint I/O, weight schedule
    inference/          — adapter that the FastAPI /transform route calls
    data/               — MAESTRO loader, conditioned-noise generator (phase 2)
"""
