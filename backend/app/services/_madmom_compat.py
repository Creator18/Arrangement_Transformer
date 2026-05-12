"""
Compatibility shim for madmom 0.16.1 on Python ≥ 3.10 and numpy ≥ 1.20.

madmom hasn't been updated since 2018 and breaks on modern Python / numpy
in three independent ways.  Importing this module BEFORE any
`from madmom...` statement papers over all three:

  1. `collections.MutableSequence` (etc.) — moved to `collections.abc` in
     Python 3.3, removed from `collections` itself in 3.10.
  2. `numpy.float`, `numpy.int`, `numpy.bool`, etc. — deprecated in numpy
     1.20, removed in 1.24.
  3. `DBNDownBeatTrackingProcessor.process` builds an `np.asarray` over a
     list of inhomogeneous `(path, log_prob)` tuples.  Newer numpy rejects
     this construction; we monkey-patch the method to extract the log-prob
     column with a list comprehension first.

Remove this shim once madmom releases a compatible build.
"""
from __future__ import annotations

import collections
import collections.abc
import numpy as np


# ── collections aliases ──────────────────────────────────────────────────────
for _name in (
    "MutableSequence",
    "MutableMapping",
    "Iterable",
    "Mapping",
    "Sequence",
    "Callable",
    "Hashable",
):
    if not hasattr(collections, _name):
        setattr(collections, _name, getattr(collections.abc, _name))


# ── numpy scalar aliases ─────────────────────────────────────────────────────
# These map to the Python builtins, which is the numpy-recommended drop-in
# (the original aliases were just convenience handles for the builtins).
# `hasattr` triggers numpy's deprecation FutureWarnings for the not-yet-removed
# names, so we silence those here — they're firing on attribute *probes*, not
# on actual use of the deprecated alias.
import warnings as _warnings
with _warnings.catch_warnings():
    _warnings.simplefilter("ignore", FutureWarning)
    for _name, _alias in (
        ("float",   float),
        ("int",     int),
        ("bool",    bool),
        ("object",  object),
        ("complex", complex),
        ("long",    int),
        ("str",     str),
        ("unicode", str),
    ):
        if not hasattr(np, _name):
            setattr(np, _name, _alias)


# ── DBN tracker patch ────────────────────────────────────────────────────────
# madmom's DBNDownBeatTrackingProcessor.process calls
#     best = np.argmax(np.asarray(results)[:, 1])
# where `results` is a list of (path_array, log_prob_scalar) tuples.  Newer
# numpy refuses to build that as a 2-D array because path_arrays have
# different lengths.  We replace the method with a version identical to the
# upstream apart from extracting column 1 with a list comprehension before
# the argmax.  Same logic, no shape construction.
def _install_dbn_patch() -> None:
    import itertools as it
    from madmom.features.downbeats import (
        DBNDownBeatTrackingProcessor, _process_dbn,
    )

    def _patched_process(self, activations, **kwargs):
        first = 0
        if self.threshold:
            idx = np.nonzero(activations >= self.threshold)[0]
            if idx.any():
                first = max(first, np.min(idx))
                last = min(len(activations), np.max(idx) + 1)
            else:
                last = first
            activations = activations[first:last]

        if not activations.any():
            return np.empty((0, 2))

        # Parallel HMM decoding of the activations
        results = list(self.map(
            _process_dbn, zip(self.hmms, it.repeat(activations)),
        ))

        # FIX: avoid the inhomogeneous-shape np.asarray on `results`.
        best = int(np.argmax([r[1] for r in results]))

        path, _ = results[best]
        st = self.hmms[best].transition_model.state_space
        om = self.hmms[best].observation_model
        positions = st.state_positions[path]
        beat_numbers = positions.astype(int) + 1

        if self.correct:
            beats = np.empty(0, dtype=int)
            beat_range = om.pointers[path] >= 1
            idx = np.nonzero(np.diff(beat_range.astype(int)))[0] + 1
            if beat_range[0]:
                idx = np.r_[0, idx]
            if beat_range[-1]:
                idx = np.r_[idx, beat_range.size]
            if idx.any():
                for left, right in idx.reshape((-1, 2)):
                    peak = np.argmax(activations[left:right]) // 2 + left
                    beats = np.hstack((beats, peak))
        else:
            beats = np.nonzero(np.diff(beat_numbers))[0] + 1

        return np.vstack((
            (beats + first) / float(self.fps),
            beat_numbers[beats],
        )).T

    DBNDownBeatTrackingProcessor.process = _patched_process


# Patch lazily — only when madmom is actually about to be used, so the
# import cost is paid once and not at server startup.
_patched = False


def apply_madmom_runtime_patches() -> None:
    global _patched
    if _patched:
        return
    _install_dbn_patch()
    _patched = True
