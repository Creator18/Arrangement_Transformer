"""
Training data plumbing.

    maestro.py     — MAESTRO v3.0.0 MIDI loader, random-segment cropper
    conditioning.py— roll → (T, 24) chroma + bass-biased root conditioning
    noise.py       — scale/chord-conditioned Gaussian noise generator
    dataset.py     — MaestroDataset + SyntheticDataset (smoke-test) yielding
                     `Batch` tuples to training.loop.train(...)
"""
from .maestro     import MaestroEntry, list_maestro, load_piano_roll, sample_segment  # noqa: F401
from .conditioning import build_conditioning_from_roll                                 # noqa: F401
from .noise        import conditioned_noise, zero_noise                                # noqa: F401
from .dataset      import MaestroDataset, SyntheticDataset                             # noqa: F401
