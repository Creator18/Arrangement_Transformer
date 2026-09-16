"""Transcribe each Demucs stem separately and recombine, against the full mix.

The transcriber currently reads the whole mix at once.  Per-stem asks whether
handing it one instrument at a time helps, and there are real arguments both
ways:

    FOR   each stem is a simpler signal, so notes masked in a dense mix may
          become visible.  And the stem NAME is a free role label -- vocals are
          the melody, bass is the bass -- which is information the full-mix path
          simply does not have.  `role` is currently "all" for every note, so
          enforce_melody_on_top has never fired on the song path at all.

    AGAINST the model was trained on MIXES.  An isolated stem carrying Demucs
          artefacts -- bleed, phase smear, spectral holes -- is out of
          distribution, and out-of-distribution input is where a supervised
          model fails silently.  Errors also ADD: five stems each contributing
          false positives gives five times the false positives.

The NMF answered this question the second way.  Per-stem produced 5,350
accompaniment notes with 1,308 duplicates; transcribing the summed
accompaniment once gave 1,406 notes with 19 duplicates.  But that was a method
which had to explain every scrap of energy, so it is not evidence about this
one, and the ear is cheap.

Renders four things through IDENTICAL downstream rules:

    mix          full mix, one pass                      (what ships)
    stems        per-stem, recombined, roles discarded    (isolates the split)
    stems+roles  per-stem, recombined, roles KEPT         (isolates the labels)
    original     the recording

The middle two differ only in whether the role survives, so if stems+roles
sounds better than stems, that is the melody-on-top rule finally working
rather than anything about separation.

    venv/Scripts/python.exe eval/ab_per_stem.py --session <uuid>
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys
import warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "backend/eval"))
warnings.filterwarnings("ignore")
import scipy.signal, scipy.signal.windows                       # noqa: E402
if not hasattr(scipy.signal, "hann"):
    scipy.signal.hann = scipy.signal.windows.hann

from eartest import PAGE, render, OUT, NAMES                    # noqa: E402
from ab_note_source import arrange, key_of                      # noqa: E402

# Demucs htdemucs_6s stem -> the arranging role it implies.
#
# Drums are excluded rather than transcribed and dropped: the model was trained
# with drums present in the mix and absent from the target, so it already knows
# to ignore them, and handing it a drums-only stem asks a question with no
# right answer.
STEM_ROLE = {
    "vocals": "melody",     # the sung line IS the tune
    "bass": "bass",
    "piano": "harmony",
    "guitar": "harmony",
    "other": "harmony",
}

# Two notes of the same pitch this close together are one note seen twice.
#
# Needed because stems bleed: a piano chord audible in `other` and in `piano`
# gets transcribed from both.  50 ms matches the mir_eval onset tolerance the
# rest of the project scores against, so "duplicate" means the same thing here
# as it does in every measurement.
DEDUP_S = 0.05


def merge(groups):
    """Concatenate per-stem notes, dropping same-pitch near-duplicates.

    Keeps the LOUDEST of a duplicate set, on the reasoning that the stem which
    heard a note most strongly is the stem it belongs to.
    """
    notes = [n for g in groups for n in g]
    notes.sort(key=lambda n: (int(n["pitch"]), float(n["onset"])))
    out, dropped = [], 0
    for n in notes:
        if out and int(out[-1]["pitch"]) == int(n["pitch"]) and \
                abs(float(n["onset"]) - float(out[-1]["onset"])) <= DEDUP_S:
            if float(n.get("velocity", 80)) > float(out[-1].get("velocity", 80)):
                out[-1] = n
            dropped += 1
            continue
        out.append(n)
    out.sort(key=lambda n: float(n["onset"]))
    return out, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--name", default="per_stem")
    ap.add_argument("--thr", type=float, default=0.94)
    args = ap.parse_args()

    from app.services import ml_transcription as ML

    sess = ROOT / "backend/temp" / args.session
    comp = sess / "components"
    meta = json.loads((sess / "components.json").read_text())
    hop, msr = float(meta.get("hop_length", 512)), float(meta.get("sr", 22050))
    beats = np.asarray(meta.get("beat_frames", []), float) * hop / msr
    OUT.mkdir(exist_ok=True)

    todo = []

    # A — the full mix, one pass.  What ships.
    mixn = ML.transcribe(sess / "audio.wav", onset_thr=args.thr)
    todo.append(("mix", f"full mix, one pass (thr {args.thr})", mixn, key_of(mixn)))

    # B/C — per stem, recombined.
    groups, counts = [], []
    for stem, role in STEM_ROLE.items():
        p = comp / f"stem_{stem}.wav"
        if not p.exists():
            print(f"  no stem_{stem}.wav, skipping")
            continue
        ns = ML.transcribe(p, onset_thr=args.thr)
        for n in ns:
            n["role"] = role
        counts.append((stem, role, len(ns)))
        groups.append(ns)

    if groups:
        merged, dropped = merge(groups)
        print()
        for stem, role, n in counts:
            print(f"  stem_{stem:<8} role {role:<8} {n:>5} notes")
        print(f"  {'merged':<19} {len(merged):>5} notes "
              f"({dropped} same-pitch duplicates dropped)")
        print(f"  {'full mix':<19} {len(mixn):>5} notes\n")

        # Roles discarded, so the ONLY difference from C is the labels.
        flat = [dict(n, role="all") for n in merged]
        todo.append(("stems", f"per-stem merged, roles discarded", flat,
                     key_of(flat)))
        todo.append(("stems_roles", f"per-stem merged, roles KEPT", merged,
                     key_of(merged)))

    sources, dur = {}, 0.0
    for key_id, label, notes, key in todo:
        arranged, steps = arrange(notes, beats, key)
        print(f"{label:<38} " + " -> ".join(f"{n}={v}" for n, v in steps)
              + f"  | key {NAMES[key[0]]} {key[1]}")
        wav = OUT / f"{args.name}_{key_id}.wav"
        dur = max(dur, render(arranged, wav))
        sources[key_id] = {
            "label": label, "key": f"{NAMES[key[0]]} {key[1]}",
            "piano": wav.name, "audio": None,
            "notes": [[int(n["pitch"]), round(float(n["onset"]), 3),
                       round(float(n["offset"]), 3), int(n.get("velocity", 80))]
                      for n in arranged]}

    orig = OUT / f"{args.name}_original.mp3"
    if not orig.exists():
        shutil.copy(sess / "original.mp3", orig)
    sources["mix_audio"] = {"label": "original mix", "key": "-", "piano": None,
                            "audio": orig.name, "notes": []}

    page = OUT / f"{args.name}.html"
    page.write_text(PAGE % {"title": f"{args.name}: per-stem vs full mix",
                            "data": json.dumps(sources), "dur": dur, "pps": 60},
                    encoding="utf-8")
    print(f"\nwrote {page}")


if __name__ == "__main__":
    main()
