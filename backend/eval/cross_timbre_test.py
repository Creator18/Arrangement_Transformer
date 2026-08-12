"""
When the input is NOT a piano, does it still project onto the right keys?

octave_ghost_test showed an 88-key dictionary resolves its own harmonic
ambiguity: 87.7% of activation energy on pitches that are sounding or ringing,
octave ghosting under 8% even at zero tolerance.  But that test rendered its
signal with the same sampler that built the dictionary — same instrument, same
samples, same room.  Templates matched the material exactly, which is the one
condition guaranteed never to hold on a real song.

This removes that.  Same MAESTRO MIDI, same ground-truth rolls, same scoring
function, same piano dictionary.  The ONLY thing that changes is what the
audio is rendered with.  So whatever moves is attributable to timbre.

Why these instruments
---------------------
The axis that should matter is not "how different does it sound" but HOW IT
DECAYS, because a piano template *is* a decay curve.

    grand piano (GM)   control — different sampler, same instrument, so it
                       separates "different rendering" from "different thing"
    upright piano      same family, different box
    nylon guitar       plucked and decaying; closest envelope to a piano
    string ensemble    bowed, sustains flat, never decays
    church organ       the extreme: perfectly steady until release

For a sustained source there is no correct static answer — a piano template
decays, so the only way NMF can hold a level is to keep re-striking.  If the
sustained instruments fall apart while the plucked one holds, that is not a
failure of the design, it is a boundary on it, and one that maps straight onto
real material: guitars and drums pluck, vocals and strings sustain.

RESULT (5 MAESTRO pieces, 30 s each, 2026-08-11)
------------------------------------------------
    voice                correct  decay  octave  fifth  other   accounted
    sfizz piano (matched)  45.8%  41.9%    2.0%   1.2%   9.0%      87.7%
    grand piano (GM)       39.1%  31.6%    5.0%   2.5%  21.8%      70.7%
    upright piano          40.7%  34.1%    5.2%   3.4%  16.6%      74.7%
    nylon guitar           29.5%  27.0%    9.7%   5.4%  28.4%      56.5%
    string ensemble        37.3%  10.9%   14.6%   7.4%  29.8%      48.2%
    church organ           15.4%  19.2%   21.8%   2.6%  41.1%      34.6%

The ordering is exactly the decay hypothesis, and it separates two costs that
are worth keeping apart:

  * Changing SAMPLER but not instrument costs ~15 points (87.7 -> 70.7/74.7).
    That is the price of templates not matching the recording, and it is what
    any real audio would pay even if it were a piano.
  * Changing INSTRUMENT costs another ~15 for a plucked source (56.5%) and
    lands far lower for sustained ones (48.2%, 34.6%).

Octave ghosting tracks sustain precisely: 2% matched, 5% cross-sampler, 9.7%
plucked, 14.6% bowed, 21.8% organ.  This is the predicted mechanism and it is
now visible — a piano template IS a decay curve, so the only way NMF can hold
a steady level is to keep restriking, and stacking octaves is the cheapest way
to fake a flat envelope.  Percussive and plucked material projects onto the
piano subspace; sustained material fights it.

So the design is viable but bounded, and the boundary maps onto real songs:
guitar, drums and piano project well; vocals, strings and pads do not.  Those
want a different treatment (sustained lines re-articulated as tremolo, held
octaves, or simply given to the melody path) rather than a better dictionary.

NOTE: the first run of this test was invalid.  sf2_sampler ignored the SF2
zone tuning generators (overridingRootKey / coarseTune / fineTune) and
subtracted the sample correction instead of adding it, so Church Organ came
out ~1400 cents flat and Nylon Guitar swung between -1537 and +906 cents.
Guitar scored 14.5% purely because it was out of tune.  The pipeline never saw
this because it only ever loads bank 0 / program 0, whose sample headers
happen to need no correction.
"""
from __future__ import annotations

import argparse, json, pathlib, struct, sys, warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "backend/eval"))
warnings.filterwarnings("ignore")
import scipy.signal, scipy.signal.windows                       # noqa: E402
if not hasattr(scipy.signal, "hann"):
    scipy.signal.hann = scipy.signal.windows.hann

from octave_ghost_test import (                                  # noqa: E402
    SR, FPS, stft_mag, nmf_h, categorise, ground_truth, load_dictionary)

GM = ROOT / "OmegaGMGS2.sf2"
UPRIGHT = (ROOT / "UprightPianoKW-SF2-20220221" /
           "UprightPianoKW-SF2-20220221" / "UprightPianoKW-20220221.sf2")

# (label, sf2 path, program, bank, sounds_st)
#
# `sounds_st` is how far the preset sounds BELOW written pitch.  Church Organ
# is -12: measured by harmonic-comb scoring across the keyboard it renders a
# consistent octave down, which is not a bug but a 16' rank, exactly as a real
# church organ is built.  Scoring it against written-pitch ground truth would
# charge the decomposition for finding the notes that are genuinely sounding.
# Every other voice measured +0.0.
VOICES = [
    ("grand piano  (GM)", GM, 0, 0, 0),
    ("upright piano", UPRIGHT, 0, 0, 0),
    ("nylon guitar", GM, 24, 0, 0),
    ("string ensemble", GM, 48, 0, 0),
    ("church organ", GM, 19, 0, -12),
]


def list_presets(path):
    data = path.read_bytes()
    out = []
    total = len(data)
    lists = {}
    from app.services.sf2_sampler import _iter_chunks
    for tag, ds, size in _iter_chunks(data, 12, total):
        if tag == "LIST":
            lists[data[ds:ds+4].decode("latin-1")] = (ds + 4, ds + size)
    ps, pe = lists["pdta"]
    for tag, ds, size in _iter_chunks(data, ps, pe):
        if tag == "phdr":
            raw = data[ds:ds+size]
            for i in range(len(raw) // 38 - 1):
                o = i * 38
                name = raw[o:o+20].decode("latin-1").rstrip("\x00")
                pr, bk = struct.unpack_from("<HH", raw, o+20)
                out.append((bk, pr, name))
    return sorted(out)


def render_midi(midi_path, seconds, sf2_path, program, bank):
    """Render the first `seconds` of a MIDI through one SF2 preset."""
    import pretty_midi
    import app.services.sf2_sampler as s

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            if n.start < seconds:
                notes.append({"pitch": int(n.pitch), "onset": float(n.start),
                              "offset": float(min(n.end, seconds)),
                              "velocity": int(n.velocity)})
    notes.sort(key=lambda n: n["onset"])

    # Swap the module's loaded preset.  Reaching into the sampler's globals is
    # blunt, but the alternative is a second SF2 parser, and a divergent copy
    # of the parser is a worse problem than a monkeypatch in an eval script.
    s._smpl_pcm, s._key_map = s._parse_sf2(sf2_path.read_bytes(),
                                           program=program, bank=bank)
    s._loaded = True
    s._pitch_cache.clear()
    y = s.render_notes_sf2(notes, sr=SR)
    return np.asarray(y, dtype=np.float32), len(notes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pieces", type=int, default=5)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--list", action="store_true", help="dump SF2 presets and exit")
    args = ap.parse_args()

    if args.list:
        for bk, pr, name in list_presets(GM):
            print(f"  bank {bk:<4} program {pr:<4} {name}")
        return

    man = json.loads((ROOT / "backend/eval/data/manifest.json").read_text())
    W = load_dictionary()
    print(f"dictionary: {W.shape}  (piano, unchanged for every voice below)\n")

    pieces = []
    for p in man["pieces"]:
        midi = ROOT / p["midi_path"]
        roll = ROOT / p["midi_path"].replace(".midi", ".roll.100.npy")
        if midi.exists() and roll.exists():
            pieces.append((p, midi, roll))
        if len(pieces) >= args.pieces:
            break

    print(f"{'voice':<20}{'correct':>9}{'decay':>8}{'octave':>8}{'fifth':>7}"
          f"{'other':>8}   {'accounted':>10}")
    print("-" * 72)

    for label, sf2, program, bank, sounds_st in VOICES:
        if not sf2.exists():
            print(f"{label:<20}  (missing {sf2.name})")
            continue
        rows = []
        for p, midi, roll in pieces:
            try:
                y, n = render_midi(midi, args.seconds, sf2, program, bank)
            except Exception as e:                              # noqa: BLE001
                print(f"  render failed {p['id'][:30]}: {type(e).__name__}: {e}")
                continue
            if y.ndim > 1:
                y = y.mean(axis=1)
            if y.size < SR:
                continue
            V = stft_mag(y)
            H = nmf_h(V, W)
            GT = ground_truth(roll, V.shape[1])
            if sounds_st:
                # Compare against what the instrument actually SOUNDS, not what
                # the score says.
                GT = np.roll(GT, sounds_st, axis=0)
                if sounds_st < 0:
                    GT[sounds_st:, :] = False
                else:
                    GT[:sounds_st, :] = False
            rows.append(categorise(H, GT))
        if not rows:
            print(f"{label:<20}  (no usable pieces)")
            continue
        a = np.array(rows)
        m = a.mean(axis=0) * 100
        print(f"{label:<20}{m[0]:8.1f}%{m[1]:7.1f}%{m[2]:7.1f}%{m[3]:6.1f}%"
              f"{m[4]:7.1f}%   {m[0]+m[1]:9.1f}%", flush=True)

    print("\n'accounted' = correct + decay: energy on a pitch that is sounding")
    print("or still ringing.  Compare against 87.7% for the matched case in")
    print("octave_ghost_test.  A large drop for the sustained voices is the")
    print("expected boundary, not a bug.")


if __name__ == "__main__":
    main()
