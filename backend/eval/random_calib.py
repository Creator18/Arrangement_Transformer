"""Which instrument presets sound the key that was pressed?  (for eval/random_notes.py, 2026-11-02)

Some soundfont presets sound mainly an octave (or two) away from the key: church organs with 16' / 2' ranks,
layered synths, some string patches.  For the "identify everything" curriculum a label must be what you hear, so
those presets are left out.  Per soundfont x program x bank variation: three test notes (low / mid / high in the
family's range), CQT, and the strongest pitch class... no: the strongest PITCH, and the lowest pitch within 10 dB
of it.  OK = both are the key itself (or, for the strongest, a harmonic of it while the key is also strong).

    python eval/random_calib.py --sf2s a,b,c --sf-table T --out random_presets.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "eval"))
import random_notes as RN                 # noqa: E402
import render_arrangements as RA          # noqa: E402
import render_base as RB                  # noqa: E402


def sounding(y, keys, starts, dur):
    import librosa
    hop = 256
    C = np.abs(librosa.cqt(y.astype(np.float64), sr=RN.SR, hop_length=hop, fmin=librosa.midi_to_hz(21), n_bins=88 * 3,
                           bins_per_octave=36, tuning=0.0))
    fps = RN.SR / hop
    out = []
    for k, s in zip(keys, starts):
        seg = C[:, int((s + 0.05) * fps):int((s + dur - 0.05) * fps)]
        if seg.size == 0 or seg.max() <= 0:
            out.append(None); continue
        e = seg.max(1)[1::3]                                  # each key's CENTRE bin (no spill from the next key)
        db = 20 * np.log10(e / e.max() + 1e-9)
        strongest = int(np.argmax(e)) + 21
        low = np.nonzero(db[:max(0, k - 21 - 1)] > -10)[0]    # strong energy 2+ semitones BELOW the key
        lowest = int(low[0]) + 21 if len(low) else k
        out.append(dict(key=k, strongest=strongest, lowest=lowest, key_db=float(db[k - 21])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sf2s", required=True); ap.add_argument("--sf-table", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    RN.init(a.sf2s.split(","), a.sf_table)
    import fluidsynth
    progs = sorted(RA.CONT_PROGS | RA.PLUCK_PROGS | set(RB.LEAD) | set(RN.PLAIN))
    res, bad = [], []
    for sf2 in RB._S["sf2s"]:
        fs = fluidsynth.Synth(samplerate=float(RN.SR), gain=0.3); sfid = fs.sfload(sf2)    # once per soundfont
        for prog in progs:
            lo, hi = RN.RANGE.get(RN.family_of(prog), (36, 96))
            keys = [lo + (hi - lo) // 5, (lo + hi) // 2, hi - (hi - lo) // 5]
            for bank in RB.variants(sf2, prog):
                s = []
                fs.program_select(0, sfid, bank, prog); fs.cc(0, 1, 0); fs.cc(0, 93, 0)
                for k in keys:
                    fs.noteon(0, k, 100)
                    y = np.asarray(fs.get_samples(int(0.6 * RN.SR)), np.float32)[::2] / 32768.0
                    fs.noteoff(0, k); fs.all_sounds_off(0) if hasattr(fs, "all_sounds_off") else None
                    fs.get_samples(int(2.0 * RN.SR))          # 2 s of silence: no tail reaches the next test note
                    s += sounding(y, [k], [0.0], 0.55)
                ok = all(x is not None and x["lowest"] == x["key"] and x["key_db"] > -12 for x in s)
                res.append(dict(sf2=sf2, prog=prog, bank=int(bank), ok=ok, notes=s))
                if not ok:
                    bad.append([pathlib.Path(sf2).name, int(bank), int(prog)])
                    print(f"OFF {pathlib.Path(sf2).name} bank {bank} prog {prog}: "
                          + "; ".join(f"key {x['key']} -> lowest {x['lowest']} strongest {x['strongest']} (key {x['key_db']:.0f} dB)"
                                      if x else "silent" for x in s), flush=True)
        fs.delete()
    pathlib.Path(a.out).write_text(json.dumps(dict(bad=bad, results=res)))
    print(f"{len(res)} presets checked, {len(bad)} sound off the key", flush=True)


if __name__ == "__main__":
    main()
