"""How much does the user's song-relative "solo" rule flag, against the fixed-margin rules of detectors v1 / v2?

The user's rule (2026-10-26): from the separated instrumental (everything but the vocal), take the song's own
loudness range -- softest to loudest while it sounds (2nd / 98th percentile, so silence, one click or a final crash
don't set it).  A moment with an active voice counts as solo-like when the instrumental is silent there (below the
song's softest) or within the bottom 30% of that range in dB: cut = soft + 0.3 * (loud - soft).

    python eval/solo_rel_stats.py --curves /scratch/basu.ang/solo_det/m2m_curves --splits /scratch/basu.ang/m2m/m2m_splits.json
"""
import argparse
import json
import pathlib

import numpy as np


def rel_classes(vdb, rdb, frac=0.3):
    """Per-frame class: 0 neither, 1 voice over a light band (bottom `frac` of the song's range), 2 voice over a silent
    band (below the song's softest), plus the song's (soft, loud, cut) in dB."""
    vmax = np.percentile(vdb, 99.5)
    active = (vdb > vmax - 35) & (vdb > -60)
    top = np.percentile(rdb, 99.5)
    sounding = rdb > max(-70.0, top - 50)                       # the instrumental is audibly playing
    soft, loud = (np.percentile(rdb[sounding], 2), np.percentile(rdb[sounding], 98)) if sounding.sum() > 50 else (top - 50, top)
    cut = soft + frac * (loud - soft)
    y = np.where(active & (rdb < soft), 2, np.where(active & (rdb <= cut), 1, 0))
    return y.astype(np.int64), (float(soft), float(loud), float(cut))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--frac", type=float, default=0.3)
    a = ap.parse_args()
    sp = json.load(open(a.splits))
    for name in ("test", "validation", "train"):
        tot = dict(voice=0, v1=0, v2=0, silent=0, light=0, n=0)
        per_song, ranges = [], []
        for sid in sp[name]:
            f = pathlib.Path(a.curves) / f"{sid}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            vdb, rdb, hop = z["vocal_db"].astype(float), z["rest_db"].astype(float), float(z["hop"])
            y, (soft, loud, cut) = rel_classes(vdb, rdb, a.frac)
            vmax = np.percentile(vdb, 99.5)
            active = (vdb > vmax - 35) & (vdb > -60)
            dom = vdb - rdb
            tot["voice"] += active.sum() * hop
            tot["v1"] += (active & (dom >= 12)).sum() * hop
            tot["v2"] += (active & (dom >= 8)).sum() * hop
            tot["silent"] += (y == 2).sum() * hop
            tot["light"] += (y == 1).sum() * hop
            tot["n"] += 1
            per_song.append(((y >= 1).sum() * hop, active.sum() * hop))
            ranges.append((soft, loud, cut))
        if not tot["n"]:
            continue
        r = np.array(ranges)
        sh = np.array([s / max(v, 1) for s, v in per_song])
        new = tot["silent"] + tot["light"]
        print(f"== {name}: {tot['n']} songs, {tot['voice'] / 60:.0f} min of active voice")
        print(f"   instrumental range per song (dB): softest median {np.median(r[:, 0]):.1f}, loudest median "
              f"{np.median(r[:, 1]):.1f}, span median {np.median(r[:, 1] - r[:, 0]):.1f}, cut median {np.median(r[:, 2]):.1f}")
        for k, v in (("v1 (voice 12 dB+ over the band)", tot["v1"]), ("v2 (voice 8 dB+ over the band)", tot["v2"]),
                     (f"new: band silent", tot["silent"]), (f"new: band light (bottom {a.frac:.0%})", tot["light"]),
                     ("new: silent or light", new)):
            print(f"   {k:<38} {v / 60:7.1f} min  ({v / max(tot['voice'], 1):5.1%} of voice time)")
        print(f"   songs where the new rule flags >= 10% / 25% / 50% of the voice: "
              f"{(sh >= 0.1).sum()} / {(sh >= 0.25).sum()} / {(sh >= 0.5).sum()} of {len(sh)}")


if __name__ == "__main__":
    main()
