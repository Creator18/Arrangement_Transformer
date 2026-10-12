"""Are the pYIN vocal labels the song's melody, or noise?

No ground-truth notes exist for MUSDB vocals, so test what correct labels
must satisfy, against chance:

  harmonic fit   the labelled vocal notes' pitch classes should be strong in
                 the FULL MIX's chroma at those moments (the singer is in tune
                 with the band) — compared with the same notes shifted by a
                 tritone, which is how a wrong-pitch label would look
  coverage       share of the vocal stem's loud frames (RMS > -35 dB of peak)
                 that carry a label — missed singing shows up here
  spurious       share of labelled frames where the vocal stem is near silent

    python eval/vocal_label_check.py --musdb .../musdb18hq --labels .../curriculum/vocal
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--musdb", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()
    import librosa
    rows = []
    for lab in sorted(pathlib.Path(args.labels).glob("labels_*.npz"))[: args.n]:
        name = lab.stem[len("labels_"):]
        tr = next((d for s in ("train", "test") for d in [pathlib.Path(args.musdb) / s / name] if d.exists()), None)
        if tr is None:
            continue
        voc = np.load(lab)["voc"]
        if len(voc) < 20:
            continue
        sr, hop = 16000, 512
        mix, _ = librosa.load(str(tr / "mixture.wav"), sr=sr, mono=True)
        vs, _ = librosa.load(str(tr / "vocals.wav"), sr=sr, mono=True)
        ch = librosa.feature.chroma_cqt(y=mix, sr=sr, hop_length=hop)
        ch = ch / (ch.max(0, keepdims=True) + 1e-9)
        f = np.clip((voc[:, 0] * sr / hop).astype(int), 0, ch.shape[1] - 1)
        pc = voc[:, 2].astype(int) % 12
        fit, chance = ch[pc, f].mean(), ch[(pc + 6) % 12, f].mean()
        rms = librosa.feature.rms(y=vs, hop_length=hop)[0]
        loud = rms > rms.max() * 10 ** (-35 / 20)
        lab_fr = np.zeros(len(rms), bool)
        for s, e, *_ in voc:
            lab_fr[int(s * sr / hop):int(e * sr / hop) + 1] = True
        cov = (lab_fr & loud).sum() / max(1, loud.sum())
        spur = (lab_fr & ~loud).sum() / max(1, lab_fr.sum())
        rows.append((fit, chance, cov, spur))
        print(f"  {name[:40]:<40} fit {fit:.2f} vs tritone {chance:.2f}  coverage {cov:.2f}  spurious {spur:.2f}",
              flush=True)
    r = np.array(rows)
    print(f"\n{len(r)} songs: harmonic fit {r[:, 0].mean():.3f} vs tritone-shifted {r[:, 1].mean():.3f} "
          f"(labels beat chance on {np.mean(r[:, 0] > r[:, 1]):.0%});  coverage of loud singing "
          f"{np.median(r[:, 2]):.2f};  labels on near-silence {np.median(r[:, 3]):.2f}")


if __name__ == "__main__":
    main()
