import sys, json, pathlib, numpy as np, soundfile as sf, librosa
sys.path.insert(0, "."); sys.path.insert(0, "eval")
from app.services import sfz_sampler as Z
from app.services.sampler import _room
from scale_fill import scale_fill
D = pathlib.Path("eval/data/hum"); SR = 44100
y, sr = librosa.load("temp/0578fd23-f94d-4e7f-bd83-e8782fb3c21e/audio.wav", sr=22050, mono=True)
hop = 256
C = np.abs(librosa.cqt(y, sr=sr, hop_length=hop, fmin=librosa.midi_to_hz(21), n_bins=88, bins_per_octave=12))
def energy(p, t):
    k = int(p) - 21; f = int(round(t * sr / hop))
    return float(C[k, max(0, min(C.shape[1] - 1, f))]) if 0 <= k < 88 else 0.0
def render(notes, name):
    a = _room(Z.render_notes_sfz(notes, sr=SR), SR).astype(np.float32)
    sf.write(D / name, (a / (np.abs(a).max() + 1e-9) * 0.9).astype(np.float32), SR)
for tag in ("hum_piano_transcriber", "basicpitch_notes"):
    notes = json.load(open(D / f"0578fd23_{tag}.json"))
    kept, info = scale_fill(notes, energy=energy)
    render(kept, f"0578fd23_{tag}_scalefill.wav")
    print(tag, len(notes), "->", {k: v for k, v in info.items() if k != "keys"})
    print("   keys:", info["keys"])
