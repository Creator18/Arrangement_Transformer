"""
Minimal SF2 sampler — reads a piano soundfont directly.
No FluidSynth binary required. Uses numpy + scipy for pitch-shifting.
Priority: TimGM6mb (Acoustic Grand Piano, bundled with pretty_midi)
          → UprightPianoKW (if present in project root)
"""
import struct
import numpy as np
from pathlib import Path
from functools import lru_cache
from scipy.signal import resample_poly
from math import gcd

# ── locate soundfont ──────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).parent.parent.parent.parent

def _pretty_midi_sf2() -> Path | None:
    try:
        import pretty_midi
        p = Path(pretty_midi.__file__).parent / "TimGM6mb.sf2"
        return p if p.exists() else None
    except ImportError:
        return None

_SF2_CANDIDATES = [
    _pretty_midi_sf2(),
    _REPO_ROOT / "UprightPianoKW-SF2-20220221" / "UprightPianoKW-SF2-20220221" / "UprightPianoKW-20220221.sf2",
]
SF2_PATH = next((p for p in _SF2_CANDIDATES if p is not None and p.exists()), None)


# ── RIFF parser ───────────────────────────────────────────────────────────────

def _iter_chunks(data: bytes, start: int, end: int):
    pos = start
    while pos + 8 <= end:
        tag = data[pos:pos+4].decode("latin-1")
        size = struct.unpack_from("<I", data, pos+4)[0]
        yield tag, pos+8, size
        pos += 8 + size + (size & 1)


def _parse_sf2(data: bytes):
    assert data[:4] == b"RIFF" and data[8:12] == b"sfbk", "Not a valid SF2 file"
    total = len(data)

    lists = {}
    for tag, dstart, size in _iter_chunks(data, 12, total):
        if tag == "LIST":
            ltype = data[dstart:dstart+4].decode("latin-1")
            lists[ltype] = (dstart+4, dstart+size)

    # ── sample PCM data ───────────────────────────────────────────────────────
    smpl_pcm = None
    sdta_s, sdta_e = lists["sdta"]
    for tag, dstart, size in _iter_chunks(data, sdta_s, sdta_e):
        if tag == "smpl":
            smpl_pcm = np.frombuffer(data[dstart:dstart+size], dtype="<i2").astype(np.float32) / 32768.0
            break

    # ── pdta sub-chunks ───────────────────────────────────────────────────────
    pdta_s, pdta_e = lists["pdta"]
    pdta = {}
    for tag, dstart, size in _iter_chunks(data, pdta_s, pdta_e):
        pdta[tag] = data[dstart:dstart+size]

    # ── sample headers (shdr) ─────────────────────────────────────────────────
    shdr_raw = pdta["shdr"]
    sample_hdrs = []
    for i in range(len(shdr_raw) // 46):
        o = i * 46
        name = shdr_raw[o:o+20].decode("latin-1").rstrip("\x00")
        start, end, sloop, eloop, sr = struct.unpack_from("<IIIII", shdr_raw, o+20)
        root = struct.unpack_from("B", shdr_raw, o+40)[0]
        correction = struct.unpack_from("b", shdr_raw, o+41)[0]
        stype = struct.unpack_from("<H", shdr_raw, o+44)[0]
        sample_hdrs.append({
            "name": name, "start": start, "end": end,
            "sloop": sloop, "eloop": eloop, "sr": sr,
            "root": root, "correction": correction, "type": stype,
        })

    # ── preset chain → find Bank 0 / Program 0 (Acoustic Grand Piano) ──────────
    phdr_raw = pdta["phdr"]
    pbag_raw = pdta["pbag"]
    pgen_raw = pdta["pgen"]

    pbags = [struct.unpack_from("<HH", pbag_raw, i*4) for i in range(len(pbag_raw) // 4)]
    pgens = [struct.unpack_from("<HH", pgen_raw, i*4) for i in range(len(pgen_raw) // 4)]

    piano_inst_idx = 0  # fallback: first instrument
    phdr_count = len(phdr_raw) // 38
    for i in range(phdr_count - 1):  # last entry is terminal EOP
        o = i * 38
        w_preset = struct.unpack_from("<H", phdr_raw, o+20)[0]
        w_bank   = struct.unpack_from("<H", phdr_raw, o+22)[0]
        bag_ndx  = struct.unpack_from("<H", phdr_raw, o+24)[0]
        next_bag = struct.unpack_from("<H", phdr_raw, o+38+24)[0]
        if w_preset == 0 and w_bank == 0:
            # Walk this preset's bags looking for generator 41 (instrument index)
            for pb in range(bag_ndx, next_bag):
                g_start = pbags[pb][0]
                g_end   = pbags[pb + 1][0]
                for g in range(g_start, g_end):
                    if pgens[g][0] == 41:          # sfGenOper = instrument
                        piano_inst_idx = pgens[g][1]
                        break
            break

    # ── instrument generators (igen) → map key → sample ──────────────────────
    ibag_raw = pdta["ibag"]
    igen_raw = pdta["igen"]
    inst_raw = pdta["inst"]

    ibag_count = len(ibag_raw) // 4
    ibags = [struct.unpack_from("<HH", ibag_raw, i*4) for i in range(ibag_count)]

    inst_count = len(inst_raw) // 22
    insts = []
    for i in range(inst_count):
        o = i * 22
        iname = inst_raw[o:o+20].decode("latin-1").rstrip("\x00")
        bag_idx = struct.unpack_from("<H", inst_raw, o+20)[0]
        insts.append((iname, bag_idx))

    igen_count = len(igen_raw) // 4
    igens = [struct.unpack_from("<HH", igen_raw, i*4) for i in range(igen_count)]

    # Build key→sample mapping from the piano instrument
    key_map = {}  # midi_pitch → sample_hdr
    pi   = piano_inst_idx
    pi1  = piano_inst_idx + 1

    for bag_i in range(insts[pi][1], insts[pi1][1] if pi1 < len(insts) else ibag_count - 1):
        zone_gen_start = ibags[bag_i][0]
        zone_gen_end = ibags[bag_i + 1][0]

        key_lo, key_hi = 0, 127
        vel_lo, vel_hi = 0, 127
        sample_id = None

        for g in range(zone_gen_start, zone_gen_end):
            oper, amount = igens[g]
            lo = amount & 0xFF
            hi = (amount >> 8) & 0xFF
            if oper == 43:   # keyRange
                key_lo, key_hi = lo, hi
            elif oper == 44: # velRange
                vel_lo, vel_hi = lo, hi
            elif oper == 53: # sampleID
                sample_id = lo | (hi << 8)

        if sample_id is not None and sample_id < len(sample_hdrs):
            sh = sample_hdrs[sample_id]
            if sh["type"] in (1, 2, 4):  # mono/right/left sample
                for k in range(key_lo, key_hi + 1):
                    if k not in key_map:
                        key_map[k] = sh

    return smpl_pcm, key_map


# ── module-level lazy load + per-pitch cache ──────────────────────────────────
_smpl_pcm: np.ndarray | None = None
_key_map:  dict | None = None
_loaded = False
_pitch_cache: dict[tuple, np.ndarray] = {}   # (pitch, sr) → resampled samples


def _ensure_loaded():
    global _smpl_pcm, _key_map, _loaded
    if _loaded:
        return
    if SF2_PATH is None:
        raise FileNotFoundError("Upright piano SF2 not found in project directory.")
    data = SF2_PATH.read_bytes()
    _smpl_pcm, _key_map = _parse_sf2(data)
    _loaded = True


def _get_shifted_sample(pitch: int, out_sr: int) -> np.ndarray:
    """
    Return the pitch-shifted sample for `pitch` at `out_sr`.
    Results are cached so each of the 88 MIDI pitches is computed only once
    per session, regardless of how many notes use that pitch.
    """
    cache_key = (pitch, out_sr)
    if cache_key in _pitch_cache:
        return _pitch_cache[cache_key]

    sh = _key_map.get(pitch)
    if sh is None:
        closest = min(_key_map.keys(), key=lambda k: abs(k - pitch))
        sh = _key_map[closest]

    raw = _smpl_pcm[sh["start"]:sh["end"]]
    if len(raw) == 0:
        _pitch_cache[cache_key] = raw
        return raw

    semitones = (pitch - sh["root"]) - sh["correction"] / 100.0
    ratio = 2.0 ** (semitones / 12.0)
    combined = ratio * sh["sr"] / out_sr

    precision = 1000
    up   = max(1, round(precision / combined))
    down = precision
    g    = gcd(up, down)
    up, down = up // g, down // g

    shifted = resample_poly(raw, up, down).astype(np.float32)
    _pitch_cache[cache_key] = shifted
    return shifted


def render_notes_sf2(notes: list, sr: int = 22050) -> np.ndarray:
    """
    Render note list using upright piano SF2 samples.
    Each MIDI pitch is resampled once and cached — subsequent notes at
    the same pitch are just array copies, making rendering ~10-50× faster.
    Falls back to additive synthesis if SF2 is unavailable.
    """
    try:
        _ensure_loaded()
    except FileNotFoundError:
        from app.services.audio_utils import render_notes_to_audio
        return render_notes_to_audio(notes, sr=sr)

    if not notes:
        return np.zeros(sr, dtype=np.float32)

    max_time = max(float(n.get("offset", n["onset"] + 0.5)) for n in notes)
    total_samples = int((max_time + 3.0) * sr)
    audio = np.zeros(total_samples, dtype=np.float32)

    for n in notes:
        pitch    = int(n["pitch"])
        amp      = float(n.get("velocity", 80)) / 127.0
        onset_s  = int(float(n["onset"]) * sr)
        note_dur = float(n.get("offset", n["onset"] + 0.5)) - float(n["onset"])

        shifted = _get_shifted_sample(pitch, sr)
        if len(shifted) == 0:
            continue

        sustain_samples = int(note_dur * sr)
        tail_samples    = int(min(3.0, note_dur * 1.5) * sr)
        use_len         = min(len(shifted), sustain_samples + tail_samples)

        seg = shifted[:use_len].copy()

        attack  = min(int(0.002 * sr), use_len // 8)
        if attack > 0:
            seg[:attack] *= np.linspace(0, 1, attack)

        rel_start = min(sustain_samples, use_len)
        rel_end   = min(rel_start + int(0.08 * sr), use_len)
        rel_len   = rel_end - rel_start
        if rel_len > 0:
            seg[rel_start:rel_end] *= np.linspace(1, 0, rel_len)
        if rel_end < use_len:
            seg[rel_end:] = 0.0

        end_s     = min(onset_s + use_len, total_samples)
        place_len = end_s - onset_s
        audio[onset_s:end_s] += seg[:place_len] * amp

    peak = np.abs(audio).max()
    if peak > 0:
        audio = (audio / peak * 0.92).astype(np.float32)
    return audio
