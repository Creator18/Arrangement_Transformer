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
    _REPO_ROOT / "OmegaGMGS2.sf2",
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
# cache: (pitch, sr) → (resampled_samples, loop_start, loop_end)
_pitch_cache: dict[tuple, tuple] = {}


def _ensure_loaded():
    global _smpl_pcm, _key_map, _loaded
    if _loaded:
        return
    if SF2_PATH is None:
        raise FileNotFoundError("No SF2 found in project directory.")
    data = SF2_PATH.read_bytes()
    _smpl_pcm, _key_map = _parse_sf2(data)
    _loaded = True


def _get_shifted_sample(pitch: int, out_sr: int) -> tuple:
    """
    Return (samples, loop_start, loop_end) for `pitch` at `out_sr`.
    loop_start/loop_end are indices into `samples` in output-sample units.
    If the SF2 sample has no valid loop region, loop_start == loop_end == 0.
    Results are cached per (pitch, sr).
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
        result = (raw, 0, 0)
        _pitch_cache[cache_key] = result
        return result

    semitones = (pitch - sh["root"]) - sh["correction"] / 100.0
    ratio    = 2.0 ** (semitones / 12.0)
    combined = ratio * sh["sr"] / out_sr

    precision = 1000
    up   = max(1, round(precision / combined))
    down = precision
    g    = gcd(up, down)
    up, down = up // g, down // g

    shifted = resample_poly(raw, up, down).astype(np.float32)
    scale   = len(shifted) / max(len(raw), 1)

    # Scale loop points from raw-sample coordinates to resampled coordinates
    sloop_rel = sh["sloop"] - sh["start"]
    eloop_rel = sh["eloop"] - sh["start"]
    loop_start = int(sloop_rel * scale)
    loop_end   = int(eloop_rel * scale)

    # Validate — loop must be a non-trivial region inside the resampled array
    loop_start = max(0, min(loop_start, len(shifted) - 1))
    loop_end   = max(0, min(loop_end,   len(shifted)))
    if loop_end - loop_start < 64:   # degenerate / missing loop → no loop
        loop_start = loop_end = 0

    result = (shifted, loop_start, loop_end)
    _pitch_cache[cache_key] = result
    return result


def _seamless_tile(loop_region: np.ndarray, needed: int) -> np.ndarray:
    """
    Tile loop_region to fill `needed` samples without clicks.
    Pre-applies a crossfade at the loop boundary so each repeat blends
    smoothly into the next, eliminating the DC-jump that causes choppiness.
    """
    loop_len = len(loop_region)
    if loop_len == 0:
        return np.zeros(needed, dtype=np.float32)

    xfade = min(512, loop_len // 4)

    if xfade < 4:
        reps = (needed // loop_len) + 2
        return np.tile(loop_region, reps)[:needed].astype(np.float32)

    fade_out = np.linspace(1.0, 0.0, xfade, dtype=np.float32)
    fade_in  = np.linspace(0.0, 1.0, xfade, dtype=np.float32)

    # Blend the end of the loop into its own beginning so tiling is seamless
    seamless = loop_region.copy()
    seamless[-xfade:] = loop_region[-xfade:] * fade_out + loop_region[:xfade] * fade_in

    reps = (needed // loop_len) + 2
    return np.tile(seamless, reps)[:needed].astype(np.float32)


def _crossfade_join(a: np.ndarray, b: np.ndarray, xfade: int) -> np.ndarray:
    """Concatenate two arrays with a short crossfade at the join point."""
    xfade = min(xfade, len(a), len(b))
    if xfade < 2:
        return np.concatenate([a, b]).astype(np.float32)

    fade_out = np.linspace(1.0, 0.0, xfade, dtype=np.float32)
    fade_in  = np.linspace(0.0, 1.0, xfade, dtype=np.float32)
    blend    = a[-xfade:] * fade_out + b[:xfade] * fade_in
    return np.concatenate([a[:-xfade], blend, b[xfade:]]).astype(np.float32)


def _build_note_segment(shifted: np.ndarray, loop_start: int, loop_end: int,
                        sustain_samples: int, release_samples: int, sr: int) -> np.ndarray:
    """
    Build a note segment:
      - Sustain: looped sample held for sustain_samples
      - Release: natural ring-out over release_samples with exponential decay
      - Always padded to full sustain+release length so the fade never cuts short
    """
    xfade        = min(256, sr // 88)
    sustain_samples = max(0, sustain_samples)
    total_needed = max(1, sustain_samples + release_samples)
    has_loop     = (loop_end > loop_start + 64)

    if not has_loop or len(shifted) >= total_needed:
        use_len = min(len(shifted), total_needed)
        seg = np.zeros(total_needed, dtype=np.float32)
        seg[:use_len] = shifted[:use_len]
    else:
        pre_loop    = shifted[:loop_start]
        loop_region = shifted[loop_start:loop_end]
        post_loop   = shifted[loop_end:]

        sustain_after_pre = max(0, sustain_samples - len(pre_loop))
        looped = _seamless_tile(loop_region, sustain_after_pre + xfade)

        tail_len = min(len(post_loop), release_samples + xfade)
        tail = post_loop[:tail_len] if tail_len > 0 else \
               np.zeros(release_samples, dtype=np.float32)

        if len(pre_loop) > 0:
            mid = _crossfade_join(pre_loop, looped, xfade)
        else:
            mid = looped
        raw_seg = _crossfade_join(mid, tail, xfade)

        # Pad to total_needed so release fade always has full room
        seg = np.zeros(total_needed, dtype=np.float32)
        copy_len = min(len(raw_seg), total_needed)
        seg[:copy_len] = raw_seg[:copy_len]

    # Attack ramp (3 ms)
    attack = min(int(0.003 * sr), len(seg) // 8)
    if attack > 0:
        seg[:attack] *= np.linspace(0.0, 1.0, attack)

    # Exponential release decay — sounds more natural than linear
    if release_samples > 0:
        t = np.linspace(0.0, 1.0, release_samples, dtype=np.float32)
        decay = np.exp(-4.0 * t).astype(np.float32)   # e^0=1 → e^-4≈0.018
        seg[sustain_samples:sustain_samples + release_samples] *= decay

    return seg


def render_notes_sf2(notes: list, sr: int = 22050) -> np.ndarray:
    """
    Render note list using SF2 piano samples with proper sustain looping.
    Falls back to additive synthesis if no SF2 is available.
    """
    try:
        _ensure_loaded()
    except FileNotFoundError:
        from app.services.audio_utils import render_notes_to_audio
        return render_notes_to_audio(notes, sr=sr)

    if not notes:
        return np.zeros(sr, dtype=np.float32)

    max_time      = max(float(n.get("offset", n["onset"] + 0.5)) for n in notes)
    total_samples = int((max_time + 5.0) * sr)
    audio         = np.zeros(total_samples, dtype=np.float32)

    for n in notes:
        pitch    = int(n["pitch"])
        amp      = float(n.get("velocity", 80)) / 127.0
        onset_s  = int(float(n["onset"]) * sr)
        note_dur = float(n.get("offset", n["onset"] + 0.5)) - float(n["onset"])

        if onset_s < 0:
            continue
        shifted, loop_start, loop_end = _get_shifted_sample(pitch, sr)
        if len(shifted) == 0:
            continue

        note_dur = max(0.05, note_dur)   # minimum 50 ms — guards against offset ≤ onset
        sustain_samples = int(note_dur * sr)
        # Generous ring-out: at least 1.5 s, scales with note length, max 3 s
        release_samples = int(min(3.0, max(1.5, note_dur * 0.8)) * sr)

        seg = _build_note_segment(shifted, loop_start, loop_end,
                                  sustain_samples, release_samples, sr)

        end_s     = min(onset_s + len(seg), total_samples)
        place_len = end_s - onset_s
        audio[onset_s:end_s] += seg[:place_len] * amp

    peak = np.abs(audio).max()
    if peak > 0:
        audio = (audio / peak * 0.92).astype(np.float32)
    return audio
