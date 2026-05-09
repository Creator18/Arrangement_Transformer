from typing import List, Dict, Any
import librosa


MELODY_ZONE_LOW = 48
VOICE_ZONE_LOW = 60


def _midi_to_cdef_name(pitch: int) -> str:
    """Convert MIDI pitch to CDEF letter+octave (e.g. 60 → C4)."""
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave = (pitch // 12) - 1
    name = note_names[pitch % 12]
    return f"{name}{octave}"


def notes_to_cdef(
    notes: List[Dict],
    tempo: float,
    time_sig_num: int = 4,
) -> str:
    """
    Convert transformed notes list to CDEF text format.
    Returns multi-line string with RH and LH lines per measure.
    """
    if not notes:
        return "RH: .\nLH: ."

    seconds_per_beat = 60.0 / tempo
    beat_resolution = seconds_per_beat

    # Collect all onsets to find measure boundaries
    max_offset = max(n["offset"] for n in notes)

    rh_notes = [n for n in notes if n["pitch"] >= MELODY_ZONE_LOW]
    lh_notes = [n for n in notes if n["pitch"] < MELODY_ZONE_LOW]

    def build_hand_line(hand_notes: List[Dict]) -> str:
        if not hand_notes:
            return "."

        measures_data: Dict[int, Dict[int, List[str]]] = {}

        for note in hand_notes:
            onset = note["onset"]
            measure_idx = int(onset / (seconds_per_beat * time_sig_num))
            beat_in_measure = int((onset % (seconds_per_beat * time_sig_num)) / seconds_per_beat)

            if measure_idx not in measures_data:
                measures_data[measure_idx] = {i: [] for i in range(time_sig_num)}
            if beat_in_measure < time_sig_num:
                measures_data[measure_idx][beat_in_measure].append(_midi_to_cdef_name(note["pitch"]))

        max_measure = max(measures_data.keys()) if measures_data else 0
        parts = []
        for m in range(max_measure + 1):
            beats = measures_data.get(m, {i: [] for i in range(time_sig_num)})
            beat_strs = []
            for b in range(time_sig_num):
                pitches = beats.get(b, [])
                if pitches:
                    beat_strs.append("-".join(sorted(set(pitches))))
                else:
                    beat_strs.append(".")
            parts.append(" ".join(beat_strs))

        return " | ".join(parts) + " |"

    rh_line = build_hand_line(rh_notes)
    lh_line = build_hand_line(lh_notes)

    return f"RH: {rh_line}\nLH: {lh_line}"
