"""Ear-test rig: everything the pitchmap contains, minus one rule at a time.

Built for listening, not for scoring.  The pipeline's reduction stages —
polyphony cap, melody-on-top, hand assignment, the two-voice restriction —
were each added for a reason and never heard in isolation, so there is no way
to tell which of them is helping.  This starts from EVERYTHING the
decomposition detects and applies exactly the stages named on the command
line, so each rule can be judged by ear on its own.

Stage 1 (the default) is scale filtering alone: every note the projection
finds, with out-of-key pitches removed and nothing else touched.  That is
deliberately far too many notes.  The point is to hear what the surplus
actually consists of before designing a rule to remove it — measured on this
song, the two-voice decode was keeping 23% of the activation energy and
discarding 77%, and the discarded part was the arpeggio.

PER-STEM MODE separates two failures that sound identical in the full mix: a
note the decomposition invented, and a note Demucs put in the wrong stem.
Each stem gets its own decomposition, its own roll and its own render, next to
the stem audio it came from, so "is that guitar note real?" is answerable by
listening to the guitar alone.  Drums are never included — they carry no pitch
and the pipeline drops them before the NMF.

Writes a self-contained HTML page, so notes can be seen and heard together
without going through the app.
"""
from __future__ import annotations

import argparse, json, pathlib, shutil, sys, warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
warnings.filterwarnings("ignore")
import scipy.signal, scipy.signal.windows                       # noqa: E402
if not hasattr(scipy.signal, "hann"):
    scipy.signal.hann = scipy.signal.windows.hann

OUT = ROOT / "backend" / "eartest"
NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def notes_from(H, stages, thresh_q, key, beats, downs, bpb, wav=None,
               per_band=0):
    from app.services.piano_projection import extract_notes, velocities, LOW
    from app.services import melody_decode as MD

    root, scale_name = key
    est = sorted(extract_notes(H, thresh_q=thresh_q, per_band=per_band),
                 key=lambda n: (n[0], n[2]))
    vels = velocities(H, est)
    notes = [{"pitch": int(p), "onset": float(a), "offset": float(b),
              "velocity": int(v), "role": "all"}
             for (a, b, p), v in zip(est, vels)]
    counts = [("raw", len(notes))]

    if "voices" in stages and wav is not None:
        from app.services.piano_projection import onset_times
        notes = MD.decode_two_voices(H, LOW, beats, downs, bpb,
                                     onsets=onset_times(wav))
        counts.append(("two-voice decode", len(notes)))
    if "scale" in stages:
        from app.services.transformation import drop_non_scale
        from app.services.music_theory import _scale_pitch_classes
        notes = drop_non_scale(notes, _scale_pitch_classes(root, scale_name))
        counts.append(("drop non-scale", len(notes)))
    if "snap" in stages:
        from app.services.transformation import snap_to_beat_grid
        notes = snap_to_beat_grid(notes, beats)
        counts.append(("snap to grid", len(notes)))
    if "top" in stages:
        from app.services.transformation import enforce_melody_on_top
        notes = enforce_melody_on_top(notes)
        counts.append(("melody on top", len(notes)))
    if "cap" in stages:
        from app.services.transformation import cap_polyphony, MAX_SOUNDING
        notes = cap_polyphony(notes, MAX_SOUNDING)
        counts.append((f"cap {MAX_SOUNDING}", len(notes)))
    return notes, counts


def render(notes, path):
    import soundfile as sf
    from app.services.sampler import render_notes
    from app.services.piano_projection import SR
    y = np.asarray(render_notes(notes, sr=SR), dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    pk = float(np.abs(y).max())
    if pk > 0:
        y = y / pk * 0.89
    sf.write(str(path), y, SR)
    return len(y) / SR


def copy_audio(src: pathlib.Path, dst: pathlib.Path):
    """Downmix to mono 22.05k so the page is not hauling 50 MB stereo WAVs."""
    import soundfile as sf, librosa
    from app.services.piano_projection import SR
    y, _ = librosa.load(str(src), sr=SR, mono=True)
    pk = float(np.abs(y).max())
    sf.write(str(dst), (y / pk * 0.89 if pk > 0 else y), SR)


PAGE = """<!doctype html><meta charset=utf-8><title>ear test — %(title)s</title>
<style>
 body{margin:0;background:#0a0a0c;color:#e8e4dc;font:13px/1.5 ui-monospace,monospace}
 header{padding:12px 16px;border-bottom:1px solid #24242a}
 .row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:5px 0}
 b{color:#f0a92c} .m{color:#7a7a86}
 button{background:#1a1a20;color:#e8e4dc;border:1px solid #33333c;border-radius:5px;
        padding:5px 11px;cursor:pointer;font:inherit}
 button:hover{border-color:#f0a92c}
 button.on{background:#f0a92c;color:#111;border-color:#f0a92c}
 #wrap{position:relative;overflow-x:auto;overflow-y:hidden} canvas{display:block}
 #seek{flex:1;min-width:280px;accent-color:#f0a92c;height:20px}
</style>
<header>
 <div class=row><b>%(title)s</b><span class=m id=info></span></div>
 <div class=row>
   <button id=play>play</button>
   <span class=m>source</span><span id=srcs></span>
 </div>
 <div class=row>
   <span class=m>hear</span>
   <button id=hp class=on>piano</button><button id=ha>stem audio</button>
   <button id=b10>&minus;10s</button><button id=f10>+10s</button>
 </div>
 <div class=row>
   <span class=m id=t>0:00</span>
   <input type=range id=seek min=0 max=%(dur)f step=0.01 value=0>
   <span class=m id=tot>0:00</span>
 </div>
</header>
<div id=wrap><canvas id=c></canvas></div>
<script>
const D=%(data)s, DUR=%(dur)f, PPS=%(pps)d;
const LO=21,HI=108,KH=7;
const c=document.getElementById('c'),x=c.getContext('2d');
c.width=Math.ceil(DUR*PPS); c.height=(HI-LO+1)*KH;
let cur=Object.keys(D)[0], mode='piano', el=null;
const au={};
for(const k in D){for(const w of ['piano','audio']){
  if(!D[k][w])continue; const a=new Audio(D[k][w]); a.preload='auto'; au[k+'|'+w]=a;}}
const black=p=>[1,3,6,8,10].includes(p%%12);
function draw(t){
 x.fillStyle='#0d0d11';x.fillRect(0,0,c.width,c.height);
 for(let p=LO;p<=HI;p++){const y=(HI-p)*KH;
  x.fillStyle=black(p)?'#101015':'#141419';x.fillRect(0,y,c.width,KH-1);
  if(p%%12===0){x.fillStyle='#1e1e26';x.fillRect(0,y,c.width,1);}}
 for(let s=0;s<DUR;s++){x.fillStyle=s%%5?'#15151b':'#22222c';x.fillRect(s*PPS,0,1,c.height);}
 for(const n of D[cur].notes){const y=(HI-n[0])*KH,w=Math.max(2,(n[2]-n[1])*PPS),v=n[3]/127;
  x.fillStyle=`hsl(${34+70*v} 85%% ${34+30*v}%%)`;x.fillRect(n[1]*PPS,y,w,KH-1);}
 x.fillStyle='#f0a92c';x.fillRect(t*PPS,0,1.5,c.height);
}
function player(){return au[cur+'|'+mode]||au[cur+'|piano'];}
const wrap=document.getElementById('wrap'),seek=document.getElementById('seek');
const fmt=t=>Math.floor(t/60)+':'+String(Math.floor(t%%60)).padStart(2,'0');
document.getElementById('tot').textContent=fmt(DUR);
let dragging=false;
function tick(){const a=player(),t=a?a.currentTime:0;draw(t);
 document.getElementById('t').textContent=fmt(t);
 if(!dragging)seek.value=t;
 wrap.scrollLeft=t*PPS-wrap.clientWidth*0.35;
 if(a&&!a.paused)requestAnimationFrame(tick);}
// While dragging, move only the playhead; committing on release keeps a big
// WAV from re-seeking on every pixel of travel, which stalls the audio.
seek.addEventListener('input',()=>{dragging=true;
 draw(+seek.value);document.getElementById('t').textContent=fmt(+seek.value);});
seek.addEventListener('change',()=>{dragging=false;goto(+seek.value);});
function goto(t){const a=player();if(!a)return;
 a.currentTime=Math.max(0,Math.min(t,DUR));tick();}
document.getElementById('b10').onclick=()=>{const a=player();if(a)goto(a.currentTime-10);};
document.getElementById('f10').onclick=()=>{const a=player();if(a)goto(a.currentTime+10);};
document.addEventListener('keydown',e=>{const a=player();if(!a)return;
 if(e.code==='Space'){e.preventDefault();document.getElementById('play').click();}
 if(e.code==='ArrowLeft')goto(a.currentTime-(e.shiftKey?1:5));
 if(e.code==='ArrowRight')goto(a.currentTime+(e.shiftKey?1:5));});
function swap(f){const a=player();const t=a?a.currentTime:0,p=a&&!a.paused;
 for(const k in au)au[k].pause(); f();
 const b=player(); if(b){b.currentTime=t; if(p)b.play();}
 document.getElementById('info').textContent=
   ' \\u00b7 '+D[cur].label+' \\u00b7 '+D[cur].notes.length+' notes \\u00b7 '+D[cur].key;
 [...document.querySelectorAll('#srcs button')].forEach(x=>
   x.classList.toggle('on',x.dataset.k===cur));
 document.getElementById('hp').classList.toggle('on',mode==='piano');
 document.getElementById('ha').classList.toggle('on',mode==='audio');
 tick();}
const sd=document.getElementById('srcs');
for(const k in D){const b=document.createElement('button');
 b.textContent=D[k].label;b.dataset.k=k;b.onclick=()=>swap(()=>{cur=k;});sd.appendChild(b);}
document.getElementById('hp').onclick=()=>swap(()=>{mode='piano';});
document.getElementById('ha').onclick=()=>swap(()=>{mode='audio';});
document.getElementById('play').onclick=e=>{const a=player();if(!a)return;
 if(a.paused){a.play();e.target.textContent='pause';tick();}
 else{a.pause();e.target.textContent='play';}};
c.onclick=e=>{const r=c.getBoundingClientRect();const a=player();
 if(a)a.currentTime=(e.clientX-r.left)/PPS;draw(a?a.currentTime:0);};
swap(()=>{});
</script>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--stages", default="scale")
    ap.add_argument("--stems", default="vocals,bass,guitar,other,piano",
                    help="drums are never valid here — no pitch content")
    ap.add_argument("--thresh-q", type=float, default=None)
    ap.add_argument("--name", default="stems")
    ap.add_argument("--per-band", type=int, default=0,
                    help="semitones per threshold band (12 = per octave); "
                         "0 keeps one threshold for all 88 pitches")
    ap.add_argument("--band-split", action="store_true",
                    help="bandpass the audio into octaves and decompose each "
                         "separately -- the literal reading, harmonics and all")
    ap.add_argument("--no-align-vocals", action="store_true",
                    help="leave the vocal line's timing untouched")
    ap.add_argument("--cascade", action="store_true",
                    help="use eval/reseparate.py output as the sources")
    ap.add_argument("--sum-first", action="store_true",
                    help="sum the pitched stems and transcribe ONCE, instead "
                         "of transcribing each stem and recombining notes")
    ap.add_argument("--rebuild-page", action="store_true",
                    help="re-emit the HTML from the last run's data; no NMF")
    args = ap.parse_args()

    if args.rebuild_page:
        import re
        page = OUT / f"{args.name}.html"
        src = page.read_text(encoding="utf-8")
        data = re.search(r"const D=(\{.*?\}), DUR=([0-9.]+)", src, re.S)
        if not data:
            raise SystemExit(f"could not read data out of {page}")
        page.write_text(PAGE % {
            "title": f"{args.name}: {args.stages}", "data": data.group(1),
            "dur": float(data.group(2)), "pps": 60,
        }, encoding="utf-8")
        print(f"rebuilt {page} from existing data (no decomposition run)")
        return

    from app.services.piano_projection import (
        _activations, key_from_activations, THRESH_Q)
    from app.services import melody_decode as MD

    sess = ROOT / "backend/temp" / args.session
    stages = {s.strip() for s in args.stages.split(",") if s.strip()}
    tq = args.thresh_q if args.thresh_q else THRESH_Q
    OUT.mkdir(exist_ok=True)

    meta = json.loads((sess / "components.json").read_text())
    hop, msr = float(meta.get("hop_length", 512)), float(meta.get("sr", 22050))
    beats = np.asarray(meta.get("beat_frames", []), float) * hop / msr
    downs = np.asarray(meta.get("downbeat_frames", []), float) * hop / msr
    bpb = int(meta.get("beats_per_bar", 4))

    # The key is taken ONCE from the full drumless mix and reused for every
    # stem.  Detecting it per stem would let an isolated bass or a guitar line
    # vote itself into a different key, and then each stem would be scale-
    # filtered against a different set — which is not a comparison.
    H_full = _activations(sess / "components/harmonic.wav",
                          MD.BASS_LO, MD.VOCAL_HI)
    key = key_from_activations(H_full) or (0, "major")
    print(f"key (from drumless mix, reused for all stems): "
          f"{NAMES[key[0]]} {key[1]}\n")

    sources, dur = {}, 0.0

    # SUM FIRST, then transcribe once.  The per-stem path decomposes each stem
    # on its own, so every Demucs misassignment becomes a transcription
    # problem: a guitar note filed under `piano` is decomposed as piano, and
    # the two stems then collide in the recombination.  Summing the pitched
    # accompaniment back together cancels that class of error outright — we
    # never ask which stem a note belonged to, because we never split them.
    #
    # `other` is included here.  The reason to drop it was that its separate
    # transcription could not be trusted; in a summed mix it is not being
    # transcribed separately, so that reason does not apply.
    if args.band_split:
        # The literal reading: cut the drumless mix into octave bands and give
        # each its own decomposition.  Expected to fail, and the failure is the
        # point of building it — a bass E2's partials land on MIDI 40, 52, 59,
        # 64, 68, 71, 74, 76, so every band above it receives harmonics with
        # the fundamental that explains them filtered out.  Each band can only
        # account for what it can see, so it must call those harmonics real
        # notes.  The whole-range decomposition has the option of attributing
        # them to the bass; a band-limited one does not.
        import soundfile as sf, librosa
        from scipy.signal import butter, sosfiltfilt
        from app.services.piano_projection import SR as _SR
        y, _ = librosa.load(str(sess / "components/harmonic.wav"), sr=_SR, mono=True)
        todo = []
        for lo in range(24, 96, 12):
            hi = lo + 11
            f_lo = 440.0 * 2 ** ((lo - 0.5 - 69) / 12)
            f_hi = min(440.0 * 2 ** ((hi + 0.5 - 69) / 12), _SR / 2 * 0.98)
            sos = butter(4, [f_lo / (_SR / 2), f_hi / (_SR / 2)], btype="band",
                         output="sos")
            yb = sosfiltfilt(sos, y).astype(np.float32)
            pk = float(np.abs(yb).max())
            if pk < 1e-4:
                continue
            path = OUT / f"band_{lo}.wav"
            sf.write(str(path), yb / pk * 0.89, _SR)
            todo.append((f"b{lo}", f"C{lo // 12 - 1}-B{lo // 12 - 1} "
                                  f"({f_lo:.0f}-{f_hi:.0f} Hz)", path, None))
        print(f"band-split into {len(todo)} octaves")
        args.stems = ""
    elif args.cascade:
        todo = [("rsvocals", "vocals (pass 1)", OUT / "rs_vocals.wav", None),
                ("rsbass", "bass (re-separated)", OUT / "rs_bass.wav", None),
                ("rsrest", "remainder", OUT / "rs_rest.wav", None)]
        missing = [p for _, _, p, _ in todo if not p.exists()]
        if missing:
            raise SystemExit(f"run eval/reseparate.py first; missing {missing}")
        args.stems = ""
    elif args.sum_first:
        import soundfile as sf, librosa
        from app.services.piano_projection import SR as _SR
        acc = OUT / "accompaniment.wav"
        parts = []
        for s in ("bass", "guitar", "piano", "other"):
            p = sess / f"components/stem_{s}.wav"
            if p.exists():
                y, _ = librosa.load(str(p), sr=_SR, mono=True)
                parts.append(y)
        n = max(len(y) for y in parts)
        mix = np.zeros(n, dtype=np.float32)
        for y in parts:
            mix[: len(y)] += y
        pk = float(np.abs(mix).max())
        sf.write(str(acc), (mix / pk * 0.89 if pk > 0 else mix), _SR)
        print(f"summed {len(parts)} pitched stems -> accompaniment.wav")
        todo = [("accomp", "accompaniment (summed)", acc, None),
                ("vocals", "vocals", sess / "components/stem_vocals.wav", None)]
        args.stems = ""
    else:
        todo = [("harmonic", "drumless mix",
                 sess / "components/harmonic.wav", H_full)]
    for s in [x.strip() for x in args.stems.split(",") if x.strip()]:
        if s == "drums":
            print("skipping drums: no pitch content, and the NMF never sees it")
            continue
        p = sess / f"components/stem_{s}.wav"
        if p.exists():
            todo.append((s, s, p, None))
        else:
            print(f"skipping {s}: {p.name} not found")

    from app.services.figure_peel import peel_stem, merge_overlaps, PEEL_STEMS
    from app.services.piano_projection import LOW, FPS
    combine = {}                       # stems that feed the recombination

    for key_id, label, wav, H in todo:
        if H is None:
            if key_id.startswith("b") and key_id[1:].isdigit():
                _lo = int(key_id[1:])
                H = _activations(wav, _lo, min(_lo + 11, MD.VOCAL_HI))
            else:
                H = _activations(wav, MD.BASS_LO, MD.VOCAL_HI)
        if H is None:
            print(f"{label}: silent stem, skipped")
            continue
        notes, counts = notes_from(H, stages, tq, key, beats, downs, bpb, wav,
                                   per_band=args.per_band)

        # Applies to EVERY source, unlike the peel and harmony stages.  The
        # baseline — full extraction, scale filter only — is the version that
        # sounded best, so the stage meant to improve it has to run on it.
        # A sung line is monophonic and legato: there are no simultaneous
        # notes to flam, so alignment has nothing to correct and can only
        # damage the expressive timing that makes a melody sound sung.
        if "align" in stages and not (args.no_align_vocals and "vocals" in key_id):
            from app.services.align_events import align_events, spread_report
            from app.services.piano_projection import onset_times as _ot
            before = spread_report(notes)
            notes, ai = align_events(notes, _ot(wav) if wav else None)
            after = spread_report(notes)
            counts.append(("align", len(notes)))
            print(f"  {label:<12} spread median "
                  f"{before.get('median_ms', 0):.0f}->{after.get('median_ms', 0):.0f} ms"
                  f"   >20ms {before.get('pct_over_20ms', 0):.0f}%->"
                  f"{after.get('pct_over_20ms', 0):.0f}%"
                  f"   {ai['events']} events, {ai['dropped_dupes']} dupes fused")

        # A: cut the bar down before looking for repetition.  Applied to the
        # accompaniment stems only, for the same reason the peel is.
        if "clean" in stages and key_id in PEEL_STEMS:
            from app.services.figure_peel import clean_by_onsets
            from app.services.piano_projection import onset_times
            notes = clean_by_onsets(notes, onset_times(wav), downs)
            counts.append(("onset+top4", len(notes)))

        # B: keep chord tones of each bar; needs no repetition at all.
        if "harmony" in stages and key_id in PEEL_STEMS:
            from app.services.figure_peel import harmony_filter
            from app.services.music_theory import _scale_pitch_classes
            notes, hinfo = harmony_filter(notes, downs,
                                          _scale_pitch_classes(*key))
            counts.append(("harmony", hinfo["kept"]))

        # Peel only the pitched ACCOMPANIMENT stems.  A sung line is
        # through-composed and "keep what repeats" would delete it; `other` is
        # whatever Demucs could not name and is dropped from the recombination
        # entirely, though it stays renderable so it can still be listened to.
        if "peel" in stages and key_id in PEEL_STEMS:
            notes, info = peel_stem(notes, H, LOW, downs, FPS)
            counts.append(("peeled", info["kept"]))
            print(f"{label:<14} " + "  ".join(f"{n}={v}" for n, v in counts))
            print(f"               cycle {info['cycle_bars']} bars, "
                  f"{info['sections']} sections, dropped {info['dropped']}")
            for line in info["log"]:
                print(line)
        else:
            print(f"{label:<14} " + "  ".join(f"{n}={v}" for n, v in counts))

        if (key_id in ("accomp", "vocals", "rsvocals", "rsbass", "rsrest")
                or key_id.startswith("b") and key_id[1:].isdigit()
                or key_id in PEEL_STEMS):
            combine[key_id] = notes
        pw, aw = OUT / f"{args.name}_{key_id}.wav", OUT / f"{args.name}_{key_id}_src.wav"
        dur = max(dur, render(notes, pw))
        copy_audio(wav, aw)
        sources[key_id] = {
            "label": label, "key": f"{NAMES[key[0]]} {key[1]}",
            "piano": pw.name, "audio": aw.name,
            "notes": [[int(n["pitch"]), round(float(n["onset"]), 3),
                       round(float(n["offset"]), 3), int(n.get("velocity", 80))]
                      for n in notes],
        }

    # Recombination: bass + guitar + piano + vocals, `other` deliberately out.
    if combine:
        merged = merge_overlaps([n for v in combine.values() for n in v])
        raw = sum(len(v) for v in combine.values())
        print(f"\nrecombined {' + '.join(sorted(combine))}: "
              f"{raw} -> {len(merged)} after merging overlaps "
              f"({raw - len(merged)} duplicates across stems)")
        pw = OUT / f"{args.name}_combined.wav"
        dur = max(dur, render(merged, pw))
        sources = {"combined": {
            "label": "RECOMBINED", "key": f"{NAMES[key[0]]} {key[1]}",
            "piano": pw.name, "audio": None,
            "notes": [[int(n["pitch"]), round(float(n["onset"]), 3),
                       round(float(n["offset"]), 3), int(n.get("velocity", 80))]
                      for n in merged]}, **sources}

    orig = OUT / "original.mp3"
    if not orig.exists():
        shutil.copy(sess / "original.mp3", orig)
    sources["mix"] = {"label": "original mix", "key": f"{NAMES[key[0]]} {key[1]}",
                      "piano": None, "audio": orig.name, "notes": []}

    page = OUT / f"{args.name}.html"
    page.write_text(PAGE % {
        "title": f"{args.name}: {args.stages}", "data": json.dumps(sources),
        "dur": dur, "pps": 60,
    }, encoding="utf-8")
    print(f"\nwrote {page}")


if __name__ == "__main__":
    main()
