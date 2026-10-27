import { useState, useRef, useEffect, useCallback, useMemo } from 'react'
import { playNote } from '../utils/pianovoice'
import { heldKeys } from '../utils/hands'
import { attach as attachTone, setTone } from '../utils/tonechain'
import { motion, AnimatePresence } from 'framer-motion'
import { wipeReveal } from '../utils/animefx'

// ── layout constants ──────────────────────────────────────────────────────────
const NOTE_W_FALLBACK = 8     // initial px per semitone before container is measured
const NOTE_W_MIN      = 6     // floor — keyboard scrolls horizontally if narrower
// Ceiling on per-semitone width, and it has to depend on context.
//
// This was 14, sized when the roll was only ever a 370px embed.  On the
// full-screen results stage that left the keyboard ~120px short of its own
// container, so it was raised to 22 — which then inflated the Layers embeds
// too, giving five stacked review panels giant keyboards and ~550px each.
// Caught by screenshotting the Layers tab; a global constant was the wrong
// shape for a component that now renders at two very different sizes.
//
// Derived from rollHeight rather than added as another prop, so no call site
// has to know about it: a roll given the screen gets keys to match, and a
// compact embed keeps the original proportions.
const NOTE_W_MAX_STAGE   = 22
const NOTE_W_MAX_COMPACT = 14
const KEY_H           = 78    // keyboard panel height (fixed at bottom)
const PX_PER_S        = 80    // px per second (time → Y axis)
const TIMELINE_W      = 28    // left strip for measure / beat labels
// Default roll viewport height.  Overridable per instance via `rollHeight`,
// because the results view now gives the roll the whole screen while smaller
// embeds (the Layers tab) still want a compact one.
const ROLL_MAX_H      = 370
// "Future-empty" padding at the top of the SVG so the now-line can stay glued
// to the bottom of the viewport (= top of the keyboard) at the very end of
// the song without the scroll going negative.  This is the MINIMUM: the pad
// used is at least the roll's own height (`pad` below).  A fixed 380 px was
// shorter than the full-screen results roll, so the scroll clamped at 0 for
// the last few seconds and the final notes froze above the keys.
const VIEWPORT_PAD    = 380

// Which pitch classes are black keys
const BLACK_PCS = new Set([1, 3, 6, 8, 10])
const isBlack   = p => BLACK_PCS.has(p % 12)

// ── Diatonic key geometry ────────────────────────────────────────────────────
// A real piano has 7 white keys per octave, all the same bottom-width, with
// black keys riding on top between specific pairs.  These tables map each
// pitch class onto either its white-key index within the octave or, for black
// keys, the white-key boundary it straddles.
const PC_TO_WHITE_IDX = { 0: 0, 2: 1, 4: 2, 5: 3, 7: 4, 9: 5, 11: 6 }
const BLACK_BOUNDARY  = { 1: 1, 3: 2, 6: 4, 8: 5, 10: 6 }
const isWhitePc       = (pc) => pc in PC_TO_WHITE_IDX

// Compute the visual { x, w } of a pitch within the keyboard.
//   wkw    = white-key bottom width = octave_width / 7 = (12·noteW)/7
//   blackW = 60% of wkw (matches a real piano roughly)
// White pitches are placed at their diatonic slot; black pitches sit centered
// on the boundary between the two whites they belong between.  Total octave
// width remains 12·noteW so the overall keyboard span doesn't change.
function pitchPos(pitch, midiLo, noteW) {
  const wkw    = (noteW * 12) / 7
  const blackW = wkw * 0.6
  const octaveIdx    = Math.floor((pitch - midiLo) / 12)
  const pc           = ((pitch % 12) + 12) % 12
  const octaveStartX = octaveIdx * 12 * noteW
  if (isWhitePc(pc)) {
    return { x: octaveStartX + PC_TO_WHITE_IDX[pc] * wkw, w: wkw, isWhite: true }
  }
  return {
    x: octaveStartX + BLACK_BOUNDARY[pc] * wkw - blackW / 2,
    w: blackW,
    isWhite: false,
  }
}

// ── register colours ──────────────────────────────────────────────────────────
const REG_COLOR = {
  bass:   { fill: '#3b82f6', glow: 'rgba(59,130,246,0.75)'  },
  melody: { fill: '#f87171', glow: 'rgba(248,113,113,0.75)' },
  voice:  { fill: '#a78bfa', glow: 'rgba(167,139,250,0.75)' },
}

function classify(pitch) {
  if (pitch < 48)  return 'bass'
  if (pitch <= 60) return 'melody'
  return 'voice'
}

// ── part selector config ──────────────────────────────────────────────────────
const PARTS = [
  { key: 'all',    label: 'All',    color: '#a1a1aa' },
  { key: 'bass',   label: 'Bass',   color: '#3b82f6' },
  { key: 'melody', label: 'Melody', color: '#f87171' },
  { key: 'voice',  label: 'Voice',  color: '#a78bfa' },
]
const PART_RANGE = {
  all:    [0,   127],
  bass:   [0,   47],
  melody: [48,  60],
  voice:  [61,  127],
}

function fmt(s) {
  if (!isFinite(s) || s < 0) return '0:00'
  const m   = Math.floor(s / 60)
  const sec = Math.floor(s % 60)
  return `${m}:${sec.toString().padStart(2, '0')}`
}

// ── Horizontal Keyboard SVG ───────────────────────────────────────────────────
// Runs LEFT → RIGHT: bass on the left, treble on the right, with real-piano
// geometry — every white key the same width (12·noteW/7), black keys riding
// over the boundary between the two whites they sit between.
//
// The look follows the flat "Keys" keyboard design, in its dark-mode palette
// because the app is dark: plain white keys with a hairline border and
// rounded bottoms, black keys three-fifths of the height, an orange pressed
// state, and a label on every C.  The previous ivory gradients, sheen stripes,
// press-dip and per-register glow are gone.
//
// Kept on purpose, because they carry information rather than decoration:
//   - keys light up from playback (activeKeys, with the hand-reach model)
//   - the re-strike flash, keyed on the onset, so a repeated note reads as
//     two strikes rather than one long press
//   - click-to-hear while paused
const KEY_COLORS = {
  white:        '#e5e5e5',   // neutral-200
  whiteBorder:  '#404040',   // neutral-700
  whitePressed: '#fdba74',   // orange-300
  black:        '#171717',   // neutral-900
  blackBorder:  '#0a0a0a',   // neutral-950
  blackPressed: '#f97316',   // orange-500
  label:        '#a3a3a3',   // neutral-400
}

// A key with square top corners and rounded bottom ones (rounded-b-md).
function bottomRoundedPath(x, w, h, r) {
  const rr = Math.max(0, Math.min(r, w / 2, h / 2))
  return [
    `M ${x} 0`, `H ${x + w}`, `V ${h - rr}`,
    `Q ${x + w} ${h} ${x + w - rr} ${h}`, `H ${x + rr}`,
    `Q ${x} ${h} ${x} ${h - rr}`, 'Z',
  ].join(' ')
}

// A pressed key takes the colour of the note landing on it (the roll colours
// notes by register), with a soft glow in that colour.  Glow and the black
// keys' cast shadow are CSS drop-shadows, not SVG filters: measured with the
// roll fps rig, CSS drop-shadow composites cheaply where an SVG filter re-runs
// every frame.
const keyGlow = (pitch) => REG_COLOR[classify(pitch)]

function KeyboardSVG({ midiLo, midiHi, activeKeys, noteW, interactive, onKeyDown }) {
  const pitchRange = midiHi - midiLo + 1
  const totalW     = TIMELINE_W + pitchRange * noteW
  const WHITE_H    = KEY_H - 2
  const BLACK_H    = Math.round(KEY_H * 0.6)          // three-fifths of the height
  const wkw        = (noteW * 12) / 7
  const GAP        = Math.max(1, Math.min(2, wkw * 0.08))   // separate keys, as on the reference
  const radius     = Math.min(6, wkw * 0.22)          // rounded bottoms, scaled for narrow keys
  const labelSize  = Math.max(6, Math.min(10, wkw * 0.42))
  const pressable  = (pitch) => interactive
    ? { onMouseDown: (e) => { e.stopPropagation(); onKeyDown?.(pitch) },
        style: { cursor: 'pointer' } }
    : { style: { cursor: 'default' } }

  return (
    <svg width={totalW} height={KEY_H} style={{ display: 'block', flexShrink: 0 }}>
      {/* Timeline-strip patch */}
      <rect x={0} y={0} width={TIMELINE_W} height={KEY_H} fill="rgba(0,0,0,0.45)" />
      <text x={TIMELINE_W / 2} y={KEY_H / 2}
        fontSize={6} fill="rgba(245,158,11,0.45)"
        fontFamily="ui-monospace,monospace" fontWeight="700"
        textAnchor="middle" dominantBaseline="middle" letterSpacing="0.5">
        KEY
      </text>

      {/* ── White keys (bottom layer) ── */}
      {Array.from({ length: pitchRange }, (_, i) => {
        const pitch = midiLo + i
        if (isBlack(pitch)) return null
        const pos    = pitchPos(pitch, midiLo, noteW)
        const x      = TIMELINE_W + pos.x + GAP / 2 + 0.5     // half-pixel: crisp 1px border
        const w      = pos.w - GAP - 1
        const active = activeKeys.has(pitch)
        const d      = bottomRoundedPath(x, w, WHITE_H, radius)
        const g      = keyGlow(pitch)
        return (
          <g key={pitch} {...pressable(pitch)}>
            <path d={d}
              fill={active ? g.fill : KEY_COLORS.white}
              stroke={active ? g.fill : KEY_COLORS.whiteBorder} strokeWidth={1}
              style={active ? { filter: `drop-shadow(0 0 6px ${g.glow})` } : undefined} />
            {/* Re-strike flash: keyed on the ONSET so a repeated note mounts a
                new element and the CSS keyframe restarts. */}
            {active && (
              <path key={activeKeys.get(pitch)} className="key-strike" d={d} fill="#ffffff" />
            )}
            {pitch % 12 === 0 && (
              <text x={x + w / 2} y={WHITE_H - 6}
                fontSize={labelSize} fill={KEY_COLORS.label}
                fontFamily="ui-sans-serif,system-ui,sans-serif"
                textAnchor="middle" style={{ pointerEvents: 'none' }}>
                C{Math.floor(pitch / 12) - 1}
              </text>
            )}
          </g>
        )
      })}

      {/* ── Black keys (top layer), three-fifths of the height, casting a
           shadow onto the white keys.  One filter on the whole group, so the
           shadow costs one composite rather than one per key. ── */}
      <g style={{ filter: 'drop-shadow(0 2px 2px rgba(0,0,0,0.55))' }}>
      {Array.from({ length: pitchRange }, (_, i) => {
        const pitch = midiLo + i
        if (!isBlack(pitch)) return null
        const pos    = pitchPos(pitch, midiLo, noteW)
        const bw     = pos.w * 0.9                        // a touch narrower, as on the reference
        const x      = TIMELINE_W + pos.x + (pos.w - bw) / 2
        const active = activeKeys.has(pitch)
        const d      = bottomRoundedPath(x, bw, BLACK_H, Math.min(radius, bw * 0.35))
        const g      = keyGlow(pitch)
        return (
          <g key={pitch} {...pressable(pitch)}>
            <path d={d}
              fill={active ? g.fill : KEY_COLORS.black}
              stroke={active ? g.fill : KEY_COLORS.blackBorder} strokeWidth={1}
              style={active ? { filter: `drop-shadow(0 0 7px ${g.glow})` } : undefined} />
            {active && (
              <path key={activeKeys.get(pitch)} className="key-strike" d={d} fill="#ffffff" />
            )}
          </g>
        )
      })}
      </g>

      {/* Soft shadow under the keyboard's top edge: the keys sit below the roll */}
      <rect x={TIMELINE_W} y={0} width={pitchRange * noteW} height={6}
        fill="url(#key-top-shade)" style={{ pointerEvents: 'none' }} />
      <defs>
        <linearGradient id="key-top-shade" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#000" stopOpacity="0.45" />
          <stop offset="100%" stopColor="#000" stopOpacity="0" />
        </linearGradient>
      </defs>

      {/* ── Middle-C dashed divider ── */}
      {midiLo <= 60 && 60 <= midiHi && (() => {
        const divX = TIMELINE_W + (60 - midiLo) * noteW
        return (
          <line x1={divX} y1={0} x2={divX} y2={KEY_H}
            stroke="rgba(245,158,11,0.55)" strokeWidth={1} strokeDasharray="3 3" />
        )
      })()}
    </svg>
  )
}

// ── Main component ────────────────────────────────────────────────────────────
export default function PianoRollPlayer({
  notes           = [],
  getPartUrl      = () => '',
  label           = 'Track',
  tempo           = 120,
  keyboardStart   = 36,   // lowest MIDI note shown (C2)
  keyboardOctaves = 7,
  defaultPart     = 'all', // controlled from parent (results page part pads)
  tone            = null,  // { bassDb, trebleDb, room } — live monitoring EQ
  rollHeight      = ROLL_MAX_H,
  chrome          = true,  // false hides the header/transport for embedded use
}) {
  const [activePart,  setActivePart]  = useState(defaultPart)
  const [playing,     setPlaying]     = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration,    setDuration]    = useState(0)
  const [loading,     setLoading]     = useState(false)
  const [dragging,    setDragging]    = useState(false)

  const audioRef   = useRef(null)
  const rollRef    = useRef(null)   // roll viewport (clips; does NOT scroll)
  const stageRef   = useRef(null)   // the transformed layer inside it
  const wrapperRef = useRef(null)   // outer wrapper — measured for fit-to-width
  const isDragging = useRef(false)
  const rafRef     = useRef(null)

  // Frame-accurate clock, deliberately NOT state.
  //
  // The rAF loop below used to setCurrentTime() every frame, which re-rendered
  // this whole component 60 times a second — and this component renders one
  // SVG <g> per note, so at ~2500 notes that is 5000 React elements rebuilt
  // and diffed per frame, times the four players the results page mounts.
  // Nothing the clock drives actually needs a React render: the scroll is a
  // transform, the readout is two text nodes, and the key highlights only
  // change when a note starts or ends.  So the clock lives in a ref and the
  // few things it touches are written directly to the DOM.
  const timeRef     = useRef(0)
  const scrollRef   = useRef(0)
  const clockElRef  = useRef(null)
  const barElRef    = useRef(null)

  // Move the roll by TRANSFORM, not scrollTop.
  //
  // scrollTop on a container holding a 20 000 px SVG repaints that SVG every
  // frame — the browser has no way to know the vector content is unchanged.
  // translate3d is composited on the GPU: the layer is rasterised once and
  // thereafter only moved, so a long arrangement costs the same as a short
  // one.  `scrollRef` keeps the logical scroll offset so seek arithmetic is
  // unchanged from when this really was a scroll container.
  const applyScroll = useCallback((v) => {
    const y = Math.max(0, v)
    scrollRef.current = y
    const st = stageRef.current
    if (st) st.style.transform = `translate3d(0, ${-y}px, 0)`
  }, [])

  // Sync when parent results-page part pad changes
  useEffect(() => { setActivePart(defaultPart) }, [defaultPart])

  // MIDI range
  // Stage rolls get wide keys; compact review embeds keep the old scale.
  const noteWMax   = rollHeight > 450 ? NOTE_W_MAX_STAGE : NOTE_W_MAX_COMPACT

  const midiLo     = keyboardStart
  const midiHi     = keyboardStart + keyboardOctaves * 12 - 1
  const pitchRange  = midiHi - midiLo + 1

  // ── Responsive sizing ──────────────────────────────────────────────────────
  // Measure the wrapper width on mount + on resize, then derive a per-semitone
  // column width (`noteW`) so that the full 7-octave keyboard fits inside the
  // holder without horizontal cutoff.  Clamped to [NOTE_W_MIN, NOTE_W_MAX]; on
  // very narrow viewports the keyboard reverts to scrolling at the floor width.
  const [wrapperW, setWrapperW] = useState(0)
  useEffect(() => {
    const el = wrapperRef.current
    if (!el) return
    const update = () => setWrapperW(el.clientWidth)
    update()
    const ro = new ResizeObserver(update)
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const noteW = wrapperW > 0
    ? Math.max(NOTE_W_MIN, Math.min(noteWMax, (wrapperW - TIMELINE_W) / pitchRange))
    : NOTE_W_FALLBACK

  // SVG dimensions
  const rollSvgW   = TIMELINE_W + pitchRange * noteW

  const maxTime    = notes.length
    ? Math.max(...notes.map(n => n.offset ?? n.onset + 0.5))
    : 10
  // SVG layout (notes fall *into* the keyboard):
  //   • y = 0 sits at "future-empty" padding above the latest note
  //   • y = pad is t = maxTime
  //   • y grows downward as time goes earlier — so notes with smaller onsets
  //     end up nearer the bottom, ready to drop into the keyboard.
  // The +1.5 s of past-padding at the bottom keeps the view from clipping the
  // very first downbeat when the user pauses at t = 0.
  // ...plus the audio's tail past the last note (release + reverb, ~2 s): playback runs to `duration`, so the
  // now-line must still be able to reach the keyboard there.
  const pad        = Math.max(VIEWPORT_PAD, rollHeight) + Math.max(0, (duration || 0) - maxTime) * PX_PER_S
  const rollSvgH   = Math.max(800, (maxTime + 1.5) * PX_PER_S + pad)
  const yOf        = (t) => (maxTime - t) * PX_PER_S + pad

  // Beat / measure layout
  const beatDur    = 60 / tempo
  const beatCount  = Math.ceil(maxTime / beatDur)

  // Part filter
  const [partLo, partHi] = PART_RANGE[activePart]

  // ── Active pitches ──────────────────────────────────────────────────────────
  // Map of pitch -> onset of the note currently sounding on that key.
  //
  // Two things this fixes over the old Set of pitches.
  //
  // HARD CAP OF 10.  A hand has five fingers; a player has two hands.  The
  // roll was colouring a key for a note's whole SOUNDING duration, which is
  // not the same thing as the key being held — under pedal a note keeps
  // ringing long after the finger has left.  Conflating the two is how the
  // keyboard ended up showing more keys down than a human has.  When more
  // than ten are sounding we keep the ten most recently struck, which is
  // both the physical truth (you release the oldest to free a finger) and a
  // strict guarantee the display can never break.  Everything older is still
  // audible — that is what the sustain pedal is for — it just is not drawn
  // as a pressed key, because it isn't one.
  //
  // ONSET AS THE VALUE, not just membership.  A repeated note kept the same
  // pitch in the Set continuously, so the key never visibly lifted between
  // two strikes even though the roll clearly showed two. Keying the flash on
  // the onset makes a re-strike re-trigger.
  //
  // The candidate notes are prepared ONCE, sorted by onset, so the per-frame
  // work is a walk over a window rather than a scan of the whole song.  With
  // the scan, every frame touched all ~2500 notes and parseInt'd each pitch;
  // sorted, a frame only visits notes that have already started and have not
  // yet been passed by the cursor.
  const timed = useMemo(() => {
    const out = []
    for (const n of notes) {
      const p = parseInt(n.pitch, 10)
      if (p < midiLo || p > midiHi) continue
      if (!(activePart === 'all' || (p >= partLo && p <= partHi))) continue
      const on = parseFloat(n.onset)
      out.push({ pitch: p, onset: on, offset: n.offset ?? on + 0.25 })
    }
    out.sort((a, b) => a.onset - b.onset)
    return out
  }, [notes, activePart, partLo, partHi, midiLo, midiHi])

  const [activeKeys, setActiveKeys] = useState(() => new Map())
  const cursorRef = useRef(0)      // index of the first note not yet started

  const keysAt = useCallback((t) => {
    if (t <= 0) return new Map()
    // Advance or rewind the cursor to the first note with onset > t.  Seeking
    // backwards is rare, so it simply walks back rather than re-searching.
    let c = cursorRef.current
    while (c < timed.length && timed[c].onset <= t) c++
    while (c > 0 && timed[c - 1].onset > t) c--
    cursorRef.current = c
    const sounding = []
    // Notes are sorted by onset, not by offset, so a long note can still be
    // sounding well behind the cursor.  Looking back over a bounded window is
    // enough in practice and keeps this O(1) per frame; 512 covers any
    // plausible pile-up of simultaneous onsets under a held chord.
    for (let i = Math.max(0, c - 512); i < c; i++) {
      const n = timed[i]
      if (t < n.offset) sounding.push({ pitch: n.pitch, onset: n.onset })
    }
    // heldKeys applies the reach model: each hand is at most five fingers
    // inside a ~14-semitone window, and the two hands do not interleave.
    // Anything sounding that no hand could be on is still ringing under the
    // pedal — it is just not a pressed key, so it is not drawn as one.
    return heldKeys(sounding)
  }, [timed])

  // Same keys held, same onsets => the keyboard would paint identically, so
  // there is nothing to re-render.  During a sustained chord this is every
  // frame, which is why the check is worth making.
  const sameKeys = (a, b) => {
    if (a.size !== b.size) return false
    for (const [p, on] of a) if (b.get(p) !== on) return false
    return true
  }

  const syncKeys = useCallback((t) => {
    const next = keysAt(t)
    setActiveKeys(prev => (sameKeys(prev, next) ? prev : next))
  }, [keysAt])

  // Re-resolve when the note set or the part filter changes, not on the clock.
  useEffect(() => { cursorRef.current = 0; syncKeys(timeRef.current) }, [syncKeys])

  // Click-to-hear.  Only when stopped: while the arrangement plays, clicks on
  // the roll are seeks, and a synth note over the recording would muddy the
  // very thing the user is trying to judge.
  const strikeKey = useCallback((pitch) => {
    if (playing) return
    playNote(pitch, { velocity: 80 })
  }, [playing])

  // ── Audio events ────────────────────────────────────────────────────────────
  useEffect(() => {
    const audio = audioRef.current
    if (!audio) return
    // Only needed while paused/seeking; during playback the rAF loop above
    // samples at 60 Hz and this would just duplicate its work.
    const onTime    = () => {
      if (!isDragging.current && audio.paused) setCurrentTime(audio.currentTime)
    }
    const onDur     = () => setDuration(audio.duration)
    const onEnded   = () => { setPlaying(false); setCurrentTime(0) }
    const onCanPlay = () => setLoading(false)
    const onWait    = () => setLoading(true)
    audio.addEventListener('timeupdate',     onTime)
    audio.addEventListener('durationchange', onDur)
    audio.addEventListener('ended',          onEnded)
    audio.addEventListener('canplay',        onCanPlay)
    audio.addEventListener('waiting',        onWait)
    return () => {
      audio.removeEventListener('timeupdate',     onTime)
      audio.removeEventListener('durationchange', onDur)
      audio.removeEventListener('ended',          onEnded)
      audio.removeEventListener('canplay',        onCanPlay)
      audio.removeEventListener('waiting',        onWait)
      audio.pause()
    }
  }, [])

  // Live tone shaping over the playing audio.  Attached lazily on the first
  // real play attempt rather than on mount: createMediaElementSource re-routes
  // the element permanently, and browsers will not start an AudioContext
  // before a user gesture anyway.
  useEffect(() => {
    const audio = audioRef.current
    if (!audio || !tone) return
    attachTone(audio)
    setTone(audio, tone)
  }, [tone])

  // Reload audio when the SOURCE actually changes.
  //
  // This effect used to depend on `getPartUrl`, which every caller passes as
  // an inline arrow — a fresh function identity on every parent render.  So
  // the effect re-ran constantly and each run did
  // `audio.pause(); setCurrentTime(0)`, snapping playback back to the start.
  // That is why seeking appeared to restart the track: the seek itself worked
  // fine, then a re-render threw it away.  Depending on the resolved URL
  // string instead means the reload happens when, and only when, the audio
  // being played is genuinely different.
  const partUrl = getPartUrl(activePart) || ''
  useEffect(() => {
    const audio = audioRef.current
    if (!audio || !partUrl) return
    const wasPlaying = !audio.paused
    audio.pause(); setPlaying(false); setCurrentTime(0); setDuration(0); setLoading(true)
    audio.src = partUrl
    audio.load()
    if (wasPlaying) {
      audio.play()
        .then(() => { setPlaying(true); setLoading(false) })
        .catch(() => setLoading(false))
    }
  }, [partUrl])

  // ── rAF auto-scroll (vertical) ──────────────────────────────────────────────
  // Scroll so the current time sits at the bottom edge of the roll viewport
  // (the top edge of the keyboard = the "now" indicator).
  useEffect(() => {
    if (!playing) {
      if (rafRef.current) { cancelAnimationFrame(rafRef.current); rafRef.current = null }
      return
    }
    let lastClock = -1
    function tick() {
      const audio = audioRef.current
      const roll  = rollRef.current
      if (audio && roll && !isDragging.current) {
        // Sample the clock every FRAME, not every `timeupdate`.
        //
        // timeupdate fires roughly 4 times a second in Chrome, so the
        // keyboard's active-key set was being resolved at 4 Hz.  Two notes
        // 150 ms apart fell inside one sample and read as a single held
        // press — which is exactly why fast repeated notes looked wrong while
        // the audio was correct: the audio element plays continuously, only
        // the visual sampling was coarse.  At 60 Hz a re-strike as short as
        // ~17 ms apart now resolves.
        const t = audio.currentTime
        timeRef.current = t

        // Pin the now-line to the bottom edge of the viewport (= top of the
        // keyboard).  Future notes sit above; as time advances the stage
        // "falls" — the offset decreases and notes glide down into the keys.
        applyScroll((maxTime - t) * PX_PER_S + pad - roll.clientHeight)

        // The keyboard is the only part of the tree the clock can change, and
        // only when a note actually starts or stops.
        syncKeys(t)

        // Readout and progress bar written straight to the DOM.  They are two
        // text nodes and one width; routing them through state would drag the
        // entire note layer along for the ride.
        const whole = Math.floor(t)
        if (whole !== lastClock) {
          lastClock = whole
          if (clockElRef.current) clockElRef.current.textContent = fmt(t)
        }
        const d = audio.duration
        if (barElRef.current && d > 0) {
          barElRef.current.style.width = `${(t / d) * 100}%`
        }
      }
      rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)
    return () => {
      if (rafRef.current) { cancelAnimationFrame(rafRef.current); rafRef.current = null }
      // Hand the clock back to React on pause, so the readout and the paused
      // pinning effect resume from where playback actually stopped.
      setCurrentTime(timeRef.current)
    }
  }, [playing, maxTime, pad, applyScroll, syncKeys])

  // ── Transport ───────────────────────────────────────────────────────────────
  const togglePlay = useCallback(() => {
    const audio = audioRef.current
    if (!audio) return
    if (playing) { audio.pause(); setPlaying(false) }
    else {
      // Inside the click handler, which is the only place a browser will let
      // an AudioContext start.
      if (tone) { attachTone(audio); setTone(audio, tone) }
      setLoading(true)
      audio.play()
        .then(() => { setPlaying(true); setLoading(false) })
        .catch(() => setLoading(false))
    }
  }, [playing, tone])

  // ── Seek (y-based: click = time position) ──────────────────────────────────
  // With falling notes, smaller SVG-y ⇒ later time.  Click higher in viewport
  // ⇒ jump further into the future.
  const seekToY = useCallback((clientY) => {
    const c     = rollRef.current
    const audio = audioRef.current
    if (!c || !audio) return
    const rect = c.getBoundingClientRect()
    const svgY = (clientY - rect.top) + scrollRef.current
    const t    = Math.max(0, Math.min(maxTime - (svgY - pad) / PX_PER_S, duration || maxTime))
    audio.currentTime = t
    timeRef.current = t
    setCurrentTime(t)
    applyScroll(((maxTime - t) * PX_PER_S + pad) - c.clientHeight)
  }, [duration, maxTime, pad, applyScroll])

  const onMouseDown = useCallback((e) => {
    if (e.button !== 0) return
    isDragging.current = true; setDragging(true); seekToY(e.clientY)
  }, [seekToY])
  const onMouseMove = useCallback((e) => {
    if (!isDragging.current) return; seekToY(e.clientY)
  }, [seekToY])
  const onMouseUp = useCallback(() => {
    isDragging.current = false; setDragging(false)
  }, [])

  useEffect(() => {
    if (!dragging) return
    window.addEventListener('mousemove', onMouseMove)
    window.addEventListener('mouseup',   onMouseUp)
    return () => {
      window.removeEventListener('mousemove', onMouseMove)
      window.removeEventListener('mouseup',   onMouseUp)
    }
  }, [dragging, onMouseMove, onMouseUp])

  const onTouchStart = useCallback((e) => seekToY(e.touches[0].clientY), [seekToY])
  const onTouchMove  = useCallback((e) => {
    e.preventDefault(); seekToY(e.touches[0].clientY)
  }, [seekToY])

  // Progress-bar seek (transport-style horizontal scrub).  Sets audio time
  // from the click's horizontal fraction across the bar and snaps the roll's
  // scroll position to match, so the piano-roll content jumps to the new
  // playhead too.  No-op until duration is known.
  const seekByBar = useCallback((e) => {
    const audio = audioRef.current
    if (!audio || !(duration > 0)) return
    const bar = e.currentTarget.getBoundingClientRect()
    const fraction = Math.max(0, Math.min(1, (e.clientX - bar.left) / bar.width))
    const t = fraction * duration
    audio.currentTime = t
    timeRef.current = t
    setCurrentTime(t)
    if (rollRef.current) {
      applyScroll((maxTime - t) * PX_PER_S + pad - rollRef.current.clientHeight)
    }
  }, [duration, maxTime, pad, applyScroll])

  // When NOT playing (initial mount, paused, end-of-track), keep the roll
  // pinned to `currentTime`.  Without this the roll would snap to scrollTop=0
  // (= the future-empty padding) on first render and the song would appear
  // "above" the visible area.
  useEffect(() => {
    const roll = rollRef.current
    if (!roll || playing) return
    timeRef.current = currentTime
    applyScroll((maxTime - currentTime) * PX_PER_S + pad - roll.clientHeight)
    syncKeys(currentTime)
  }, [currentTime, playing, maxTime, pad, notes, applyScroll, syncKeys])

  // Read the clock from the ref while playing.  A render triggered by
  // something else (a key lighting up) would otherwise repaint these two from
  // the last state value and visibly jerk the readout and the bar backwards
  // until the next frame corrected them.
  const shownTime = playing ? timeRef.current : currentTime
  const pct = duration > 0 ? (shownTime / duration) * 100 : 0

  // anime.js: reveal the roll with a bottom-up clip wipe on first mount.
  // One clip-path animation on the container — cheap even with thousands
  // of SVG note rects inside.
  const revealed = useRef(false)
  useEffect(() => {
    if (!revealed.current && rollRef.current && notes.length) {
      revealed.current = true
      wipeReveal(rollRef.current, { duration: 1100, delay: 150 })
    }
  }, [notes])

  // ── Memoised SVG layers ────────────────────────────────────────────────────
  // The roll's contents do not depend on the clock — a note bar is at the same
  // place in the stage whether or not it is currently sounding, because the
  // playhead moves the stage rather than the notes.  So these are built when
  // the music or the geometry changes and reused otherwise.  Without this, a
  // key lighting up rebuilt every note and every beat line to paint one key.
  const beatGrid = useMemo(() => Array.from({ length: beatCount + 1 }, (_, b) => {
    const y          = yOf(b * beatDur)
    const isMeasure  = b % 4 === 0
    const measureNum = Math.floor(b / 4) + 1
    return (
      <g key={b}>
        {/* Beat line across full roll */}
        <line x1={TIMELINE_W} y1={y} x2={rollSvgW} y2={y}
          stroke={isMeasure ? 'rgba(255,255,255,0.09)' : 'rgba(255,255,255,0.030)'}
          strokeWidth={isMeasure ? 1 : 0.5} />
        {/* Tick into timeline strip */}
        <line
          x1={isMeasure ? TIMELINE_W * 0.30 : TIMELINE_W * 0.58}
          y1={y}
          x2={TIMELINE_W - 1} y2={y}
          stroke={isMeasure ? 'rgba(245,158,11,0.55)' : 'rgba(255,255,255,0.18)'}
          strokeWidth={isMeasure ? 1.5 : 0.75} />
        {/* Measure number — placed below the line in viewport */}
        {isMeasure && (
          <text x={TIMELINE_W / 2} y={y + 9}
            fontSize={8} fill="rgba(245,158,11,0.70)"
            fontFamily="ui-monospace,monospace" fontWeight="700"
            textAnchor="middle">
            {measureNum}
          </text>
        )}
      </g>
    )
  }), [beatCount, beatDur, maxTime, pad, rollSvgW])

  const noteLayer = useMemo(() => notes.map((n, i) => {
    const p = parseInt(n.pitch, 10)
    if (p < midiLo || p > midiHi) return null

    const pos      = pitchPos(p, midiLo, noteW)
    const onsetT   = parseFloat(n.onset)
    const offsetT  = (n.offset ?? onsetT + 0.25)
    const yTop     = yOf(offsetT)
    const yBot     = yOf(onsetT)
    const h        = Math.max(4, yBot - yTop)
    const reg      = classify(p)
    const inPart   = activePart === 'all' || (p >= partLo && p <= partHi)

    // Velocity drives brightness.  The model's velocity head is
    // now calibrated to the true dynamic range, so this is real
    // expression rather than decoration: a quiet note should LOOK
    // quiet, and the roll should show phrasing at a glance.
    const vel = Math.max(1, Math.min(127, parseInt(n.velocity ?? 80, 10)))
    const v   = vel / 127
    const w   = Math.max(2, pos.w - 2)
    const x   = TIMELINE_W + pos.x + 1

    return (
      <g key={i} opacity={inPart ? 1 : 0.10}>
        <rect
          x={x} y={yTop} width={w} height={h}
          rx={Math.min(3, w / 2)}
          fill={`url(#note-${reg})`}
          opacity={0.42 + v * 0.5}
        />
        {/* Attack edge.  The bar's BOTTOM is the onset — the edge
            that meets the keyboard — so a bright cap there reads as
            the strike, and makes two repeated notes legible as two
            events rather than one long bar. */}
        {inPart && h > 5 && (
          <rect
            x={x} y={Math.max(yTop, yBot - 2.5)}
            width={w} height={2.5}
            rx={1.2}
            fill={REG_COLOR[reg].fill}
            opacity={0.55 + v * 0.45}
          />
        )}
      </g>
    )
  }), [notes, midiLo, midiHi, noteW, activePart, partLo, partHi, maxTime, pad])

  // ── LH/RH divider x position ───────────────────────────────────────────────
  const handDivX = (midiLo <= 60 && 60 <= midiHi)
    ? TIMELINE_W + (60 - midiLo) * noteW
    : null

  // ── Render ──────────────────────────────────────────────────────────────────
  return (
    <div className="space-y-2">

      {/* Transport row */}
      <div className="flex items-center gap-3 px-1 flex-wrap">
        <span className="text-[10px] font-bold uppercase tracking-widest flex-shrink-0"
          style={{ color: '#f59e0b', letterSpacing: '0.14em' }}>
          {label}
        </span>

        <button onClick={togglePlay} disabled={loading && !playing}
          className="w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 active:scale-90 transition-all duration-150 disabled:opacity-40"
          style={{
            background: 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)',
            boxShadow: playing
              ? '0 0 10px rgba(245,158,11,0.45), 0 2px 8px rgba(0,0,0,0.4)'
              : '0 2px 8px rgba(0,0,0,0.4)',
          }}>
          <AnimatePresence mode="wait">
            {loading && !playing ? (
              <motion.span key="spin" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                className="w-3 h-3 border-2 border-white border-t-transparent rounded-full animate-spin" />
            ) : playing ? (
              <motion.svg key="pause" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                className="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 24 24">
                <rect x="6" y="4" width="4" height="16" rx="1" /><rect x="14" y="4" width="4" height="16" rx="1" />
              </motion.svg>
            ) : (
              <motion.svg key="play" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                className="w-3 h-3 text-white ml-0.5" fill="currentColor" viewBox="0 0 24 24">
                <path d="M8 5v14l11-7z" />
              </motion.svg>
            )}
          </AnimatePresence>
        </button>

        <span className="text-[11px] tabular-nums text-zinc-400 flex-shrink-0">
          <span ref={clockElRef}>{fmt(shownTime)}</span>
          <span className="text-zinc-700"> / </span>{fmt(duration)}
        </span>

        <div
          onClick={seekByBar}
          className="flex-1 rounded-full min-w-[40px] cursor-pointer"
          title="Click to seek"
          style={{
            background: 'rgba(255,255,255,0.07)',
            // Bigger hit-target (12 px) wrapping a visually-thin (2 px) bar so
            // the click area is generous without changing the layout's look.
            height: 12,
            display: 'flex',
            alignItems: 'center',
            padding: '0 1px',
          }}
        >
          <div className="h-0.5 w-full rounded-full overflow-hidden"
               style={{ background: 'rgba(255,255,255,0.04)' }}>
            <div ref={barElRef} className="h-full rounded-full transition-none" style={{
              width: `${pct}%`,
              background: 'linear-gradient(90deg, #f59e0b, #d97706)',
              boxShadow: pct > 0 ? '0 0 6px rgba(245,158,11,0.4)' : 'none',
            }} />
          </div>
        </div>
      </div>

      {/* Part selector */}
      <div className="flex gap-1.5 px-1 flex-wrap">
        {PARTS.map(({ key, label: pLabel, color }) => {
          const isActive = activePart === key
          return (
            <button key={key} onClick={() => setActivePart(key)}
              className="px-2.5 py-1 rounded-full text-[10px] font-semibold tracking-wide transition-all duration-150 active:scale-95"
              style={{
                background: isActive ? `${color}22` : 'rgba(255,255,255,0.04)',
                border: `1px solid ${isActive ? `${color}55` : 'rgba(255,255,255,0.08)'}`,
                color:  isActive ? color : '#71717a',
                boxShadow: isActive ? `0 0 8px ${color}30` : 'none',
              }}>
              {pLabel}
            </button>
          )
        })}
        <span className="text-[9px] text-zinc-700 ml-auto self-center pr-1">
          {keyboardOctaves} oct · click to seek
        </span>
      </div>

      {/* ── Roll + Keyboard ── */}
      {/* wrapperRef measures the holder so noteW fits the available width.    */}
      {/* The flex column centers the SVGs when the keyboard is narrower than  */}
      {/* the holder (very wide screens, capped at NOTE_W_MAX).                */}
      <div
        ref={wrapperRef}
        className="rounded-xl overflow-hidden select-none flex flex-col items-center"
        style={{
          // OPAQUE, not translucent.  This was rgba(0,0,0,0.55) over a moving
          // backdrop, so the waveform field showed through the roll and
          // crossed the note bars — two things animating in the same pixels,
          // and the roll is the one that carries information.  Giving it a
          // solid surface removes the conflict at the source instead of
          // dimming the backdrop until it no longer reads.
          background: 'linear-gradient(180deg, #0b0b0f 0%, #08080b 100%)',
          // No outline.  This was the last hard rectangle on the results
          // page, and a box around the roll makes it read as a widget placed
          // on the page rather than the page's subject.  While dragging to
          // seek, a soft accent glow says "you are steering this" without
          // reinstating the frame.
          border: 'none',
          // One boxShadow, not two.  This object declared the key twice — the
          // static shadow silently overwrote the drag-state one, so the accent
          // glow never rendered and the `dragging` branch was dead code.
          boxShadow: dragging
            ? 'inset 0 1px 0 rgba(255,255,255,0.04), 0 0 0 1px rgba(245,158,11,0.22), 0 0 40px rgba(245,158,11,0.07)'
            : 'inset 0 1px 0 rgba(255,255,255,0.04), 0 4px 18px rgba(0,0,0,0.35)',
          transition: 'box-shadow 0.15s ease',
          overflowX: 'auto',
        }}
      >
        {/* Scrollable roll */}
        <div
          ref={rollRef}
          style={{
            // The viewport CLIPS; it no longer scrolls.  Motion is a transform
            // on the stage inside it, so the SVG rasterises once and the
            // browser only recomposites — see applyScroll.
            overflow: 'hidden',
            height: rollHeight,
            cursor: 'row-resize',
            width: rollSvgW,
            // Isolate this subtree: whatever changes inside cannot invalidate
            // layout or paint for the rest of the page, which matters because
            // the results page mounts four of these at once.
            contain: 'strict',
          }}
          onMouseDown={onMouseDown}
          onTouchStart={onTouchStart}
          onTouchMove={onTouchMove}
        >
         <div
           ref={stageRef}
           style={{
             // Promoted to its own compositor layer up front.  Without this
             // the first frame of playback pays for the promotion, which is
             // exactly the moment the eye is watching.
             willChange: 'transform',
             transform: 'translate3d(0,0,0)',
           }}
         >
          <svg width={rollSvgW} height={rollSvgH} style={{ display: 'block' }}>

            {/* ── Timeline strip background ── */}
            <rect x={0} y={0} width={TIMELINE_W} height={rollSvgH}
              fill="rgba(0,0,0,0.30)" />

            {/* Roll-side gradient defs — barrel shading for note bars  */}
            <defs>
              {Object.entries(REG_COLOR).map(([reg, { fill }]) => (
                <linearGradient key={reg} id={`note-${reg}`} x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%"   stopColor={fill} stopOpacity="0.55" />
                  <stop offset="35%"  stopColor={fill} stopOpacity="1"    />
                  <stop offset="65%"  stopColor={fill} stopOpacity="1"    />
                  <stop offset="100%" stopColor={fill} stopOpacity="0.55" />
                </linearGradient>
              ))}
            </defs>

            {/* ── Black-key column shading — uses diatonic positions ── */}
            {Array.from({ length: pitchRange }, (_, i) => {
              const pitch = midiLo + i
              if (!isBlack(pitch)) return null
              const pos = pitchPos(pitch, midiLo, noteW)
              return (
                <rect key={pitch}
                  x={TIMELINE_W + pos.x} y={0}
                  width={pos.w} height={rollSvgH}
                  fill="rgba(0,0,0,0.22)" />
              )
            })}

            {/* ── C-note vertical grid lines (octave starts) ── */}
            {Array.from({ length: pitchRange }, (_, i) => {
              const pitch = midiLo + i
              if (pitch % 12 !== 0) return null
              const pos = pitchPos(pitch, midiLo, noteW)
              const x   = TIMELINE_W + pos.x
              return (
                <line key={pitch}
                  x1={x} y1={0} x2={x} y2={rollSvgH}
                  stroke="rgba(255,255,255,0.07)" strokeWidth={1} />
              )
            })}

            {/* ── LH / RH zone tints + vertical divider ── */}
            {/*    Text labels removed — they were anchored to fixed SVG y    */}
            {/*    and would only ever be visible at the song boundaries      */}
            {/*    once the time axis was inverted.                           */}
            {handDivX !== null && (() => {
              const lhW = handDivX - TIMELINE_W
              const rhW = rollSvgW - handDivX
              return (
                <g>
                  <rect x={TIMELINE_W} y={0} width={lhW} height={rollSvgH}
                    fill="rgba(59,130,246,0.025)" />
                  <rect x={handDivX} y={0} width={rhW} height={rollSvgH}
                    fill="rgba(248,113,113,0.025)" />
                  <line x1={handDivX} y1={0} x2={handDivX} y2={rollSvgH}
                    stroke="rgba(245,158,11,0.35)" strokeWidth={1}
                    strokeDasharray="6 4" />
                </g>
              )
            })()}

            {/* ── Beat / measure horizontal grid ── */}
            {beatGrid}

            {/* Octave banding: a faint stripe on every C..B group so pitch
                is readable at a glance instead of by counting keys.  Drawn
                behind the notes, above the hand tint. */}
            {Array.from({ length: Math.ceil((midiHi - midiLo + 1) / 12) }, (_, o) => {
              const lo = midiLo + o * 12
              const a = pitchPos(lo, midiLo, noteW)
              return o % 2 === 0 ? null : (
                <rect key={`oct${o}`}
                  x={TIMELINE_W + a.x} y={0}
                  width={noteW * 12} height={rollSvgH}
                  fill="rgba(255,255,255,0.014)"
                />
              )
            })}

            {/* ── Notes (vertical bars) ─────────────────────────────────── */}
            {/*    Bar's BOTTOM edge sits at the onset — it's the leading    */}
            {/*    edge that falls into the keyboard.  Bar's TOP is the      */}
            {/*    offset.  X position and width come from `pitchPos`, so    */}
            {/*    white-pitch bars are wide (= white-key bottom width) and  */}
            {/*    black-pitch bars are narrow (= black-key width) — they    */}
            {/*    line up with the keys they fall onto.                     */}
            {noteLayer}

          </svg>
         </div>
        </div>

        {/* Fixed keyboard at the bottom */}
        <KeyboardSVG
          midiLo={midiLo}
          midiHi={midiHi}
          noteW={noteW}
          activeKeys={activeKeys}
          interactive={!playing}
          onKeyDown={strikeKey}
        />
      </div>

      {/* Legend */}
      <div className="flex gap-4 px-1">
        {Object.entries(REG_COLOR).map(([name, { fill }]) => (
          <div key={name} className="flex items-center gap-1.5">
            <div className="w-2.5 h-2.5 rounded-sm" style={{ background: fill }} />
            <span className="text-[10px] text-zinc-600 capitalize">{name}</span>
          </div>
        ))}
      </div>

      <audio ref={audioRef} preload="metadata" />
    </div>
  )
}
