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
// the song without the scroll going negative.
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
// Runs LEFT → RIGHT: bass (low pitch) on the left, treble (high pitch) on the right.
// Real-piano geometry: every white key has the SAME bottom width (12·noteW/7)
// and they sit edge-to-edge with no gaps along the bottom.  The TOP portion of
// each white key is narrower wherever a black key rides above it — so C/E/F/B
// have a one-sided notch and D/G/A have notches on both sides, but every key's
// bottom remains visually identical.  Black keys are short and float above
// the boundary between the two whites they belong between.
function KeyboardSVG({ midiLo, midiHi, activeKeys, noteW, interactive, onKeyDown }) {
  const pitchRange = midiHi - midiLo + 1
  const totalW     = TIMELINE_W + pitchRange * noteW
  const WHITE_H    = KEY_H - 1
  const BLACK_H    = Math.round(KEY_H * 0.62)
  const wkw        = (noteW * 12) / 7    // white-key bottom width (= octave / 7)
  const blackW     = wkw * 0.6            // black-key visual width

  // Press feel — same shape, two depths so blacks dip a touch less than whites.
  const PRESS_W = 2.5
  const PRESS_B = 1.8
  const pressTransition = 'transform 70ms cubic-bezier(0.2, 0.85, 0.3, 1.15)'

  // Build the notched white-key path.  All coordinates are in keyboard space
  // (TIMELINE_W is added at the call site). Corners on the OUTSIDE of the key
  // (the four exposed corners) get a small radius; the two inner step-out
  // corners are left sharp because they sit under the black keys and don't
  // visually need rounding.
  const wkPath = (topL, topR, botL, botR) => {
    const r = 2.5
    return [
      `M ${topL + r} 0`,
      `L ${topR - r} 0`,
      `Q ${topR} 0 ${topR} ${r}`,
      `L ${topR} ${BLACK_H}`,
      `L ${botR} ${BLACK_H}`,
      `L ${botR} ${KEY_H - r}`,
      `Q ${botR} ${KEY_H} ${botR - r} ${KEY_H}`,
      `L ${botL + r} ${KEY_H}`,
      `Q ${botL} ${KEY_H} ${botL} ${KEY_H - r}`,
      `L ${botL} ${BLACK_H}`,
      `L ${topL} ${BLACK_H}`,
      `L ${topL} ${r}`,
      `Q ${topL} 0 ${topL + r} 0`,
      'Z',
    ].join(' ')
  }

  return (
    <svg width={totalW} height={KEY_H} style={{ display: 'block', flexShrink: 0 }}>
      <defs>
        {/* Ivory: bright top → mid body → soft cool shadow at the bottom */}
        <linearGradient id="wk-grad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%"   stopColor="#ffffff" />
          <stop offset="6%"   stopColor="#fbfbfd" />
          <stop offset="55%"  stopColor="#ececf0" />
          <stop offset="100%" stopColor="#bcbcc4" />
        </linearGradient>
        {/* Ebony: glossy top sheen, deep matte body, soft reflective rim */}
        <linearGradient id="bk-grad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%"   stopColor="#3a3a42" />
          <stop offset="10%"  stopColor="#1c1c20" />
          <stop offset="55%"  stopColor="#0a0a0c" />
          <stop offset="100%" stopColor="#000000" />
        </linearGradient>
        {/* Active-key gradients (one per register, full saturation top → softer bottom) */}
        {Object.entries(REG_COLOR).map(([reg, { fill }]) => (
          <linearGradient key={reg} id={`active-${reg}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%"   stopColor={fill} stopOpacity="1"    />
            <stop offset="55%"  stopColor={fill} stopOpacity="0.92" />
            <stop offset="100%" stopColor={fill} stopOpacity="0.65" />
          </linearGradient>
        ))}
      </defs>

      {/* Timeline-strip patch */}
      <rect x={0} y={0} width={TIMELINE_W} height={KEY_H} fill="rgba(0,0,0,0.45)" />
      <text x={TIMELINE_W / 2} y={KEY_H / 2}
        fontSize={6} fill="rgba(245,158,11,0.45)"
        fontFamily="ui-monospace,monospace" fontWeight="700"
        textAnchor="middle" dominantBaseline="middle" letterSpacing="0.5">
        KEY
      </text>

      {/* ── White keys (bottom layer) — uniform bottom, notched top ── */}
      {Array.from({ length: pitchRange }, (_, i) => {
        const pitch = midiLo + i
        if (isBlack(pitch)) return null
        const pos = pitchPos(pitch, midiLo, noteW)
        const botL = TIMELINE_W + pos.x
        const botR = botL + pos.w                        // bottom span = wkw

        // Top span: pinched in by half a black-key width on whichever sides
        // have an adjacent black neighbour.
        const pc      = ((pitch % 12) + 12) % 12
        const leftPc  = (pc + 11) % 12
        const rightPc = (pc + 1) % 12
        const topL = (leftPc  in BLACK_BOUNDARY) ? botL + blackW / 2 : botL
        const topR = (rightPc in BLACK_BOUNDARY) ? botR - blackW / 2 : botR

        const active = activeKeys.has(pitch)
        const reg    = classify(pitch)
        const isC    = pitch % 12 === 0
        const octave = Math.floor(pitch / 12) - 1
        return (
          <g
            key={pitch}
            onMouseDown={interactive ? (e) => { e.stopPropagation(); onKeyDown?.(pitch) } : undefined}
            style={{
              transform: active ? `translateY(${PRESS_W}px)` : 'translateY(0px)',
              transition: pressTransition,
              cursor: interactive ? 'pointer' : 'default',
            }}
          >
            {/* Body — notched path */}
            <path
              d={wkPath(topL, topR, botL, botR)}
              fill={active ? `url(#active-${reg})` : 'url(#wk-grad)'}
              stroke={active ? 'none' : 'rgba(0,0,0,0.22)'}
              strokeWidth={0.6}
              style={active ? {
                filter: [
                  `drop-shadow(0 0 6px ${REG_COLOR[reg].glow})`,
                  `drop-shadow(0 0 14px ${REG_COLOR[reg].glow})`,
                  `drop-shadow(0 0 26px ${REG_COLOR[reg].glow})`,
                ].join(' '),
              } : undefined}
            />
            {/* Re-strike flash.  Keyed on the ONSET so a repeated note mounts
                a NEW element, and a CSS keyframe animation restarts on mount
                (a plain transition would not — a remounted element is painted
                straight into its final state with nothing to animate from,
                which is exactly why two consecutive notes used to look like
                one long press). */}
            {active && (
              <path
                key={activeKeys.get(pitch)}
                className="key-strike"
                d={wkPath(topL, topR, botL, botR)}
                fill="#ffffff"
              />
            )}
            {/* Top highlight stripe — 3D ivory sheen, sized to top span */}
            {!active && (
              <rect
                x={topL + 1} y={1.5}
                width={Math.max(1, topR - topL - 2)} height={2}
                rx={1}
                fill="rgba(255,255,255,0.95)"
                style={{ pointerEvents: 'none' }}
              />
            )}
            {/* Inner bottom shadow — across the wider bottom span */}
            {!active && (
              <rect
                x={botL + 0.5} y={KEY_H - 4}
                width={Math.max(1, botR - botL - 1)} height={3}
                rx={1.5}
                fill="rgba(0,0,0,0.10)"
                style={{ pointerEvents: 'none' }}
              />
            )}
            {isC && !active && (
              <text
                x={(botL + botR) / 2} y={KEY_H - 5}
                fontSize={5} fill="rgba(80,80,90,0.75)"
                fontFamily="ui-monospace,monospace" fontWeight="700"
                textAnchor="middle">
                C{octave}
              </text>
            )}
          </g>
        )
      })}

      {/* ── Black keys (top layer) — float over the white-key boundary ── */}
      {Array.from({ length: pitchRange }, (_, i) => {
        const pitch  = midiLo + i
        if (!isBlack(pitch)) return null
        const pos    = pitchPos(pitch, midiLo, noteW)
        const xL     = TIMELINE_W + pos.x
        const w      = pos.w
        const active = activeKeys.has(pitch)
        const reg    = classify(pitch)
        return (
          <g
            key={pitch}
            onMouseDown={interactive ? (e) => { e.stopPropagation(); onKeyDown?.(pitch) } : undefined}
            style={{
              transform: active ? `translateY(${PRESS_B}px)` : 'translateY(0px)',
              transition: pressTransition,
              cursor: interactive ? 'pointer' : 'default',
            }}
          >
            <rect
              x={xL} y={0}
              width={w} height={BLACK_H}
              rx={2}
              fill={active ? `url(#active-${reg})` : 'url(#bk-grad)'}
              style={active ? {
                filter: [
                  `drop-shadow(0 0 6px ${REG_COLOR[reg].glow})`,
                  `drop-shadow(0 0 14px ${REG_COLOR[reg].glow})`,
                  `drop-shadow(0 0 26px ${REG_COLOR[reg].glow})`,
                ].join(' '),
              } : undefined}
            />
            {active && (
              <rect
                key={activeKeys.get(pitch)}
                className="key-strike"
                x={xL} y={0} width={w} height={BLACK_H} rx={2}
                fill="#ffffff"
              />
            )}
            {/* Glossy top sheen */}
            {!active && (
              <rect
                x={xL + 0.6} y={1}
                width={Math.max(0.5, w - 1.2)} height={1.5}
                rx={0.8}
                fill="rgba(255,255,255,0.22)"
                style={{ pointerEvents: 'none' }}
              />
            )}
            {/* Subtle reflective rim near the bottom */}
            {!active && (
              <rect
                x={xL + 0.6} y={BLACK_H - 4}
                width={Math.max(0.5, w - 1.2)} height={1}
                rx={0.5}
                fill="rgba(255,255,255,0.07)"
                style={{ pointerEvents: 'none' }}
              />
            )}
          </g>
        )
      })}

      {/* ── Middle-C dashed divider only — no harsh top separator ── */}
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
  const rollRef    = useRef(null)   // scrollable roll container
  const wrapperRef = useRef(null)   // outer wrapper — measured for fit-to-width
  const isDragging = useRef(false)
  const rafRef     = useRef(null)

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
  //   • y = VIEWPORT_PAD is t = maxTime
  //   • y grows downward as time goes earlier — so notes with smaller onsets
  //     end up nearer the bottom, ready to drop into the keyboard.
  // The +1.5 s of past-padding at the bottom keeps the view from clipping the
  // very first downbeat when the user pauses at t = 0.
  const rollSvgH   = Math.max(800, (maxTime + 1.5) * PX_PER_S + VIEWPORT_PAD)
  const yOf        = (t) => (maxTime - t) * PX_PER_S + VIEWPORT_PAD

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
  const activeKeys = useMemo(() => {
    if (currentTime <= 0) return new Map()
    const sounding = []
    for (const n of notes) {
      const p = parseInt(n.pitch, 10)
      if (p < midiLo || p > midiHi) continue
      const inPart = activePart === 'all' || (p >= partLo && p <= partHi)
      if (inPart && currentTime >= n.onset && currentTime < (n.offset ?? n.onset + 0.25)) {
        sounding.push({ pitch: p, onset: n.onset })
      }
    }
    // heldKeys applies the reach model: each hand is at most five fingers
    // inside a ~14-semitone window, and the two hands do not interleave.
    // Anything sounding that no hand could be on is still ringing under the
    // pedal — it is just not a pressed key, so it is not drawn as one.
    return heldKeys(sounding)
  }, [notes, currentTime, activePart, partLo, partHi, midiLo, midiHi])

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
    function tick() {
      const audio = audioRef.current
      const roll  = rollRef.current
      if (audio && !isDragging.current) {
        // Sample the clock every FRAME, not every `timeupdate`.
        //
        // timeupdate fires roughly 4 times a second in Chrome, so the
        // keyboard's active-key set was being resolved at 4 Hz.  Two notes
        // 150 ms apart fell inside one sample and read as a single held
        // press — which is exactly why fast repeated notes looked wrong while
        // the audio was correct: the audio element plays continuously, only
        // the visual sampling was coarse.  At 60 Hz a re-strike as short as
        // ~17 ms apart now resolves.
        setCurrentTime(audio.currentTime)
      }
      if (audio && roll && !isDragging.current) {
        // Pin the now-line to the bottom edge of the viewport (= top of the
        // keyboard).  Future notes sit above; as time advances the SVG
        // "falls" — scrollTop decreases and notes glide down into the keys.
        const target = (maxTime - audio.currentTime) * PX_PER_S + VIEWPORT_PAD - roll.clientHeight
        roll.scrollTop = Math.max(0, target)
      }
      rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)
    return () => { if (rafRef.current) { cancelAnimationFrame(rafRef.current); rafRef.current = null } }
  }, [playing, maxTime])

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
    const svgY = (clientY - rect.top) + c.scrollTop
    const t    = Math.max(0, Math.min(maxTime - (svgY - VIEWPORT_PAD) / PX_PER_S, duration || maxTime))
    audio.currentTime = t
    setCurrentTime(t)
    c.scrollTop = Math.max(0, ((maxTime - t) * PX_PER_S + VIEWPORT_PAD) - c.clientHeight)
  }, [duration, maxTime])

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
    setCurrentTime(t)
    if (rollRef.current) {
      const target = (maxTime - t) * PX_PER_S + VIEWPORT_PAD - rollRef.current.clientHeight
      rollRef.current.scrollTop = Math.max(0, target)
    }
  }, [duration, maxTime])

  // When NOT playing (initial mount, paused, end-of-track), keep the roll
  // pinned to `currentTime`.  Without this the roll would snap to scrollTop=0
  // (= the future-empty padding) on first render and the song would appear
  // "above" the visible area.
  useEffect(() => {
    const roll = rollRef.current
    if (!roll || playing) return
    const target = (maxTime - currentTime) * PX_PER_S + VIEWPORT_PAD - roll.clientHeight
    roll.scrollTop = Math.max(0, target)
  }, [currentTime, playing, maxTime, notes])

  const pct = duration > 0 ? (currentTime / duration) * 100 : 0

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
          {fmt(currentTime)} <span className="text-zinc-700">/</span> {fmt(duration)}
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
            <div className="h-full rounded-full transition-none" style={{
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
            overflowY: 'auto',
            overflowX: 'hidden',
            maxHeight: rollHeight,
            cursor: 'row-resize',
            width: rollSvgW,
          }}
          onMouseDown={onMouseDown}
          onTouchStart={onTouchStart}
          onTouchMove={onTouchMove}
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
            {Array.from({ length: beatCount + 1 }, (_, b) => {
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
            })}

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
            {notes.map((n, i) => {
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
            })}

          </svg>
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
