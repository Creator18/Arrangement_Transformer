import { useState, useRef, useEffect, useCallback, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

// ── layout constants ──────────────────────────────────────────────────────────
const NOTE_W     = 8     // px per semitone column  (pitch → X axis)
const KEY_H      = 68    // keyboard panel height   (fixed at bottom)
const PX_PER_S   = 80    // px per second           (time → Y axis, downward)
const TIMELINE_W = 28    // left strip for measure / beat labels
const ROLL_MAX_H = 370   // max height of the scrollable roll viewport

// Which pitch classes are black keys
const BLACK_PCS = new Set([1, 3, 6, 8, 10])
const isBlack   = p => BLACK_PCS.has(p % 12)

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
// Exactly mirrors a real piano viewed from the player's seat.
//
// White-key width rules (for pitch P at column i):
//   • If pitch P-1 is black (left neighbour) → extend left  edge by NOTE_W/2
//   • If pitch P+1 is black (right neighbour) → extend right edge by NOTE_W/2
// This gives natural piano proportions:
//   C / E / F / B  → 1.5 × NOTE_W
//   D / G / A      → 2.0 × NOTE_W
//
// Black keys are top-aligned (protruding toward the roll), 60% height, 70% width.
function KeyboardSVG({ midiLo, midiHi, activePitches }) {
  const pitchRange = midiHi - midiLo + 1
  const totalW     = TIMELINE_W + pitchRange * NOTE_W
  const WHITE_H    = KEY_H - 1
  const BLACK_H    = Math.round(KEY_H * 0.60)
  const BLACK_W    = Math.round(NOTE_W * 0.70)

  function whiteKeyBounds(pitch, col) {
    const xBase = TIMELINE_W + col * NOTE_W
    let xL = xBase
    let xR = xBase + NOTE_W
    if (col > 0               && isBlack(pitch - 1)) xL = xBase - NOTE_W / 2
    if (col < pitchRange - 1  && isBlack(pitch + 1)) xR = xBase + NOTE_W * 1.5
    return { xL, xR }
  }

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
        const { xL, xR } = whiteKeyBounds(pitch, i)
        const w      = xR - xL
        const active = activePitches.has(pitch)
        const reg    = classify(pitch)
        const isC    = pitch % 12 === 0
        const octave = Math.floor(pitch / 12) - 1
        return (
          <g key={pitch}>
            <rect
              x={xL + 0.5} y={1}
              width={w - 1} height={WHITE_H - 1}
              rx={3}
              fill={active ? REG_COLOR[reg].fill : 'rgba(228,228,232,0.94)'}
              stroke={active ? 'none' : 'rgba(0,0,0,0.10)'}
              strokeWidth={0.5}
              style={active ? {
                filter: [
                  `drop-shadow(0 0 6px ${REG_COLOR[reg].glow})`,
                  `drop-shadow(0 0 12px ${REG_COLOR[reg].glow})`,
                  `drop-shadow(0 0 20px ${REG_COLOR[reg].glow})`,
                ].join(' '),
              } : undefined}
            />
            {isC && !active && (
              <text
                x={xL + w / 2} y={KEY_H - 5}
                fontSize={5} fill="rgba(80,80,90,0.70)"
                fontFamily="ui-monospace,monospace" fontWeight="700"
                textAnchor="middle">
                C{octave}
              </text>
            )}
          </g>
        )
      })}

      {/* ── Black keys (top layer — protrude toward the roll) ── */}
      {Array.from({ length: pitchRange }, (_, i) => {
        const pitch  = midiLo + i
        if (!isBlack(pitch)) return null
        const xBase  = TIMELINE_W + i * NOTE_W
        const xL     = xBase + (NOTE_W - BLACK_W) / 2
        const active = activePitches.has(pitch)
        const reg    = classify(pitch)
        return (
          <rect
            key={pitch}
            x={xL} y={0}
            width={BLACK_W} height={BLACK_H}
            rx={2}
            fill={active ? REG_COLOR[reg].fill : 'rgba(14,14,18,0.98)'}
            style={active ? {
              filter: [
                `drop-shadow(0 0 6px ${REG_COLOR[reg].glow})`,
                `drop-shadow(0 0 12px ${REG_COLOR[reg].glow})`,
                `drop-shadow(0 0 20px ${REG_COLOR[reg].glow})`,
              ].join(' '),
            } : undefined}
          />
        )
      })}

      {/* Top separator (between roll and keyboard) */}
      <line x1={0} y1={0.5} x2={totalW} y2={0.5}
        stroke="rgba(255,255,255,0.12)" strokeWidth={1} />

      {/* ── Middle-C LH / RH vertical divider ── */}
      {midiLo <= 60 && 60 <= midiHi && (() => {
        const divX = TIMELINE_W + (60 - midiLo) * NOTE_W
        return (
          <g>
            <line x1={divX} y1={0} x2={divX} y2={KEY_H}
              stroke="rgba(245,158,11,0.75)" strokeWidth={1.5} />
            <text x={divX - 5} y={KEY_H / 2}
              textAnchor="end" dominantBaseline="middle"
              fontSize={5.5} fill="rgba(245,158,11,0.85)"
              fontFamily="ui-monospace,monospace" fontWeight="700">
              LH
            </text>
            <text x={divX + 5} y={KEY_H / 2}
              textAnchor="start" dominantBaseline="middle"
              fontSize={5.5} fill="rgba(245,158,11,0.85)"
              fontFamily="ui-monospace,monospace" fontWeight="700">
              RH
            </text>
          </g>
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
}) {
  const [activePart,  setActivePart]  = useState(defaultPart)
  const [playing,     setPlaying]     = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration,    setDuration]    = useState(0)
  const [loading,     setLoading]     = useState(false)
  const [dragging,    setDragging]    = useState(false)

  const audioRef   = useRef(null)
  const rollRef    = useRef(null)   // scrollable roll container
  const isDragging = useRef(false)
  const rafRef     = useRef(null)

  // Sync when parent results-page part pad changes
  useEffect(() => { setActivePart(defaultPart) }, [defaultPart])

  // MIDI range
  const midiLo     = keyboardStart
  const midiHi     = keyboardStart + keyboardOctaves * 12 - 1
  const pitchRange  = midiHi - midiLo + 1

  // SVG dimensions
  const rollSvgW   = TIMELINE_W + pitchRange * NOTE_W

  const maxTime    = notes.length
    ? Math.max(...notes.map(n => n.offset ?? n.onset + 0.5))
    : 10
  const rollSvgH   = Math.max(600, maxTime * PX_PER_S + 60)   // +60 bottom padding

  // Beat / measure layout
  const beatDur    = 60 / tempo
  const beatCount  = Math.ceil(maxTime / beatDur)

  // Part filter
  const [partLo, partHi] = PART_RANGE[activePart]

  // ── Active pitches ──────────────────────────────────────────────────────────
  const activePitches = useMemo(() => {
    const active = new Set()
    if (currentTime <= 0) return active
    for (const n of notes) {
      const p = parseInt(n.pitch, 10)
      if (p < midiLo || p > midiHi) continue
      const inPart = activePart === 'all' || (p >= partLo && p <= partHi)
      if (inPart && currentTime >= n.onset && currentTime < (n.offset ?? n.onset + 0.25)) {
        active.add(p)
      }
    }
    return active
  }, [notes, currentTime, activePart, partLo, partHi, midiLo, midiHi])

  // ── Audio events ────────────────────────────────────────────────────────────
  useEffect(() => {
    const audio = audioRef.current
    if (!audio) return
    const onTime    = () => { if (!isDragging.current) setCurrentTime(audio.currentTime) }
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

  // Reload audio on part change
  useEffect(() => {
    const audio = audioRef.current
    if (!audio) return
    const wasPlaying = playing
    audio.pause(); setPlaying(false); setCurrentTime(0); setDuration(0); setLoading(true)
    const url = getPartUrl(activePart)
    if (url) {
      audio.src = url
      audio.load()
      if (wasPlaying) {
        audio.play()
          .then(() => { setPlaying(true); setLoading(false) })
          .catch(() => setLoading(false))
      }
    }
  }, [activePart, getPartUrl])

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
      if (audio && roll && !isDragging.current) {
        roll.scrollTop = Math.max(0, audio.currentTime * PX_PER_S - roll.clientHeight)
      }
      rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)
    return () => { if (rafRef.current) { cancelAnimationFrame(rafRef.current); rafRef.current = null } }
  }, [playing])

  // ── Transport ───────────────────────────────────────────────────────────────
  const togglePlay = useCallback(() => {
    const audio = audioRef.current
    if (!audio) return
    if (playing) { audio.pause(); setPlaying(false) }
    else {
      setLoading(true)
      audio.play()
        .then(() => { setPlaying(true); setLoading(false) })
        .catch(() => setLoading(false))
    }
  }, [playing])

  // ── Seek (y-based: click = time position) ──────────────────────────────────
  const seekToY = useCallback((clientY) => {
    const c     = rollRef.current
    const audio = audioRef.current
    if (!c || !audio) return
    const rect = c.getBoundingClientRect()
    const y    = clientY - rect.top + c.scrollTop
    const t    = Math.max(0, Math.min(y / PX_PER_S, duration || maxTime))
    audio.currentTime = t
    setCurrentTime(t)
    c.scrollTop = Math.max(0, t * PX_PER_S - c.clientHeight)
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

  const pct = duration > 0 ? (currentTime / duration) * 100 : 0

  // ── LH/RH divider x position ───────────────────────────────────────────────
  const handDivX = (midiLo <= 60 && 60 <= midiHi)
    ? TIMELINE_W + (60 - midiLo) * NOTE_W
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

        <div className="flex-1 h-0.5 rounded-full overflow-hidden min-w-[40px]"
          style={{ background: 'rgba(255,255,255,0.07)' }}>
          <div className="h-full rounded-full transition-none" style={{
            width: `${pct}%`,
            background: 'linear-gradient(90deg, #f59e0b, #d97706)',
            boxShadow: pct > 0 ? '0 0 6px rgba(245,158,11,0.4)' : 'none',
          }} />
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
      <div
        className="rounded-xl overflow-hidden select-none"
        style={{
          background: 'rgba(0,0,0,0.45)',
          border: `1px solid ${dragging ? 'rgba(245,158,11,0.35)' : 'rgba(255,255,255,0.07)'}`,
          transition: 'border-color 0.15s ease',
          overflowX: 'auto',   // horizontal scroll if viewport is narrow
        }}
      >
        {/* Scrollable roll */}
        <div
          ref={rollRef}
          style={{
            overflowY: 'auto',
            overflowX: 'hidden',
            maxHeight: ROLL_MAX_H,
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

            {/* ── Black-key column shading ── */}
            {Array.from({ length: pitchRange }, (_, i) => {
              const pitch = midiLo + i
              if (!isBlack(pitch)) return null
              return (
                <rect key={pitch}
                  x={TIMELINE_W + i * NOTE_W} y={0}
                  width={NOTE_W} height={rollSvgH}
                  fill="rgba(0,0,0,0.18)" />
              )
            })}

            {/* ── C-note vertical grid lines ── */}
            {Array.from({ length: pitchRange }, (_, i) => {
              const pitch = midiLo + i
              if (pitch % 12 !== 0) return null
              const x = TIMELINE_W + i * NOTE_W
              return (
                <line key={pitch}
                  x1={x} y1={0} x2={x} y2={rollSvgH}
                  stroke="rgba(255,255,255,0.07)" strokeWidth={1} />
              )
            })}

            {/* ── LH / RH zone tints + vertical divider ── */}
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
                  <text x={TIMELINE_W + lhW / 2} y={14}
                    textAnchor="middle" fontSize={7}
                    fill="rgba(59,130,246,0.18)"
                    fontFamily="ui-monospace,monospace" fontWeight="700"
                    letterSpacing="1.5">LEFT HAND</text>
                  <text x={handDivX + rhW / 2} y={14}
                    textAnchor="middle" fontSize={7}
                    fill="rgba(248,113,113,0.18)"
                    fontFamily="ui-monospace,monospace" fontWeight="700"
                    letterSpacing="1.5">RIGHT HAND</text>
                  <text x={handDivX + 3} y={rollSvgH - 8}
                    fontSize={6.5} fill="rgba(245,158,11,0.45)"
                    fontFamily="ui-monospace,monospace" fontWeight="700">
                    C4
                  </text>
                </g>
              )
            })()}

            {/* ── Beat / measure horizontal grid ── */}
            {Array.from({ length: beatCount + 1 }, (_, b) => {
              const y          = b * beatDur * PX_PER_S
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
                  {/* Measure number */}
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

            {/* ── Notes (vertical bars: x = pitch column, y = time) ── */}
            {notes.map((n, i) => {
              const p = parseInt(n.pitch, 10)
              if (p < midiLo || p > midiHi) return null

              const col    = p - midiLo
              const x      = TIMELINE_W + col * NOTE_W
              const y      = n.onset * PX_PER_S
              const h      = Math.max(4, ((n.offset ?? n.onset + 0.25) - n.onset) * PX_PER_S)
              const { fill } = REG_COLOR[classify(p)]

              const inPart   = activePart === 'all' || (p >= partLo && p <= partHi)
              const isActive = activePitches.has(p)

              return (
                <rect key={i}
                  x={x + 1} y={y}
                  width={NOTE_W - 2} height={h}
                  rx={2}
                  fill={fill}
                  opacity={inPart ? (isActive ? 1 : 0.55) : 0.08}
                />
              )
            })}

          </svg>
        </div>

        {/* Fixed keyboard at the bottom */}
        <KeyboardSVG
          midiLo={midiLo}
          midiHi={midiHi}
          activePitches={activePitches}
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
