import { useState, useRef, useEffect, useCallback, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

// ── layout constants ──────────────────────────────────────────────────────────
const NOTE_H    = 8      // px per semitone row
const KEY_W     = 56     // keyboard panel width
const PX_PER_S  = 80     // roll px per second
const HEADER_H  = 28     // top strip (measure markers)

// Which pitch classes are black keys
const BLACK_PCS = new Set([1, 3, 6, 8, 10])
const isBlack   = (pitch) => BLACK_PCS.has(pitch % 12)

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

// ── Keyboard SVG ──────────────────────────────────────────────────────────────
// Renders a vertical DAW-style piano keyboard aligned to the roll rows.
// White keys span full width; black keys are narrower and right-aligned
// (simulating depth). Both use rounded "Apple button" corners.
function KeyboardSVG({ midiLo, midiHi, activePitches }) {
  const pitchRange = midiHi - midiLo + 1
  const totalH     = HEADER_H + pitchRange * NOTE_H
  const WHITE_W    = KEY_W - 1          // white key width (1px gap for panel border)
  const BLACK_W    = Math.round(KEY_W * 0.55)
  const BLACK_X    = KEY_W - BLACK_W    // right-aligned so they "float" in front

  // Note-name labels on C keys
  const noteNames = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

  return (
    <svg
      width={KEY_W}
      height={totalH}
      style={{ display: 'block', flexShrink: 0 }}
    >
      {/* Header patch (aligns with roll header) */}
      <rect x={0} y={0} width={KEY_W} height={HEADER_H}
        fill="rgba(0,0,0,0.4)" />
      <text x={KEY_W / 2} y={HEADER_H * 0.68}
        fontSize={7} fill="rgba(245,158,11,0.5)"
        fontFamily="ui-monospace,monospace" fontWeight="700"
        textAnchor="middle" letterSpacing="1">
        KEY
      </text>

      {/* ── White keys (bottom layer) ── */}
      {Array.from({ length: pitchRange }, (_, i) => {
        const pitch = midiHi - i
        if (isBlack(pitch)) return null
        const y      = HEADER_H + i * NOTE_H
        const active = activePitches.has(pitch)
        const reg    = classify(pitch)
        const isC    = pitch % 12 === 0
        const octave = Math.floor(pitch / 12) - 1
        return (
          <g key={pitch}>
            <rect
              x={1} y={y + 1}
              width={WHITE_W - 1} height={NOTE_H - 2}
              rx={3}
              fill={active ? REG_COLOR[reg].fill : 'rgba(230,230,235,0.92)'}
              stroke={active ? 'none' : 'rgba(0,0,0,0.08)'}
              strokeWidth={0.5}
              style={active
                ? { filter: `drop-shadow(0 0 4px ${REG_COLOR[reg].glow}) drop-shadow(0 0 8px ${REG_COLOR[reg].glow})` }
                : undefined}
            />
            {/* C note octave label */}
            {isC && !active && (
              <text
                x={4} y={y + NOTE_H - 2}
                fontSize={5.5}
                fill="rgba(80,80,90,0.7)"
                fontFamily="ui-monospace,monospace"
                fontWeight="700"
              >
                C{octave}
              </text>
            )}
          </g>
        )
      })}

      {/* ── Black keys (top layer — rendered after so they appear above) ── */}
      {Array.from({ length: pitchRange }, (_, i) => {
        const pitch = midiHi - i
        if (!isBlack(pitch)) return null
        const y      = HEADER_H + i * NOTE_H
        const active = activePitches.has(pitch)
        const reg    = classify(pitch)
        return (
          <rect
            key={pitch}
            x={BLACK_X} y={y + 1}
            width={BLACK_W} height={NOTE_H - 2}
            rx={3}
            fill={active ? REG_COLOR[reg].fill : 'rgba(16,16,20,0.97)'}
            style={active
              ? { filter: `drop-shadow(0 0 4px ${REG_COLOR[reg].glow}) drop-shadow(0 0 8px ${REG_COLOR[reg].glow})` }
              : undefined}
          />
        )
      })}

      {/* Right-edge separator */}
      <line x1={KEY_W - 0.5} y1={HEADER_H} x2={KEY_W - 0.5} y2={totalH}
        stroke="rgba(255,255,255,0.10)" strokeWidth={1} />
    </svg>
  )
}

// ── Main component ────────────────────────────────────────────────────────────
export default function PianoRollPlayer({
  notes           = [],
  getPartUrl      = () => '',
  label           = 'Track',
  tempo           = 120,
  keyboardStart   = 36,   // lowest MIDI note shown (C2 = 36)
  keyboardOctaves = 7,    // number of octaves displayed
}) {
  const [activePart,  setActivePart]  = useState('all')
  const [playing,     setPlaying]     = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration,    setDuration]    = useState(0)
  const [loading,     setLoading]     = useState(false)
  const [dragging,    setDragging]    = useState(false)

  const audioRef     = useRef(null)
  const rollRef      = useRef(null)    // scrollable roll container
  const isDragging   = useRef(false)
  const rafRef       = useRef(null)

  // MIDI range for this keyboard/roll
  const midiLo    = keyboardStart
  const midiHi    = keyboardStart + keyboardOctaves * 12 - 1
  const pitchRange = midiHi - midiLo + 1
  const rollHeight = pitchRange * NOTE_H
  const totalH     = HEADER_H + rollHeight

  // Roll width from note content
  const maxTime   = notes.length
    ? Math.max(...notes.map(n => n.offset ?? n.onset + 0.5))
    : 10
  const rollWidth = Math.max(800, maxTime * PX_PER_S)

  // Beat / measure layout
  const beatDur   = 60 / tempo
  const beatCount = Math.ceil(maxTime / beatDur)

  // Part filter range
  const [partLo, partHi] = PART_RANGE[activePart]

  // ── Active pitches at current playhead position ─────────────────────────
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

  // ── Audio events ──────────────────────────────────────────────────────────
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
    audio.pause()
    setPlaying(false)
    setCurrentTime(0)
    setDuration(0)
    setLoading(true)
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

  // rAF loop: scroll roll so current time aligns with the left keyboard edge
  useEffect(() => {
    if (!playing) {
      if (rafRef.current) { cancelAnimationFrame(rafRef.current); rafRef.current = null }
      return
    }
    function tick() {
      const audio = audioRef.current
      const roll  = rollRef.current
      if (audio && roll && !isDragging.current) {
        roll.scrollLeft = audio.currentTime * PX_PER_S
      }
      rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)
    return () => { if (rafRef.current) { cancelAnimationFrame(rafRef.current); rafRef.current = null } }
  }, [playing])

  // ── Transport ─────────────────────────────────────────────────────────────
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

  // ── Seek ──────────────────────────────────────────────────────────────────
  const seekToX = useCallback((clientX) => {
    const c     = rollRef.current
    const audio = audioRef.current
    if (!c || !audio) return
    const rect = c.getBoundingClientRect()
    const x    = clientX - rect.left + c.scrollLeft
    const t    = Math.max(0, Math.min(x / PX_PER_S, duration || maxTime))
    audio.currentTime = t
    setCurrentTime(t)
    c.scrollLeft = t * PX_PER_S
  }, [duration, maxTime])

  const onMouseDown = useCallback((e) => {
    if (e.button !== 0) return
    isDragging.current = true; setDragging(true); seekToX(e.clientX)
  }, [seekToX])
  const onMouseMove = useCallback((e) => {
    if (!isDragging.current) return; seekToX(e.clientX)
  }, [seekToX])
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

  const onTouchStart = useCallback((e) => seekToX(e.touches[0].clientX), [seekToX])
  const onTouchMove  = useCallback((e) => seekToX(e.touches[0].clientX), [seekToX])

  const pct = duration > 0 ? (currentTime / duration) * 100 : 0

  // ── Render ────────────────────────────────────────────────────────────────
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
                color: isActive ? color : '#71717a',
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

      {/* ── Keyboard + Roll ── */}
      <div
        className="rounded-xl overflow-hidden select-none"
        style={{
          display: 'flex',
          background: 'rgba(0,0,0,0.45)',
          border: `1px solid ${dragging ? 'rgba(245,158,11,0.35)' : 'rgba(255,255,255,0.07)'}`,
          transition: 'border-color 0.15s ease',
        }}
      >
        {/* Fixed keyboard panel */}
        <KeyboardSVG
          midiLo={midiLo}
          midiHi={midiHi}
          activePitches={activePitches}
        />

        {/* Scrollable roll */}
        <div
          ref={rollRef}
          style={{ overflowX: 'auto', flex: 1, cursor: 'col-resize' }}
          onMouseDown={onMouseDown}
          onTouchStart={onTouchStart}
          onTouchMove={onTouchMove}
        >
          <svg width={rollWidth} height={totalH} style={{ display: 'block' }}>

            {/* Roll background */}
            <rect x={0} y={0} width={rollWidth} height={totalH} fill="rgba(0,0,0,0.0)" />

            {/* Measure header background */}
            <rect x={0} y={0} width={rollWidth} height={HEADER_H} fill="rgba(0,0,0,0.3)" />
            <line x1={0} y1={HEADER_H - 0.5} x2={rollWidth} y2={HEADER_H - 0.5}
              stroke="rgba(255,255,255,0.06)" strokeWidth={1} />

            {/* Beat / measure grid */}
            {Array.from({ length: beatCount + 1 }, (_, b) => {
              const x          = b * beatDur * PX_PER_S
              const isMeasure  = b % 4 === 0
              const measureNum = Math.floor(b / 4) + 1
              return (
                <g key={b}>
                  <line x1={x} y1={HEADER_H} x2={x} y2={totalH}
                    stroke={isMeasure ? 'rgba(255,255,255,0.09)' : 'rgba(255,255,255,0.035)'}
                    strokeWidth={isMeasure ? 1 : 0.5} />
                  <line
                    x1={x} y1={isMeasure ? HEADER_H * 0.3 : HEADER_H * 0.58}
                    x2={x} y2={HEADER_H - 1}
                    stroke={isMeasure ? 'rgba(245,158,11,0.55)' : 'rgba(255,255,255,0.18)'}
                    strokeWidth={isMeasure ? 1.5 : 0.75} />
                  {isMeasure && (
                    <text x={x + 4} y={HEADER_H * 0.7} fontSize={9}
                      fill="rgba(245,158,11,0.7)" fontFamily="ui-monospace,monospace" fontWeight="700">
                      {measureNum}
                    </text>
                  )}
                </g>
              )
            })}

            {/* C-note horizontal lines + subtle row shading for black keys */}
            {Array.from({ length: pitchRange }, (_, i) => {
              const pitch   = midiHi - i
              const isC     = pitch % 12 === 0
              const isBlack_ = isBlack(pitch)
              return (
                <g key={pitch}>
                  {isBlack_ && (
                    <rect x={0} y={HEADER_H + i * NOTE_H} width={rollWidth} height={NOTE_H}
                      fill="rgba(0,0,0,0.18)" />
                  )}
                  {isC && (
                    <line x1={0} y1={HEADER_H + i * NOTE_H} x2={rollWidth} y2={HEADER_H + i * NOTE_H}
                      stroke="rgba(255,255,255,0.07)" strokeWidth={1} />
                  )}
                </g>
              )
            })}

            {/* Notes */}
            {notes.map((n, i) => {
              const p = parseInt(n.pitch, 10)
              if (p < midiLo || p > midiHi) return null

              const x       = n.onset * PX_PER_S
              const w       = Math.max(4, ((n.offset ?? n.onset + 0.25) - n.onset) * PX_PER_S)
              const rowIdx  = midiHi - p
              const y       = HEADER_H + rowIdx * NOTE_H + 1
              const { fill, glow } = REG_COLOR[classify(p)]

              const inPart   = activePart === 'all' || (p >= partLo && p <= partHi)
              const isActive = activePitches.has(p)

              return (
                <rect key={i}
                  x={x} y={y}
                  width={w} height={NOTE_H - 2}
                  rx={3}
                  fill={fill}
                  opacity={inPart ? (isActive ? 1 : 0.70) : 0.10}
                  style={isActive
                    ? { filter: `drop-shadow(0 0 3px ${glow}) drop-shadow(0 0 7px ${glow})` }
                    : undefined}
                />
              )
            })}

          </svg>
        </div>
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
