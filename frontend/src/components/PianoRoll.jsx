import { useMemo } from 'react'

const MIDI_LOW = 36
const MIDI_HIGH = 96
const NOTE_HEIGHT = 6
const PIXELS_PER_SECOND = 80

const PART_COLORS = {
  bass:    { fill: '#3b82f6', opacity: 0.85 },   // blue
  melody:  { fill: '#f87171', opacity: 0.85 },   // coral
  voice:   { fill: '#a78bfa', opacity: 0.80 },   // soft purple
}

function classifyPitch(pitch) {
  if (pitch < 48) return 'bass'
  if (pitch <= 60) return 'melody'
  return 'voice'
}

export default function PianoRoll({ originalNotes = [], transformedNotes = [] }) {
  const maxTime = useMemo(() => {
    const all = [...originalNotes, ...transformedNotes]
    return all.length ? Math.max(...all.map(n => n.offset || n.onset + 0.5)) : 10
  }, [originalNotes, transformedNotes])

  const width = Math.max(600, maxTime * PIXELS_PER_SECOND)
  const pitchRange = MIDI_HIGH - MIDI_LOW + 1
  const height = pitchRange * NOTE_HEIGHT

  function renderNotes(notes, yOffset) {
    return notes.map((n, i) => {
      const x = n.onset * PIXELS_PER_SECOND
      const w = Math.max(4, ((n.offset || n.onset + 0.25) - n.onset) * PIXELS_PER_SECOND)
      const y = yOffset + (MIDI_HIGH - n.pitch) * NOTE_HEIGHT
      const { fill, opacity } = PART_COLORS[classifyPitch(n.pitch)]
      return (
        <rect
          key={i}
          x={x} y={y}
          width={w} height={NOTE_HEIGHT - 1}
          rx={2}
          fill={fill}
          opacity={opacity}
        />
      )
    })
  }

  const totalHeight = height * 2 + 24

  return (
    <div className="overflow-x-auto rounded-xl">
      <svg width={width} height={totalHeight} className="block">
        {/* Background grid lines (C notes) */}
        {Array.from({ length: pitchRange }, (_, i) => {
          const pitch = MIDI_HIGH - i
          const isC = pitch % 12 === 0
          if (!isC) return null
          return (
            <line
              key={i}
              x1={0} y1={i * NOTE_HEIGHT}
              x2={width} y2={i * NOTE_HEIGHT}
              stroke="rgba(0,0,0,0.06)" strokeWidth={1}
            />
          )
        })}

        {/* Original */}
        <text x={6} y={10} fontSize={10} fill="#6e6e73" fontFamily="sans-serif">Original</text>
        {renderNotes(originalNotes, 12)}

        {/* Divider */}
        <line x1={0} y1={height + 12} x2={width} y2={height + 12} stroke="rgba(0,0,0,0.08)" strokeWidth={1} />

        {/* Transformed */}
        <text x={6} y={height + 22} fontSize={10} fill="#6e6e73" fontFamily="sans-serif">Transformed</text>
        {renderNotes(transformedNotes, height + 24)}
      </svg>

      {/* Legend */}
      <div className="flex gap-4 mt-3 px-1">
        {Object.entries(PART_COLORS).map(([label, { fill }]) => (
          <div key={label} className="flex items-center gap-1.5">
            <div className="w-3 h-3 rounded-sm" style={{ background: fill }} />
            <span className="text-xs text-[#6e6e73] capitalize">{label}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
