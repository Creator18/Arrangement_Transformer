import { useEffect, useRef, useMemo } from 'react'
import anime from '../utils/animefx'
import { onScrollScrub } from '../utils/scrollfx'

/**
 * MasterDial — the workstation's "master knob".
 *
 * Visual anatomy (all SVG, viewBox 0 0 400 400):
 *   - Outer arc segments: 6 glowing strokes in the app palette, breathing
 *   - Tick ring: 72 hairline ticks, the group slowly rotates with scroll
 *   - Bezel circles: dark rings for machined depth
 *   - LED core: dot-matrix cluster pulsing outward from the center
 *   - Value arc: bright indicator arc scrubbed by page scroll progress
 */

const PALETTE = ['#f59e0b', '#a78bfa', '#06b6d4', '#34d399', '#f87171', '#fbbf24']
const CX = 200
const CY = 200

function arcPath(cx, cy, r, startDeg, endDeg) {
  const s = (startDeg - 90) * Math.PI / 180
  const e = (endDeg   - 90) * Math.PI / 180
  const large = endDeg - startDeg > 180 ? 1 : 0
  return `M ${cx + r * Math.cos(s)} ${cy + r * Math.sin(s)} ` +
         `A ${r} ${r} 0 ${large} 1 ${cx + r * Math.cos(e)} ${cy + r * Math.sin(e)}`
}

export default function MasterDial({ size = 420 }) {
  const rootRef  = useRef(null)
  const ticksRef = useRef(null)
  const valueRef = useRef(null)
  const ledsRef  = useRef(null)

  // 6 arc segments with even gaps around the ring
  const segments = useMemo(() => {
    const n = PALETTE.length
    const span = 360 / n
    const gap  = 14
    return PALETTE.map((color, i) => ({
      color,
      d: arcPath(CX, CY, 178, i * span + gap / 2, (i + 1) * span - gap / 2),
    }))
  }, [])

  // 72-tick ring
  const ticks = useMemo(() => (
    Array.from({ length: 72 }, (_, i) => {
      const a = (i * 5 - 90) * Math.PI / 180
      const r1 = 152, r2 = i % 6 === 0 ? 140 : 146
      return {
        x1: CX + r1 * Math.cos(a), y1: CY + r1 * Math.sin(a),
        x2: CX + r2 * Math.cos(a), y2: CY + r2 * Math.sin(a),
        major: i % 6 === 0,
      }
    })
  ), [])

  // LED dot-matrix core — concentric grid clipped to a circle
  const leds = useMemo(() => {
    const out = []
    const step = 17
    for (let gx = -4; gx <= 4; gx++) {
      for (let gy = -4; gy <= 4; gy++) {
        const x = gx * step, y = gy * step
        if (Math.hypot(x, y) <= 72) out.push({ x: CX + x, y: CY + y })
      }
    }
    return out
  }, [])

  useEffect(() => {
    const root = rootRef.current
    if (!root) return

    // LED core: breathing pulse radiating from the center — the dial's
    // "the machine is alive" idle animation.
    const ledEls = ledsRef.current ? Array.from(ledsRef.current.children) : []
    const ledAnim = anime({
      targets: ledEls,
      opacity: [0.12, 0.85],
      scale:   [0.55, 1],
      delay:   anime.stagger(55, { grid: [9, 9], from: 'center' }),
      direction: 'alternate',
      loop: true,
      duration: 1050,
      easing: 'easeInOutSine',
    })

    // Value arc: full sweep is 300°, scrubbed by scroll.
    const valueEl = valueRef.current
    const VALUE_C = 2 * Math.PI * 166 * (300 / 360)
    if (valueEl) {
      valueEl.style.strokeDasharray  = `${VALUE_C} ${2 * Math.PI * 166}`
      valueEl.style.strokeDashoffset = `${VALUE_C}`
    }

    // Scroll scrub: tick-ring rotation + value arc draw follow the page.
    const stopScrub = onScrollScrub(root, (p) => {
      if (ticksRef.current) {
        ticksRef.current.style.transform = `rotate(${p * 90}deg)`
      }
      if (valueEl) {
        valueEl.style.strokeDashoffset = `${VALUE_C * (1 - p)}`
      }
    })

    return () => { ledAnim.pause(); stopScrub() }
  }, [])

  return (
    <div ref={rootRef} style={{ width: size, height: size, position: 'relative' }} aria-hidden>
      <svg viewBox="0 0 400 400" width={size} height={size}>
        {/* Bezel */}
        <circle cx={CX} cy={CY} r={190} fill="rgba(255,255,255,0.015)"
                stroke="rgba(255,255,255,0.06)" strokeWidth="1" />
        <circle cx={CX} cy={CY} r={132} fill="rgba(0,0,0,0.35)"
                stroke="rgba(255,255,255,0.05)" strokeWidth="1" />
        <circle cx={CX} cy={CY} r={96} fill="rgba(255,255,255,0.01)"
                stroke="rgba(255,255,255,0.04)" strokeWidth="1" />

        {/* Outer glowing arc segments */}
        {segments.map((s, i) => (
          <path key={i} d={s.d} fill="none"
                stroke={s.color} strokeWidth="5" strokeLinecap="round"
                style={{ filter: `drop-shadow(0 0 6px ${s.color})`, opacity: 0.9 }} />
        ))}

        {/* Value arc — scrubbed by scroll */}
        <circle
          ref={valueRef}
          cx={CX} cy={CY} r={166}
          fill="none"
          stroke="#fff7ed"
          strokeWidth="2.5"
          strokeLinecap="round"
          style={{
            filter: 'drop-shadow(0 0 5px rgba(255,247,237,0.8))',
            transformBox: 'view-box',
            transformOrigin: 'center',
            transform: 'rotate(120deg)',
            opacity: 0.9,
          }}
        />

        {/* Tick ring — rotates with scroll */}
        <g ref={ticksRef}
           style={{ transformBox: 'view-box', transformOrigin: 'center',
                    transition: 'transform 0.08s linear' }}>
          {ticks.map((t, i) => (
            <line key={i} x1={t.x1} y1={t.y1} x2={t.x2} y2={t.y2}
                  stroke={t.major ? 'rgba(255,255,255,0.35)' : 'rgba(255,255,255,0.12)'}
                  strokeWidth={t.major ? 1.6 : 0.8} />
          ))}
        </g>

        {/* LED dot-matrix core */}
        <g ref={ledsRef}>
          {leds.map((l, i) => (
            <circle key={i} cx={l.x} cy={l.y} r={6.2}
                    fill="#f59e0b"
                    style={{
                      opacity: 0.12,
                      transformBox: 'fill-box',
                      transformOrigin: 'center',
                    }} />
          ))}
        </g>
      </svg>

      {/* Label under the dial */}
      <div style={{
        position: 'absolute', bottom: -6, left: 0, right: 0,
        textAlign: 'center',
        fontSize: 9, fontWeight: 700, letterSpacing: '0.24em',
        color: 'rgba(255,255,255,0.25)',
        fontFamily: 'ui-monospace,monospace',
      }}>
        MASTER · SIGNAL ENGINE
      </div>
    </div>
  )
}
