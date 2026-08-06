import { useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import { countUp, ringDraw } from '../utils/animefx'

const TIERS = {
  green: {
    label: 'High Fidelity',
    color: '#22c55e',
    bg: 'rgba(34,197,94,0.08)',
    border: 'rgba(34,197,94,0.3)',
    glow: 'rgba(34,197,94,0.2)',
  },
  yellow: {
    label: 'Review Suggested',
    color: '#f59e0b',
    bg: 'rgba(245,158,11,0.08)',
    border: 'rgba(245,158,11,0.3)',
    glow: 'rgba(245,158,11,0.2)',
  },
  red: {
    label: 'Manual Review Required',
    color: '#ef4444',
    bg: 'rgba(239,68,68,0.08)',
    border: 'rgba(239,68,68,0.3)',
    glow: 'rgba(239,68,68,0.2)',
  },
}

const RING_R = 46

export default function FidelityBadge({ score, tier }) {
  const t = TIERS[tier] || TIERS.green
  const pct = Math.round((score ?? 0) * 100)

  const numRef  = useRef(null)
  const ringRef = useRef(null)

  // anime.js: count the number up from 0 and draw the ring to the score
  // fraction in sync.  Runs once per score change; DOM-only, no re-renders.
  useEffect(() => {
    countUp(numRef.current, pct, { duration: 1600, suffix: '%', delay: 250 })
    ringDraw(ringRef.current, (score ?? 0), { duration: 1600, delay: 250 })
  }, [pct, score])

  return (
    <div
      className="inline-flex flex-col items-center gap-2 px-8 py-5 rounded-2xl"
      style={{
        background: t.bg,
        border: `1px solid ${t.border}`,
        boxShadow: `0 0 24px ${t.glow}, inset 0 1px 0 rgba(255,255,255,0.05)`,
      }}
    >
      {/* Ring + number */}
      <motion.div
        style={{ position: 'relative', width: RING_R * 2 + 16, height: RING_R * 2 + 16 }}
        initial={{ opacity: 0, scale: 0.7 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ type: 'spring', stiffness: 300, damping: 20 }}
      >
        <svg
          width={RING_R * 2 + 16}
          height={RING_R * 2 + 16}
          style={{ position: 'absolute', inset: 0, transform: 'rotate(-90deg)' }}
        >
          {/* Track */}
          <circle
            cx={RING_R + 8} cy={RING_R + 8} r={RING_R}
            fill="none"
            stroke="rgba(255,255,255,0.06)"
            strokeWidth={5}
          />
          {/* Score arc — drawn by anime.js via strokeDashoffset */}
          <circle
            ref={ringRef}
            cx={RING_R + 8} cy={RING_R + 8} r={RING_R}
            fill="none"
            stroke={t.color}
            strokeWidth={5}
            strokeLinecap="round"
            style={{ filter: `drop-shadow(0 0 6px ${t.glow})` }}
          />
        </svg>
        <span
          ref={numRef}
          className="text-4xl font-bold tabular-nums"
          style={{
            position: 'absolute',
            inset: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: t.color,
            textShadow: `0 0 24px ${t.glow}`,
          }}
        >
          0%
        </span>
      </motion.div>

      <span
        className="text-xs font-bold uppercase tracking-widest"
        style={{ color: t.color, opacity: 0.8 }}
      >
        {t.label}
      </span>
    </div>
  )
}
