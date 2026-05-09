import { motion } from 'framer-motion'

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

export default function FidelityBadge({ score, tier }) {
  const t = TIERS[tier] || TIERS.green
  const pct = Math.round((score ?? 0) * 100)

  return (
    <div
      className="inline-flex flex-col items-center gap-2 px-8 py-5 rounded-2xl"
      style={{
        background: t.bg,
        border: `1px solid ${t.border}`,
        boxShadow: `0 0 24px ${t.glow}, inset 0 1px 0 rgba(255,255,255,0.05)`,
      }}
    >
      <motion.span
        className="text-5xl font-bold tabular-nums"
        style={{ color: t.color, textShadow: `0 0 24px ${t.glow}` }}
        initial={{ opacity: 0, scale: 0.7 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ type: 'spring', stiffness: 300, damping: 20 }}
      >
        {pct}%
      </motion.span>
      <span
        className="text-xs font-bold uppercase tracking-widest"
        style={{ color: t.color, opacity: 0.8 }}
      >
        {t.label}
      </span>
    </div>
  )
}
