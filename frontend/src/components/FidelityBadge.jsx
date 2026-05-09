import { motion } from 'framer-motion'

const TIERS = {
  green:  { bg: 'bg-green-50',  ring: 'ring-green-200',  text: 'text-green-700',  label: 'High Fidelity' },
  yellow: { bg: 'bg-yellow-50', ring: 'ring-yellow-200', text: 'text-yellow-700', label: 'Review Suggested' },
  red:    { bg: 'bg-red-50',    ring: 'ring-red-200',    text: 'text-red-700',    label: 'Manual Review Required' },
}

export default function FidelityBadge({ score, tier }) {
  const t = TIERS[tier] || TIERS.green
  const pct = Math.round((score ?? 0) * 100)

  return (
    <div className={`inline-flex flex-col items-center gap-3 px-8 py-6 rounded-2xl ring-1 ${t.bg} ${t.ring}`}>
      <motion.span
        className={`text-5xl font-bold tabular-nums ${t.text}`}
        initial={{ opacity: 0, scale: 0.7 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ type: 'spring', stiffness: 300, damping: 20 }}
      >
        {pct}%
      </motion.span>
      <span className={`text-xs font-semibold uppercase tracking-wider ${t.text}`}>{t.label}</span>
    </div>
  )
}
