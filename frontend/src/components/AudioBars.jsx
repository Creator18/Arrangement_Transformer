import { motion } from 'framer-motion'

const BAR_COUNT = 12

export default function AudioBars({ color = '#0071e3', height = 28, className = '' }) {
  return (
    <div
      className={`flex items-end gap-[3px] ${className}`}
      style={{ height }}
      aria-hidden
    >
      {Array.from({ length: BAR_COUNT }, (_, i) => {
        const minH = height * 0.15
        const maxH = height * (0.4 + Math.random() * 0.6)
        const duration = 0.55 + (i % 5) * 0.11
        const delay = (i % 4) * 0.08

        return (
          <motion.span
            key={i}
            style={{
              width: 3,
              borderRadius: 2,
              background: color,
              display: 'block',
            }}
            animate={{ height: [minH, maxH, minH * 1.4, maxH * 0.7, minH] }}
            transition={{
              duration,
              delay,
              repeat: Infinity,
              ease: 'easeInOut',
            }}
          />
        )
      })}
    </div>
  )
}
