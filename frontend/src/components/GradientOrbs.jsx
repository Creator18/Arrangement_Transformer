import { useMemo } from 'react'
import { motion } from 'framer-motion'

// Deterministic pseudo-random so SSR/hydration is stable
function seededRand(seed) {
  let s = seed
  return () => {
    s = (s * 1664525 + 1013904223) & 0xffffffff
    return (s >>> 0) / 0xffffffff
  }
}

export default function GradientOrbs() {
  const orbs = [
    { x: '8%',  y: '15%', size: 820, color: 'rgba(124,58,237,0.14)',  delay: 0,  dur: 22 },
    { x: '70%', y: '8%',  size: 700, color: 'rgba(6,182,212,0.11)',   delay: 2,  dur: 18 },
    { x: '82%', y: '62%', size: 950, color: 'rgba(245,158,11,0.09)',  delay: 4,  dur: 26 },
    { x: '18%', y: '72%', size: 680, color: 'rgba(34,197,94,0.08)',   delay: 1,  dur: 20 },
    { x: '50%', y: '40%', size: 600, color: 'rgba(236,72,153,0.07)',  delay: 3,  dur: 24 },
    { x: '38%', y: '88%', size: 750, color: 'rgba(59,130,246,0.09)',  delay: 5,  dur: 19 },
  ]

  // Floating particles — small glowing dots
  const particles = useMemo(() => {
    const rand = seededRand(42)
    return Array.from({ length: 28 }, (_, i) => ({
      id: i,
      x:    rand() * 100,
      y:    rand() * 100,
      size: rand() * 2.5 + 1,
      opacity: rand() * 0.25 + 0.05,
      delay:   rand() * 12,
      dur:     rand() * 14 + 10,
      dx:      (rand() - 0.5) * 60,
      dy:      (rand() - 0.5) * 60,
      color:   ['#f59e0b','#a78bfa','#06b6d4','#34d399','#f87171','#fbbf24'][Math.floor(rand() * 6)],
    }))
  }, [])

  return (
    <div className="fixed inset-0 overflow-hidden pointer-events-none" aria-hidden>

      {/* ── Large aurora blobs ── */}
      {orbs.map((orb, i) => (
        <motion.div
          key={i}
          style={{
            position: 'absolute',
            left: orb.x,
            top:  orb.y,
            width:  orb.size,
            height: orb.size,
            borderRadius: '50%',
            background: `radial-gradient(circle, ${orb.color} 0%, transparent 68%)`,
            marginLeft: -(orb.size / 2),
            marginTop:  -(orb.size / 2),
            filter: 'blur(72px)',
            willChange: 'transform',
          }}
          animate={{
            x: [0, 40, -25, 15, 0],
            y: [0, -30, 20, -10, 0],
            scale: [1, 1.10, 0.93, 1.05, 1],
          }}
          transition={{
            duration: orb.dur,
            delay: orb.delay,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
        />
      ))}

      {/* ── Floating particles ── */}
      {particles.map((p) => (
        <motion.div
          key={p.id}
          style={{
            position: 'absolute',
            left: `${p.x}%`,
            top:  `${p.y}%`,
            width:  p.size,
            height: p.size,
            borderRadius: '50%',
            background: p.color,
            opacity: p.opacity,
            boxShadow: `0 0 ${p.size * 3}px ${p.color}`,
            willChange: 'transform',
          }}
          animate={{
            x: [0, p.dx, 0],
            y: [0, p.dy, 0],
            opacity: [p.opacity, p.opacity * 2.5, p.opacity],
          }}
          transition={{
            duration: p.dur,
            delay: p.delay,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
        />
      ))}

      {/* ── Subtle radial vignette ── */}
      <div
        style={{
          position: 'absolute',
          inset: 0,
          background: 'radial-gradient(ellipse 80% 60% at 50% 0%, transparent 0%, #06060a 75%)',
          pointerEvents: 'none',
        }}
      />
    </div>
  )
}
