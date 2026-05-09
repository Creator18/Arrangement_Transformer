import { motion } from 'framer-motion'

export default function GradientOrbs() {
  const orbs = [
    { x: '10%', y: '20%', size: 600, color: 'rgba(94,92,230,0.18)', delay: 0 },
    { x: '65%', y: '10%', size: 500, color: 'rgba(0,113,227,0.14)', delay: 2 },
    { x: '80%', y: '60%', size: 700, color: 'rgba(255,149,0,0.10)', delay: 4 },
    { x: '20%', y: '70%', size: 450, color: 'rgba(52,199,89,0.10)', delay: 1 },
  ]

  return (
    <div className="fixed inset-0 overflow-hidden pointer-events-none" aria-hidden>
      {orbs.map((orb, i) => (
        <motion.div
          key={i}
          style={{
            position: 'absolute',
            left: orb.x,
            top: orb.y,
            width: orb.size,
            height: orb.size,
            borderRadius: '50%',
            background: `radial-gradient(circle, ${orb.color} 0%, transparent 70%)`,
            transform: 'translate(-50%, -50%)',
            filter: 'blur(40px)',
          }}
          animate={{
            x: [0, 30, -20, 0],
            y: [0, -25, 15, 0],
            scale: [1, 1.08, 0.95, 1],
          }}
          transition={{
            duration: 18 + i * 3,
            delay: orb.delay,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
        />
      ))}
    </div>
  )
}
