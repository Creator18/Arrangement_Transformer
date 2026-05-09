import { motion } from 'framer-motion'

export default function GradientOrbs() {
  const orbs = [
    { x: '10%',  y: '20%', size: 800,  color: 'rgba(124,58,237,0.15)',  delay: 0 },
    { x: '65%',  y: '10%', size: 700,  color: 'rgba(6,182,212,0.10)',   delay: 2 },
    { x: '80%',  y: '60%', size: 900,  color: 'rgba(245,158,11,0.08)',  delay: 4 },
    { x: '20%',  y: '70%', size: 650,  color: 'rgba(34,197,94,0.08)',   delay: 1 },
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
            // Use margin instead of CSS transform so Framer Motion owns transform
            // exclusively — mixing style.transform with animate x/y/scale causes
            // subpixel compositing differences across DPIs.
            marginLeft: -(orb.size / 2),
            marginTop:  -(orb.size / 2),
            filter: 'blur(60px)',
            willChange: 'transform',
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
