import { motion } from 'framer-motion'
import { forwardRef, useRef, useCallback } from 'react'

const GlassCard = forwardRef(({
  children,
  className = '',
  hover = false,
  tilt = false,
  tiltStrength = 6,
  onClick,
  style,
  ...props
}, ref) => {
  const internalRef = useRef(null)
  const cardRef = ref || internalRef

  const onMouseMove = useCallback((e) => {
    if (!tilt) return
    const el = cardRef.current
    if (!el) return
    const rect = el.getBoundingClientRect()
    const x = (e.clientX - rect.left) / rect.width  - 0.5   // -0.5 → 0.5
    const y = (e.clientY - rect.top)  / rect.height - 0.5
    el.style.transform  = `perspective(900px) rotateX(${-y * tiltStrength}deg) rotateY(${x * tiltStrength}deg) translateZ(6px)`
    el.style.transition = 'transform 0.08s ease'
    // Shift inner glare
    const glare = el.querySelector('[data-glare]')
    if (glare) {
      glare.style.opacity  = '1'
      glare.style.left     = `${(x + 0.5) * 100}%`
      glare.style.top      = `${(y + 0.5) * 100}%`
    }
  }, [tilt, tiltStrength, cardRef])

  const onMouseLeave = useCallback(() => {
    if (!tilt) return
    const el = cardRef.current
    if (!el) return
    el.style.transform  = 'perspective(900px) rotateX(0deg) rotateY(0deg) translateZ(0px)'
    el.style.transition = 'transform 0.45s cubic-bezier(0.4,0,0.2,1)'
    const glare = el.querySelector('[data-glare]')
    if (glare) glare.style.opacity = '0'
  }, [tilt, cardRef])

  return (
    <motion.div
      ref={cardRef}
      onClick={onClick}
      className={`glass rounded-2xl ${hover ? 'glass-hover cursor-pointer' : ''} ${className}`}
      style={{ position: 'relative', ...style }}
      onMouseMove={onMouseMove}
      onMouseLeave={onMouseLeave}
      {...props}
    >
      {/* Glare highlight (only when tilt=true) */}
      {tilt && (
        <div
          data-glare
          style={{
            position: 'absolute',
            width: 120,
            height: 120,
            borderRadius: '50%',
            background: 'radial-gradient(circle, rgba(255,255,255,0.06) 0%, transparent 70%)',
            transform: 'translate(-50%, -50%)',
            pointerEvents: 'none',
            opacity: 0,
            transition: 'opacity 0.2s',
            zIndex: 1,
          }}
        />
      )}
      {children}
    </motion.div>
  )
})

GlassCard.displayName = 'GlassCard'
export default GlassCard
