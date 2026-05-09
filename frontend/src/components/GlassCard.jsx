import { motion } from 'framer-motion'
import { forwardRef } from 'react'

const GlassCard = forwardRef(({ children, className = '', hover = false, onClick, ...props }, ref) => (
  <motion.div
    ref={ref}
    onClick={onClick}
    className={`glass rounded-2xl ${hover ? 'glass-hover cursor-pointer' : ''} ${className}`}
    {...props}
  >
    {children}
  </motion.div>
))

GlassCard.displayName = 'GlassCard'
export default GlassCard
