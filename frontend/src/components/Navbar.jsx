import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'

export default function Navbar() {
  return (
    <motion.nav
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.55, ease: [0.4, 0, 0.2, 1] }}
      style={{
        background: 'rgba(6,6,10,0.75)',
        backdropFilter: 'blur(32px) saturate(200%)',
        WebkitBackdropFilter: 'blur(32px) saturate(200%)',
        borderBottom: '1px solid rgba(255,255,255,0.05)',
        boxShadow: '0 1px 0 rgba(245,158,11,0.06)',
      }}
      className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-8 h-14"
    >
      <Link to="/" className="flex items-center gap-2.5">
        <span style={{
          fontSize: 13,
          color: 'rgba(245,158,11,0.5)',
          fontFamily: 'ui-monospace,monospace',
          letterSpacing: '0.04em',
        }}>
          ◈
        </span>
        <span className="text-gradient-animated font-bold" style={{ fontSize: 13, letterSpacing: '-0.01em' }}>
          Arrangement Transformer
        </span>
      </Link>

      <a
        href="#how-it-works"
        style={{
          fontSize: 10,
          fontWeight: 700,
          letterSpacing: '0.14em',
          color: 'rgba(255,255,255,0.25)',
          fontFamily: 'ui-monospace,monospace',
          transition: 'color 0.2s',
        }}
        onMouseEnter={e => e.currentTarget.style.color = 'rgba(245,158,11,0.7)'}
        onMouseLeave={e => e.currentTarget.style.color = 'rgba(255,255,255,0.25)'}
      >
        HOW IT WORKS
      </a>
    </motion.nav>
  )
}
