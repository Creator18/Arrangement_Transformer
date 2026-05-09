import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'

export default function Navbar() {
  return (
    <motion.nav
      initial={{ opacity: 0, y: -16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: [0.4, 0, 0.2, 1] }}
      style={{
        background: 'rgba(10,10,11,0.8)',
        backdropFilter: 'blur(24px) saturate(180%)',
        WebkitBackdropFilter: 'blur(24px) saturate(180%)',
        borderBottom: '1px solid rgba(245,158,11,0.2)',
      }}
      className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-8 py-4"
    >
      <Link to="/" className="text-sm font-semibold tracking-tight text-gradient-amber">
        Arrangement Transformer
      </Link>
      <a
        href="#how-it-works"
        className="text-sm text-zinc-400 hover:text-white transition-colors duration-200"
      >
        How it works
      </a>
    </motion.nav>
  )
}
