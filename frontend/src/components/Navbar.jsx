import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'

export default function Navbar() {
  return (
    <motion.nav
      initial={{ opacity: 0, y: -16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: [0.4, 0, 0.2, 1] }}
      className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-8 py-4 glass"
    >
      <Link to="/" className="text-sm font-semibold tracking-tight text-gradient">
        Arrangement Transformer
      </Link>
      <a
        href="#how-it-works"
        className="text-sm text-[#6e6e73] hover:text-[#1d1d1f] transition-colors duration-200"
      >
        How it works
      </a>
    </motion.nav>
  )
}
