import { useEffect } from 'react'
import { Routes, Route, useLocation } from 'react-router-dom'
import { AnimatePresence } from 'framer-motion'
import Navbar from './components/Navbar'
import LandingPage from './pages/LandingPage'
import ProcessingPage from './pages/ProcessingPage'
import ResultsPage from './pages/ResultsPage'
import { initSmoothScroll, getLenis } from './utils/smoothscroll'

export default function App() {
  const { pathname } = useLocation()

  // One Lenis for the whole app, started once.  It sets real scrollTop, so
  // every existing window-scroll listener keeps working unchanged.
  useEffect(() => initSmoothScroll(), [])

  // Route changes must land at the top instantly.  Lenis keeps its own
  // idea of scroll position, so a bare window.scrollTo(0,0) would be
  // overwritten on its next frame and the new page would open mid-way
  // down.  `immediate` tells Lenis to accept the jump without animating.
  useEffect(() => {
    const lenis = getLenis()
    if (lenis) lenis.scrollTo(0, { immediate: true })
    else window.scrollTo(0, 0)
  }, [pathname])

  return (
    <div className="min-h-screen">
      <Navbar />
      <AnimatePresence mode="wait">
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/processing/:sessionId" element={<ProcessingPage />} />
          <Route path="/results/:sessionId" element={<ResultsPage />} />
        </Routes>
      </AnimatePresence>
    </div>
  )
}
