/**
 * scrollsnap — one gesture, one act.
 *
 * The page reads as a sequence of full-viewport movements, so free scrolling
 * lets you stop halfway between two of them, which is the one position none
 * of them were composed for.  Snapping makes the unit of navigation the act
 * rather than the pixel.
 *
 * Why not CSS scroll-snap
 * -----------------------
 * `scroll-snap-type` and Lenis fight: Lenis animates scrollTop itself, and
 * the browser's snap engine yanks the same value toward a snap point mid-
 * animation.  The result is a stutter that looks like jank but is really two
 * systems steering.  So the wheel is intercepted in the CAPTURE phase, before
 * Lenis sees it, and the movement is handed to Lenis as an explicit
 * scrollTo — one animator, one owner.
 *
 * The exemption is the interesting part
 * -------------------------------------
 * Some sections are not a single frame.  The pipeline act is five viewports
 * of scroll-scrubbed narrative; snapping past it would skip the entire story
 * it exists to tell.  Anything marked `data-free-scroll` therefore scrolls
 * normally INSIDE itself and only snaps at its boundaries.  A snap system
 * that cannot express "not here" would have quietly destroyed that section.
 */
import { getLenis } from './smoothscroll'

const LOCK_MS = 900        // ignore input while a snap is in flight
const THRESHOLD = 12       // ignore trackpad micro-deltas

export function initScrollSnap(sectionIds) {
  if (typeof window === 'undefined') return () => {}

  // Reduced motion: snapping is forced movement the user cannot opt out of,
  // which is exactly what that preference is asking us not to do.
  if (window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) {
    return () => {}
  }

  let locked = false
  let lockTimer = null

  const els = () => sectionIds
    .map((id) => document.getElementById(id))
    .filter(Boolean)

  /** Index of the section currently filling most of the viewport. */
  const currentIndex = (list) => {
    const mid = window.innerHeight / 2
    let best = 0
    let bestDist = Infinity
    list.forEach((el, i) => {
      const r = el.getBoundingClientRect()
      const d = r.top > mid ? r.top - mid : r.bottom < mid ? mid - r.bottom : 0
      if (d < bestDist) { bestDist = d; best = i }
    })
    return best
  }

  /**
   * Inside a free-scroll section and not yet at its edge?  Then this gesture
   * belongs to that section, not to the snap system.
   */
  const insideFreeScroll = (el, dir) => {
    if (!el?.hasAttribute?.('data-free-scroll')) return false
    const r = el.getBoundingClientRect()
    const vh = window.innerHeight
    const EDGE = 8
    return dir > 0 ? r.bottom > vh + EDGE : r.top < -EDGE
  }

  const onWheel = (e) => {
    if (Math.abs(e.deltaY) < THRESHOLD) return
    const list = els()
    if (list.length < 2) return

    const i = currentIndex(list)
    const dir = e.deltaY > 0 ? 1 : -1

    if (insideFreeScroll(list[i], dir)) return   // let it scrub

    if (locked) { e.preventDefault(); e.stopPropagation(); return }

    const next = i + dir
    if (next < 0 || next >= list.length) return  // let the page edge behave

    // Capture phase + stopPropagation keeps Lenis from also acting on this
    // same wheel event, which would double the movement.
    e.preventDefault()
    e.stopPropagation()

    const lenis = getLenis()
    const target = list[next]
    if (lenis) lenis.scrollTo(target, { duration: 1.0 })
    else target.scrollIntoView({ behavior: 'smooth', block: 'start' })

    locked = true
    clearTimeout(lockTimer)
    lockTimer = setTimeout(() => { locked = false }, LOCK_MS)
  }

  const onKey = (e) => {
    if (!['PageDown', 'PageUp'].includes(e.key)) return
    onWheel({
      deltaY: e.key === 'PageDown' ? 100 : -100,
      preventDefault: () => e.preventDefault(),
      stopPropagation: () => e.stopPropagation(),
    })
  }

  window.addEventListener('wheel', onWheel, { passive: false, capture: true })
  window.addEventListener('keydown', onKey)

  return () => {
    clearTimeout(lockTimer)
    window.removeEventListener('wheel', onWheel, { capture: true })
    window.removeEventListener('keydown', onKey)
  }
}
