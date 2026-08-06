/**
 * scrollfx.js — scroll-driven animation plumbing.
 *
 * anime.js v3 has no built-in scroll observer, so we do it the classic way:
 * build the animation paused (`autoplay: false`) and scrub it with
 * `anim.seek(progress * anim.duration)` from a rAF-throttled scroll
 * listener.  That gives the "animation is glued to the scrollbar" feel of
 * the anime.js site itself.
 */

/**
 * 0 → element enters the bottom of the viewport,
 * 1 → element leaves the top.  Clamped.
 */
export function elementProgress(el) {
  const rect = el.getBoundingClientRect()
  const vh = window.innerHeight || document.documentElement.clientHeight
  const total = vh + rect.height
  const seen = vh - rect.top
  return Math.max(0, Math.min(1, seen / total))
}

/**
 * Call `cb(progress)` whenever the page scrolls or resizes, throttled to
 * animation frames.  Returns a cleanup function — call it on unmount.
 */
export function onScrollScrub(el, cb) {
  let ticking = false
  const update = () => {
    ticking = false
    if (el) cb(elementProgress(el))
  }
  const onScroll = () => {
    if (!ticking) {
      ticking = true
      requestAnimationFrame(update)
    }
  }
  window.addEventListener('scroll', onScroll, { passive: true })
  window.addEventListener('resize', onScroll, { passive: true })
  onScroll()   // initial position
  return () => {
    window.removeEventListener('scroll', onScroll)
    window.removeEventListener('resize', onScroll)
  }
}

/**
 * Run `cb(el)` when `el` first becomes meaningfully visible.
 * `once: false` re-fires with a second arg `visible` on every transition.
 */
export function whenInView(el, cb, { threshold = 0.35, once = true } = {}) {
  if (!el || typeof IntersectionObserver === 'undefined') return () => {}
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (e.isIntersecting) {
        cb(el, true)
        if (once) io.disconnect()
      } else if (!once) {
        cb(el, false)
      }
    }
  }, { threshold })
  io.observe(el)
  return () => io.disconnect()
}
