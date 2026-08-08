/**
 * smoothscroll.js — Lenis, wired as a singleton.
 *
 * Why this exists
 * ---------------
 * Every scroll-driven thing we have (ScrollStage's backdrop, SectionRail's
 * faders, RackSection's parallax) reads `window.scrollY`.  On Windows a
 * mouse wheel delivers scroll in large discrete jumps — around 100px per
 * notch — so those readers were being fed a staircase, not a ramp.  That is
 * exactly why the velocity-driven backdrop felt steppy: the input was.
 *
 * Lenis intercepts the wheel and interpolates toward the target position,
 * so `scrollY` becomes continuous.  Crucially it does this by setting real
 * `scrollTop`, which means native `scroll` events still fire and every
 * existing listener keeps working untouched — this is additive, not a
 * rewrite.
 *
 * The singleton matters for one specific reason: anything that wants to
 * MOVE the page (the rail's click-to-jump) must go through Lenis.  Calling
 * `scrollIntoView({behavior:'smooth'})` while Lenis is running sets up two
 * animators fighting over one scrollTop, which reads as a stutter or a
 * jump that overshoots and snaps back.
 */
import Lenis from 'lenis'

let lenis = null
let raf = 0

/** The live instance, or null when smooth scroll is off. */
export function getLenis() {
  return lenis
}

/**
 * Users who ask for reduced motion get the plain native scroll.  This is
 * not decoration — smooth-scroll hijacking is a common migraine and
 * vestibular trigger, and it is the one animation on the page the user
 * cannot avoid by not looking at it.
 */
function prefersReducedMotion() {
  return typeof window !== 'undefined' &&
    window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
}

/**
 * Start smooth scrolling.  Returns a cleanup function.
 * Safe to call when one is already running — it will not stack instances.
 */
export function initSmoothScroll() {
  if (lenis) return () => {}
  if (prefersReducedMotion()) return () => {}

  lenis = new Lenis({
    // ~0.9s to settle.  Long enough to read as gliding, short enough that
    // the page still feels like it obeys you.  Past ~1.4s it starts to
    // feel like lag rather than smoothness.
    duration: 0.9,
    // Exponential ease-out: fast response to the input, gentle arrival.
    easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
    smoothWheel: true,
    // Touch devices already have momentum scrolling in hardware; syncing
    // ours on top makes them feel sluggish and disconnected.  (`syncTouch`
    // is the v1 name — the old `smoothTouch` is silently ignored, which is
    // the worst kind of wrong: it looks configured and does nothing.)
    syncTouch: false,
    wheelMultiplier: 1,
    touchMultiplier: 1.6,
  })

  const frame = (time) => {
    lenis?.raf(time)
    raf = requestAnimationFrame(frame)
  }
  raf = requestAnimationFrame(frame)

  return () => {
    cancelAnimationFrame(raf)
    lenis?.destroy()
    lenis = null
  }
}

/**
 * Scroll to an element (or offset) through Lenis when it is running, and
 * fall back to the native path when it is not — reduced-motion users and
 * older browsers still need the jump to work.
 */
export function scrollToTarget(target, { offset = 0 } = {}) {
  const el = typeof target === 'string' ? document.getElementById(target) : target
  if (!el) return
  if (lenis) {
    lenis.scrollTo(el, { offset, duration: 1.1 })
  } else {
    const y = el.getBoundingClientRect().top + window.scrollY + offset
    window.scrollTo({ top: y, behavior: 'smooth' })
  }
}
