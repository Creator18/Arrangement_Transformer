/**
 * animefx.js — shared anime.js effects layered on top of the existing
 * framer-motion design.  Division of labour:
 *   framer-motion — layout transitions, enter/exit, springs on React state
 *   anime.js      — timelines, staggers, count-ups, one-shot flourishes
 *                   that target DOM nodes directly (no React re-render cost)
 */
import anime from 'animejs/lib/anime.es.js'

/** Animate a number in an element's textContent from its current value to `to`. */
export function countUp(el, to, { duration = 1400, suffix = '', decimals = 0, delay = 0 } = {}) {
  if (!el) return
  const from = parseFloat(String(el.textContent).replace(/[^\d.-]/g, '')) || 0
  const state = { v: from }
  return anime({
    targets: state,
    v: to,
    duration,
    delay,
    easing: 'easeOutExpo',
    update: () => {
      el.textContent = `${state.v.toFixed(decimals)}${suffix}`
    },
  })
}

/** Wrap each character of an element's text in a span; returns the spans. */
export function letterize(el) {
  if (!el || el.dataset.letterized) return el ? Array.from(el.children) : []
  const text = el.textContent
  el.textContent = ''
  el.dataset.letterized = '1'
  const spans = []
  for (const ch of text) {
    const s = document.createElement('span')
    s.style.display = 'inline-block'
    s.style.whiteSpace = 'pre'
    s.textContent = ch
    el.appendChild(s)
    spans.push(s)
  }
  return spans
}

/** Letter-by-letter rise-in with a soft elastic settle. */
export function letterReveal(el, { delay = 0, stagger = 28, translate = 22 } = {}) {
  const spans = letterize(el)
  if (!spans.length) return
  anime.set(spans, { opacity: 0, translateY: translate })
  return anime({
    targets: spans,
    opacity: [0, 1],
    translateY: [translate, 0],
    delay: anime.stagger(stagger, { start: delay }),
    duration: 900,
    easing: 'easeOutElastic(1, .7)',
  })
}

/** Grid-aware stagger entrance for a set of pad/card elements. */
export function gridRise(targets, { grid, delay = 0, translate = 14 } = {}) {
  if (!targets || !targets.length) return
  anime.set(targets, { opacity: 0, translateY: translate, scale: 0.94 })
  return anime({
    targets,
    opacity: [0, 1],
    translateY: [translate, 0],
    scale: [0.94, 1],
    delay: grid
      ? anime.stagger(55, { grid, from: 'first', start: delay })
      : anime.stagger(55, { start: delay }),
    duration: 650,
    easing: 'easeOutCubic',
    // Clear inline transforms afterwards so hover/tilt handlers that write
    // their own `el.style.transform` aren't fighting stale anime values.
    complete: (a) => a.animatables.forEach(({ target }) => {
      target.style.transform = ''
      target.style.opacity = ''
    }),
  })
}

/** One-shot elastic pop + glow flash (e.g. when a processing step completes). */
export function elasticPop(el, color = '#22c55e') {
  if (!el) return
  const prevShadow = el.style.boxShadow
  return anime
    .timeline({ targets: el })
    .add({
      scale: [1, 1.06],
      duration: 140,
      easing: 'easeOutQuad',
      begin: () => { el.style.boxShadow = `0 0 28px ${color}55, 0 0 6px ${color}88` },
    })
    .add({
      scale: [1.06, 1],
      duration: 700,
      easing: 'easeOutElastic(1, .5)',
      complete: () => {
        el.style.boxShadow = prevShadow
        el.style.transform = ''
      },
    })
}

/** Clip-path wipe reveal — one element, GPU-cheap, works on huge SVGs. */
export function wipeReveal(el, { duration = 900, delay = 0, direction = 'up' } = {}) {
  if (!el) return
  const fromClip = direction === 'up'
    ? 'inset(100% 0 0 0)'
    : 'inset(0 0 100% 0)'
  el.style.clipPath = fromClip
  return anime({
    targets: el,
    clipPath: [fromClip, 'inset(0% 0 0 0)'],
    duration,
    delay,
    easing: 'easeInOutQuart',
    complete: () => { el.style.clipPath = '' },
  })
}

/** Draw an SVG circle ring from 0 to `fraction` of its circumference. */
export function ringDraw(circleEl, fraction = 1, { duration = 1600, delay = 200 } = {}) {
  if (!circleEl) return
  const r = parseFloat(circleEl.getAttribute('r'))
  const c = 2 * Math.PI * r
  circleEl.style.strokeDasharray = `${c}`
  circleEl.style.strokeDashoffset = `${c}`
  return anime({
    targets: circleEl,
    strokeDashoffset: [c, c * (1 - fraction)],
    duration,
    delay,
    easing: 'easeInOutQuart',
  })
}

export default anime
