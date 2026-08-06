import { useEffect, useRef, useMemo } from 'react'
import anime from '../utils/animefx'
import { onScrollScrub } from '../utils/scrollfx'

/**
 * KeyboardAssembly — two octaves of piano keys that fly in scattered and
 * assemble into a keyboard strip under the hero (our take on the anime.js
 * site's exploded-lens sequence), then gently spread apart again as you
 * scroll past ("exploded view" scrub).
 */

const WHITE_W = 26
const WHITE_H = 92
const BLACK_W = 15
const BLACK_H = 56

// Two octaves: white-key indices 0..13, black keys after white keys
// 0,1,3,4,5 within each octave (C# D# F# G# A#).
const BLACK_AFTER = new Set([0, 1, 3, 4, 5, 7, 8, 10, 11, 12])

export default function KeyboardAssembly() {
  const rootRef  = useRef(null)
  const whiteRef = useRef(null)
  const blackRef = useRef(null)

  const whites = useMemo(() => Array.from({ length: 14 }, (_, i) => ({
    x: i * (WHITE_W + 2),
  })), [])

  const blacks = useMemo(() => (
    Array.from({ length: 14 }, (_, i) => i)
      .filter(i => BLACK_AFTER.has(i))
      .map(i => ({ x: i * (WHITE_W + 2) + WHITE_W - BLACK_W / 2 + 1 }))
  ), [])

  useEffect(() => {
    const root = rootRef.current
    if (!root) return
    const whiteEls = whiteRef.current ? Array.from(whiteRef.current.children) : []
    const blackEls = blackRef.current ? Array.from(blackRef.current.children) : []
    const keys = [...whiteEls, ...blackEls]

    // Scatter, then assemble with an elastic settle.
    keys.forEach(el => {
      el.style.opacity = '0'
      el.style.transform =
        `translateY(${anime.random(-160, -60)}px) ` +
        `translateX(${anime.random(-40, 40)}px) ` +
        `rotate(${anime.random(-35, 35)}deg)`
    })
    anime({
      targets: keys,
      opacity: 1,
      translateY: 0,
      translateX: 0,
      rotate: 0,
      delay: anime.stagger(34, { start: 550 }),
      duration: 1100,
      easing: 'easeOutElastic(1, .65)',
      complete: () => keys.forEach(el => { el.style.transform = '' }),
    })

    // Scroll scrub: exploded-view spread.  Keys drift apart horizontally
    // (odd keys up, even keys down a touch) as the strip leaves the
    // viewport — subtle, so it reads as depth rather than chaos.
    let settled = false
    const settleTimer = setTimeout(() => { settled = true }, 2200)
    const stopScrub = onScrollScrub(root, (p) => {
      if (!settled) return
      const spread = Math.max(0, p - 0.55) * 2.2     // only past mid-viewport
      keys.forEach((el, i) => {
        const dir = i % 2 === 0 ? 1 : -1
        el.style.transform =
          `translateY(${dir * spread * 18}px) translateX(${(i - keys.length / 2) * spread * 3}px)`
      })
    })
    return () => { clearTimeout(settleTimer); stopScrub() }
  }, [])

  const totalW = 14 * (WHITE_W + 2)

  return (
    <div ref={rootRef}
         style={{ width: totalW, height: WHITE_H + 8, position: 'relative', margin: '0 auto' }}
         aria-hidden>
      {/* White keys */}
      <div ref={whiteRef} style={{ position: 'absolute', inset: 0 }}>
        {whites.map((k, i) => (
          <div key={i} style={{
            position: 'absolute',
            left: k.x, top: 0,
            width: WHITE_W, height: WHITE_H,
            borderRadius: '0 0 4px 4px',
            background: 'linear-gradient(180deg, rgba(255,255,255,0.92), rgba(226,226,232,0.88))',
            boxShadow: 'inset 0 -6px 10px rgba(0,0,0,0.18), 0 4px 14px rgba(0,0,0,0.45)',
            willChange: 'transform',
          }} />
        ))}
      </div>
      {/* Black keys */}
      <div ref={blackRef} style={{ position: 'absolute', inset: 0 }}>
        {blacks.map((k, i) => (
          <div key={i} style={{
            position: 'absolute',
            left: k.x, top: 0,
            width: BLACK_W, height: BLACK_H,
            borderRadius: '0 0 3px 3px',
            background: 'linear-gradient(180deg, #26262c, #0c0c10)',
            boxShadow: '0 4px 10px rgba(0,0,0,0.6), inset 0 -3px 6px rgba(255,255,255,0.06)',
            willChange: 'transform',
          }} />
        ))}
      </div>
      {/* Amber reflection line under the strip */}
      <div style={{
        position: 'absolute', left: '6%', right: '6%', bottom: -4, height: 1,
        background: 'linear-gradient(90deg, transparent, rgba(245,158,11,0.35), transparent)',
      }} />
    </div>
  )
}
