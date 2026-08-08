import { useEffect, useRef, useState } from 'react'
import anime from '../utils/animefx'
import { scrollToTarget } from '../utils/smoothscroll'

/**
 * SectionRail — the scrollable menu, styled as a mixer channel strip.
 *
 * A fixed rail down the right edge: one channel per section, each with a
 * label, an LED, and a fader whose travel tracks how far you are THROUGH
 * that section.  Clicking a channel scrolls to it.
 *
 * Why a fader rather than a dot: on a workstation the fader position is
 * the thing you read at a glance, and it carries more information than a
 * highlight — you can see both which section you are in and how far
 * through it you are.
 */
export default function SectionRail({ sections }) {
  const [active, setActive] = useState(0)
  const railRef = useRef(null)
  const fillRefs = useRef([])
  const ledRefs = useRef([])

  useEffect(() => {
    // Entrance: channels drop in from the right, top to bottom.
    if (railRef.current) {
      const items = Array.from(railRef.current.children)
      anime.set(items, { opacity: 0, translateX: 24 })
      anime({
        targets: items,
        opacity: 1,
        translateX: 0,
        delay: anime.stagger(70, { start: 700 }),
        duration: 700,
        easing: 'easeOutCubic',
        complete: (a) => a.animatables.forEach(({ target }) => {
          target.style.transform = ''
          target.style.opacity = ''
        }),
      })
    }

    let ticking = false
    const update = () => {
      ticking = false
      const vh = window.innerHeight
      const mid = vh * 0.5
      let current = 0
      let bestDist = Infinity
      sections.forEach((s, i) => {
        const el = document.getElementById(s.id)
        if (!el) return
        const r = el.getBoundingClientRect()
        // Fraction of this section that has passed the viewport middle
        const through = Math.max(0, Math.min(1, (mid - r.top) / Math.max(1, r.height)))
        const fill = fillRefs.current[i]
        if (fill) fill.style.height = `${through * 100}%`

        // NEAREST section to the midline, not the one containing it.
        // Sections are separated by margins that belong to no section, so
        // strict containment leaves the midline homeless in every gap and
        // the active channel snaps back to the default.  Measured: at
        // y=1500 all four sections reported "not active" — engine ended at
        // 272px and the next began at 416px, with the midline at 360px.
        const dist = r.top > mid ? r.top - mid
                   : r.bottom < mid ? mid - r.bottom
                   : 0
        if (dist < bestDist) { bestDist = dist; current = i }
      })
      setActive(current)
    }
    const onScroll = () => {
      if (!ticking) { ticking = true; requestAnimationFrame(update) }
    }
    window.addEventListener('scroll', onScroll, { passive: true })
    window.addEventListener('resize', onScroll)
    update()
    return () => {
      window.removeEventListener('scroll', onScroll)
      window.removeEventListener('resize', onScroll)
    }
  }, [sections])

  // Pulse the LED of whichever channel just became active
  useEffect(() => {
    const led = ledRefs.current[active]
    if (led) anime({ targets: led, scale: [1, 1.9, 1], duration: 620,
                     easing: 'easeOutElastic(1, .6)' })
  }, [active])

  // Through Lenis, not scrollIntoView: with Lenis running, the browser's
  // own smooth-scroll animates the same scrollTop Lenis is animating and
  // the two fight, which reads as an overshoot-and-snap.  scrollToTarget
  // falls back to the native path when Lenis is off (reduced motion).
  const go = (id) => scrollToTarget(id, { offset: -80 })

  return (
    <div
      ref={railRef}
      className="hidden lg:flex"
      style={{
        position: 'fixed', right: 22, top: '50%', transform: 'translateY(-50%)',
        zIndex: 40, flexDirection: 'column', gap: 10,
      }}
      aria-label="Sections"
    >
      {sections.map((s, i) => {
        const on = i === active
        return (
          <button
            key={s.id}
            onClick={() => go(s.id)}
            title={s.label}
            style={{
              display: 'flex', alignItems: 'center', gap: 9,
              background: on ? `${s.color}12` : 'rgba(255,255,255,0.018)',
              border: `1px solid ${on ? s.color + '55' : 'rgba(255,255,255,0.06)'}`,
              borderRadius: 9,
              padding: '9px 11px 9px 9px',
              cursor: 'pointer',
              boxShadow: on ? `0 0 18px ${s.color}22, inset 0 1px 0 rgba(255,255,255,0.05)`
                            : 'inset 0 1px 0 rgba(255,255,255,0.03)',
              transition: 'background .35s, border-color .35s, box-shadow .35s',
              backdropFilter: 'blur(14px)',
            }}
          >
            {/* fader track + travel */}
            <span style={{
              position: 'relative', width: 3, height: 30, borderRadius: 2,
              background: 'rgba(255,255,255,0.08)', overflow: 'hidden', flexShrink: 0,
            }}>
              <span
                ref={(el) => { fillRefs.current[i] = el }}
                style={{
                  position: 'absolute', left: 0, right: 0, top: 0, height: '0%',
                  background: s.color, boxShadow: `0 0 7px ${s.color}`,
                  transition: 'height .12s linear',
                }}
              />
            </span>

            <span style={{
              fontSize: 8.5, fontWeight: 700, letterSpacing: '0.14em',
              fontFamily: 'ui-monospace,monospace',
              color: on ? s.color : 'rgba(255,255,255,0.30)',
              transition: 'color .35s', whiteSpace: 'nowrap',
            }}>
              {s.label}
            </span>

            <span
              ref={(el) => { ledRefs.current[i] = el }}
              style={{
                width: 5, height: 5, borderRadius: '50%', flexShrink: 0,
                background: on ? s.color : 'rgba(255,255,255,0.10)',
                boxShadow: on ? `0 0 8px ${s.color}` : 'none',
                transition: 'background .3s',
              }}
            />
          </button>
        )
      })}
    </div>
  )
}
