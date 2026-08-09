import { useEffect, useRef } from 'react'

/**
 * StageCarousel — the pipeline as a rotating ring of words.
 *
 * Replaces the bubble cluster on Processing.  Two things were wrong with
 * that: a grid of glyph icons asks you to decode six little symbols that
 * carry no real meaning, and a static cluster gives the current stage no
 * more presence than the five that are not happening.
 *
 * Here the stages sit on a circle in real 3D and the ring TURNS so the
 * active one arrives at the front.  Progress is therefore motion rather
 * than a colour change — the machine visibly advances, which is the one
 * thing this screen exists to communicate.  No icons at all: the stage
 * name, set large, is the object.
 *
 * Depth does the ranking for free.  Items facing away are further from the
 * camera, so perspective shrinks them, and opacity and blur follow their
 * facing angle — the past and future stages are legible as context without
 * competing with the present one.
 */

const RADIUS = 260          // px; ring radius in 3D space
const ITEM_W = 260

export default function StageCarousel({ items, activeIndex = 0, accent }) {
  const ringRef = useRef(null)
  const n = Math.max(1, items.length)
  const step = 360 / n

  useEffect(() => {
    const el = ringRef.current
    if (!el) return
    // Unwrapped rotation: keep turning in the same direction rather than
    // spinning backwards the long way when the index wraps.
    el.style.transform = `translateZ(-${RADIUS}px) rotateY(${-activeIndex * step}deg)`
  }, [activeIndex, step])

  return (
    <div
      style={{
        perspective: 1100,
        perspectiveOrigin: '50% 50%',
        height: 210,
        width: '100%',
        position: 'relative',
      }}
    >
      <div
        ref={ringRef}
        style={{
          position: 'absolute',
          inset: 0,
          transformStyle: 'preserve-3d',
          // Long, heavy ease — the ring should feel weighted, like something
          // being turned, not snapped.
          transition: 'transform 1100ms cubic-bezier(0.16, 1, 0.3, 1)',
        }}
      >
        {items.map((it, i) => {
          // Shortest signed angular distance from the front position.
          let d = ((i - activeIndex) % n + n) % n
          if (d > n / 2) d -= n
          const facing = Math.cos((d * step * Math.PI) / 180)   // 1 = front
          const isNow = d === 0
          return (
            <div
              key={it.key}
              style={{
                position: 'absolute',
                left: '50%',
                top: '50%',
                width: ITEM_W,
                marginLeft: -ITEM_W / 2,
                marginTop: -34,
                transform: `rotateY(${i * step}deg) translateZ(${RADIUS}px)`,
                textAlign: 'center',
                opacity: Math.max(0, facing) ** 1.6 * (isNow ? 1 : 0.5),
                filter: isNow ? 'none' : `blur(${(1 - Math.max(0, facing)) * 3.2}px)`,
                transition: 'opacity 900ms ease, filter 900ms ease',
                pointerEvents: 'none',
              }}
            >
              <div
                style={{
                  fontSize: isNow ? 34 : 24,
                  fontWeight: 300,
                  letterSpacing: '-0.02em',
                  color: isNow ? (accent || 'var(--ct-accent)') : 'rgba(255,255,255,0.7)',
                  textShadow: isNow
                    ? `0 0 34px ${accent || 'var(--ct-accent)'}66`
                    : 'none',
                  transition: 'font-size 700ms cubic-bezier(0.16,1,0.3,1), color 700ms',
                  whiteSpace: 'nowrap',
                }}
              >
                {it.label}
              </div>
              <div
                style={{
                  marginTop: 8,
                  fontFamily: 'ui-monospace,monospace',
                  fontSize: 9,
                  letterSpacing: '0.28em',
                  textTransform: 'uppercase',
                  color: 'rgba(255,255,255,0.30)',
                }}
              >
                {it.done ? 'Complete' : isNow ? 'Running' : `0${i + 1}`}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
