import { useEffect, useRef, useState } from 'react'

/**
 * Act — one full-viewport movement of the scroll story.
 *
 * The old page was a centred column of glass cards stacked vertically; you
 * read it like a document.  An Act is the opposite: it owns the whole
 * viewport, its content is anchored off-centre, and it reveals and retreats
 * as you pass through it.  You do not read the page, you move through it.
 *
 * Each Act reports its own 0..1 travel progress, which is what lets type
 * enter on the way in and dim on the way out instead of appearing once and
 * sitting there.  The reveal is driven by IntersectionObserver rather than
 * scroll maths so it does not cost a listener per section.
 */
export default function Act({
  id,
  label,
  align = 'left',          // 'left' | 'center' | 'right'
  minH = '100vh',
  onProgress,
  children,
  style,
}) {
  const ref = useRef(null)
  const [seen, setSeen] = useState(false)

  useEffect(() => {
    const el = ref.current
    if (!el) return

    const io = new IntersectionObserver(
      ([e]) => { if (e.isIntersecting) setSeen(true) },
      { threshold: 0.18 },
    )
    io.observe(el)

    if (!onProgress) return () => io.disconnect()

    let ticking = false
    const read = () => {
      ticking = false
      const r = el.getBoundingClientRect()
      const vh = window.innerHeight || 1
      // 0 as the act's top reaches the viewport bottom, 1 as its bottom
      // leaves the top — the same span the reveal animates over.
      const p = (vh - r.top) / (vh + r.height)
      onProgress(Math.max(0, Math.min(1, p)))
    }
    const onScroll = () => {
      if (!ticking) { ticking = true; requestAnimationFrame(read) }
    }
    window.addEventListener('scroll', onScroll, { passive: true })
    window.addEventListener('resize', onScroll)
    read()
    return () => {
      io.disconnect()
      window.removeEventListener('scroll', onScroll)
      window.removeEventListener('resize', onScroll)
    }
  }, [onProgress])

  const justify = align === 'center' ? 'center'
                : align === 'right' ? 'flex-end' : 'flex-start'

  return (
    <section
      id={id}
      ref={ref}
      data-act={label}
      style={{
        position: 'relative',
        zIndex: 2,
        minHeight: minH,
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: justify,
        padding: 'clamp(24px, 7vw, 130px)',
        ...style,
      }}
    >
      <div
        className={seen ? 'act-in' : undefined}
        style={{
          width: '100%',
          maxWidth: 1180,
          opacity: seen ? undefined : 0,
          textAlign: align === 'center' ? 'center' : 'left',
        }}
      >
        {label && (
          <div className="ct-eyebrow">
            <span className="ct-eyebrow-rule" />
            {label}
          </div>
        )}
        {children}
      </div>
    </section>
  )
}
