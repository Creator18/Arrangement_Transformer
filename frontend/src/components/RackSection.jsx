import { useEffect, useRef } from 'react'
import anime from '../utils/animefx'
import { whenInView, onScrollScrub } from '../utils/scrollfx'

/**
 * RackSection — one unit in the rack.
 *
 * Gives every section the same workstation furniture (screw-corners, a
 * channel label, an activity LED) and the same scroll behaviour: the
 * header rules draw outward, the body rises in, and the whole unit gets a
 * gentle parallax so sections feel like physical panels passing a window
 * rather than a scrolling document.
 */
export default function RackSection({
  id, label, color = '#f59e0b', title, subtitle, children,
  parallax = 14, className = '', style,
}) {
  const rootRef = useRef(null)
  const ruleL = useRef(null)
  const ruleR = useRef(null)
  const bodyRef = useRef(null)
  const ledRef = useRef(null)

  useEffect(() => {
    const root = rootRef.current
    if (!root) return

    const stopView = whenInView(root, () => {
      // Header rules draw outward from the label
      anime({
        targets: [ruleL.current, ruleR.current].filter(Boolean),
        scaleX: [0, 1],
        duration: 900,
        easing: 'easeOutQuart',
      })
      if (bodyRef.current) {
        anime({
          targets: bodyRef.current,
          opacity: [0, 1],
          translateY: [26, 0],
          duration: 780,
          delay: 110,
          easing: 'easeOutCubic',
          complete: () => {
            bodyRef.current.style.transform = ''
            bodyRef.current.style.opacity = ''
          },
        })
      }
      if (ledRef.current) {
        anime({ targets: ledRef.current, scale: [1, 2.1, 1], duration: 700,
                easing: 'easeOutElastic(1, .6)' })
      }
    }, { threshold: 0.22 })

    // Panel parallax — small, or it reads as drift rather than depth.
    const stopScrub = onScrollScrub(root, (p) => {
      root.style.setProperty('--rack-shift', `${(p - 0.5) * -parallax}px`)
    })

    return () => { stopView(); stopScrub() }
  }, [parallax])

  return (
    <section
      id={id}
      ref={rootRef}
      className={`relative ${className}`}
      style={{ transform: 'translateY(var(--rack-shift, 0px))', ...style }}
    >
      {/* header */}
      <div className="flex items-center gap-3 mb-5">
        <span
          ref={ruleL}
          style={{ height: 1, flex: 1, transformOrigin: 'right center',
                   background: `linear-gradient(90deg, transparent, ${color}44)` }}
        />
        <span className="flex items-center gap-2" style={{ flexShrink: 0 }}>
          <span ref={ledRef} style={{
            width: 5, height: 5, borderRadius: '50%', background: color,
            boxShadow: `0 0 8px ${color}`, display: 'inline-block',
          }} />
          <span style={{
            fontSize: 9, fontWeight: 700, letterSpacing: '0.22em',
            color: `${color}cc`, fontFamily: 'ui-monospace,monospace',
          }}>
            {label}
          </span>
        </span>
        <span
          ref={ruleR}
          style={{ height: 1, flex: 1, transformOrigin: 'left center',
                   background: `linear-gradient(90deg, ${color}44, transparent)` }}
        />
      </div>

      <div ref={bodyRef} style={{ opacity: 0 }}>
        {(title || subtitle) && (
          <div className="text-center mb-8">
            {title && (
              <h2 className="text-2xl font-bold tracking-tight text-gradient">{title}</h2>
            )}
            {subtitle && (
              <p style={{ fontSize: 12.5, color: 'rgba(255,255,255,0.34)', marginTop: 6 }}>
                {subtitle}
              </p>
            )}
          </div>
        )}
        {children}
      </div>
    </section>
  )
}
