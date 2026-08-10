import { useRef } from 'react'

/**
 * StageCarousel — the pipeline as a flat sliding row of words.
 *
 * Was a true 3D ring.  The centre word read beautifully but the flanking
 * ones did not: rotateY puts them on a plane turned away from the camera, so
 * perspective SKEWS the letterforms — one edge of the word is nearer than
 * the other and the type shears.  Type is the whole object here, and sheared
 * type just looks broken rather than distant.
 *
 * So depth is gone and distance is expressed the way it should have been:
 * the row slides, and neighbours fade and shrink.  Every word stays in the
 * picture plane, perfectly flat and unskewed, and the ranking comes from
 * opacity, scale and blur instead of rotation.  Same reading — past behind,
 * next ahead, present centred — without deforming a single glyph.
 */

const SPACING = 300         // px between adjacent stage words
const VISIBLE = 2           // how many either side stay on screen

export default function StageCarousel({ items, activeIndex = 0, accent }) {
  const n = Math.max(1, items.length)

  return (
    <div style={{
      position: 'relative', height: 150, overflow: 'hidden',
      // Break out of the narrow column this sits in.  Processing wraps its
      // content at 480px, but SPACING puts the neighbouring stage 300px from
      // centre — so "Transcribe" was rendered at x=540 inside a 480px box and
      // clipped to a single "T".  It was never invisible, it was outside.
      // The row is conceptually full-bleed, so it takes the viewport width.
      width: '100vw', marginLeft: 'calc(50% - 50vw)',
    }}>
      {items.map((it, i) => {
        const d = i - activeIndex                    // signed distance in slots
        const ad = Math.abs(d)
        const isNow = d === 0
        if (ad > VISIBLE + 0.5) {
          return null                                // far items are not rendered
        }
        // Fade and shrink with distance. No rotation anywhere, so the
        // letterforms stay flat whatever slot they are in.
        //
        // These values were guesses and they were far too timid: neighbours
        // rendered at ~0.25 opacity, blurred and scaled to 0.72 against a
        // near-black page, which made them invisible.  The carousel then read
        // as ONE word floating alone — no sense of a row, no sense of moving
        // through stages, which is the only thing this component exists to
        // show.  Seen for the first time by screenshotting the page.
        const fade = Math.max(0, 1 - ad / (VISIBLE + 1.4))
        return (
          <div
            key={it.key}
            style={{
              position: 'absolute',
              left: '50%',
              top: '50%',
              width: SPACING,
              marginLeft: -SPACING / 2,
              marginTop: -32,
              textAlign: 'center',
              transform: `translateX(${d * SPACING}px) scale(${isNow ? 1 : 0.82})`,
              opacity: isNow ? 1 : fade * 0.95,
              filter: isNow ? 'none' : `blur(${ad * 0.5}px)`,
              // One long weighted ease so the row glides as a whole when the
              // stage advances, rather than each word animating on its own.
              transition:
                'transform 1000ms cubic-bezier(0.16,1,0.3,1),' +
                'opacity 800ms ease, filter 800ms ease',
              pointerEvents: 'none',
              willChange: 'transform, opacity',
            }}
          >
            <div
              style={{
                fontSize: 38,
                fontWeight: 300,
                letterSpacing: '-0.02em',
                whiteSpace: 'nowrap',
                color: isNow ? (accent || 'var(--ct-accent)') : 'rgba(255,255,255,0.85)',
                textShadow: isNow ? `0 0 38px ${accent || 'var(--ct-accent)'}55` : 'none',
                transition: 'color 700ms',
              }}
            >
              {it.label}
            </div>
            {isNow && (
              <div
                style={{
                  marginTop: 10,
                  fontFamily: 'ui-monospace,monospace',
                  fontSize: 9,
                  letterSpacing: '0.30em',
                  textTransform: 'uppercase',
                  color: 'rgba(255,255,255,0.34)',
                }}
              >
                Running
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}
