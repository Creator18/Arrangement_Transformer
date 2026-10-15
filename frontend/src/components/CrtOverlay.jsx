/**
 * CrtOverlay — a subtle CRT tube laid over every page.
 *
 * Inspired by ThreeUI's CrtBackground, written fresh for this app: instead of
 * replacing the page with a WebGL screen, it sits ON TOP of the existing look as
 * a stack of fixed, click-through CSS layers, so the design underneath is
 * unchanged and nothing has to re-render.
 *
 *   scanlines   1 px dark lines on a fixed 3 px pitch (CSS px, so they stay crisp
 *               at any device pixel ratio)
 *   grille      a faint red/green/blue aperture-grille stripe, like a phosphor mask
 *   glass       darkened corners + a soft inner edge + a faint sheen: the curved tube
 *   roll bar    a slow, faint brighter band drifting down the screen
 *   flicker     a barely-there brightness flutter
 *
 * The roll bar and flicker stop for prefers-reduced-motion.  Strength is one
 * number (STRENGTH below) so the whole effect can be dialled up or down.
 */
const STRENGTH = 1   // 0 = off, 1 = the tuned subtle look

export default function CrtOverlay() {
  if (!STRENGTH) return null
  return (
    <div className="crt-overlay" aria-hidden="true" style={{ '--crt': STRENGTH }}>
      <div className="crt-scanlines" />
      <div className="crt-grille" />
      <div className="crt-rollbar" />
      <div className="crt-glass" />
    </div>
  )
}
