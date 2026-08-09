/**
 * hands.js — which notes two human hands could actually be HOLDING.
 *
 * The model, and why it is shaped this way
 * -----------------------------------------
 * A ten-note ceiling is not enough, because ten fingers cannot be anywhere
 * they like.  Each hand covers a contiguous stretch of keyboard bounded by
 * its own reach:
 *
 *   RIGHT HAND — the LOWEST note is the thumb, and the hand opens upward to
 *   the pinky.  Thumb on C, pinky comfortably reaching D an octave above, is
 *   14 semitones; a larger hand gets to E, which is 16.
 *
 *   LEFT HAND — mirrored: the LOWEST note is the pinky, and the hand closes
 *   upward to the thumb.  Same reach.
 *
 * So each hand is at most 5 notes inside a window of MAX_HAND_SPAN
 * semitones, and the two windows do not interleave — hands sit side by side
 * on the keyboard, low hand then high hand.  That is why the search below
 * only considers CONTIGUOUS splits of the sorted pitches: any assignment
 * where the hands interlock is not a thing hands do here.
 *
 * What happens to notes that do not fit
 * --------------------------------------
 * They are still sounding, they are simply not being held — which is exactly
 * how the instrument works.  You strike a heavy bass note, catch it with the
 * sustain pedal, lift the hand away and play the upper register; the bass is
 * still ringing under the melody with no finger on it.  Modelling this as
 * "the key is down" was the mistake: it drew hands that could not exist.
 *
 * Preference order is by RECENCY, because that is what a player does — the
 * note you release to free a finger is the one you struck longest ago.
 */

// Thumb-to-pinky reach in semitones. 14 = C up to the D an octave above,
// which is the span most adult hands manage without strain. Raise toward 16
// to model a larger hand.
export const MAX_HAND_SPAN = 14
export const MAX_PER_HAND = 5
export const MAX_KEYS_DOWN = MAX_PER_HAND * 2

/**
 * Can two hands hold exactly this set of pitches at once?
 * `sorted` must be ascending and free of duplicates.
 */
export function feasible(sorted) {
  const n = sorted.length
  if (n === 0) return true
  if (n > MAX_KEYS_DOWN) return false

  // k = how many of the low notes belong to the left hand.
  for (let k = 0; k <= n; k++) {
    const lhCount = k
    const rhCount = n - k
    if (lhCount > MAX_PER_HAND || rhCount > MAX_PER_HAND) continue
    if (lhCount > 1 && sorted[k - 1] - sorted[0] > MAX_HAND_SPAN) continue
    if (rhCount > 1 && sorted[n - 1] - sorted[k] > MAX_HAND_SPAN) continue
    return true
  }
  return false
}

/**
 * From the notes sounding at an instant, return the ones a player could
 * plausibly still have their fingers on.
 *
 * @param {Array<{pitch:number, onset:number}>} sounding
 * @returns {Map<number, number>} pitch -> onset, for held keys only
 */
export function heldKeys(sounding) {
  // Most recently struck first — the newest note is the one a hand is
  // certainly on, and older ones are the candidates for release.
  const byRecency = [...sounding].sort((a, b) => b.onset - a.onset)

  const kept = new Map()          // pitch -> onset
  const pitches = []              // kept pitches, kept sorted ascending

  for (const { pitch, onset } of byRecency) {
    if (kept.has(pitch)) continue           // same key re-listed: newest wins

    // insert into the sorted array
    let i = 0
    while (i < pitches.length && pitches[i] < pitch) i++
    pitches.splice(i, 0, pitch)

    if (feasible(pitches)) {
      kept.set(pitch, onset)
    } else {
      pitches.splice(i, 1)                  // put it back; this one sustains
    }
    if (kept.size >= MAX_KEYS_DOWN) break
  }
  return kept
}
