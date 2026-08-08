import axios from 'axios'

const BASE = import.meta.env.VITE_API_URL || ''

const api = axios.create({ baseURL: BASE })

export async function uploadAudio(file, onProgress) {
  const form = new FormData()
  form.append('file', file)
  const { data } = await api.post('/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (e) => onProgress && onProgress(Math.round((e.loaded * 100) / e.total)),
  })
  return data
}

export async function transcribeAudio(sessionId) {
  const { data } = await api.post(`/transcribe/${sessionId}`)
  return data
}

export async function separateStems(sessionId) {
  const { data } = await api.post(`/separate/${sessionId}`)
  return data
}

export async function segmentAudio(sessionId) {
  const { data } = await api.post(`/segment/${sessionId}`)
  return data
}

export async function transformArrangement(sessionId, targetOctaveRange = 5, nmfComponents = 16) {
  const { data } = await api.post(`/transform/${sessionId}`, {
    target_octave_range: targetOctaveRange,
    nmf_components: nmfComponents,
  })
  return data
}

export async function scoreFidelity(sessionId) {
  const { data } = await api.post(`/score/${sessionId}`)
  return data
}

/**
 * Render the two main listening tracks (original arrangement, transformed)
 * synchronously.  Returns when both WAVs are on disk.  Run from the
 * ProcessingPage after /transform so the Results page can play instantly.
 */
export async function renderPreviews(sessionId, force = false, only = null) {
  const params = {}
  if (force) params.force = 1
  // `only` renders a single track. Rendering both up front was 38% of the
  // whole upload-to-result wait, and you can only listen to one at a time.
  if (only) params.only = only
  const { data } = await api.post(`/previews/${sessionId}`, null, {
    params: Object.keys(params).length ? params : undefined,
  })
  return data
}

export async function getCdefText(sessionId) {
  const { data } = await api.get(`/export/${sessionId}/cdef-text`)
  return data.cdef
}

export function getOriginalAudioUrl(sessionId) {
  return `${BASE}/export/${sessionId}/audio/original`
}
export function getTranscriptionAudioUrl(sessionId) {
  return `${BASE}/export/${sessionId}/audio/transcription`
}
export function getTransformedAudioUrl(sessionId) {
  return `${BASE}/export/${sessionId}/audio/transformed`
}
// keep old name working
export function getAudioUrl(sessionId) {
  return getTransformedAudioUrl(sessionId)
}

export function getMidiUrl(sessionId) {
  return `${BASE}/export/${sessionId}/midi`
}

export function getPdfUrl(sessionId) {
  return `${BASE}/export/${sessionId}/pdf`
}

export function getMusicXmlUrl(sessionId) {
  return `${BASE}/export/${sessionId}/musicxml`
}

/**
 * Per-part audio URL.
 * @param {string} sessionId
 * @param {'transcription'|'transformed'} arrangement
 * @param {'bass'|'melody'|'voice'|'all'} part
 */
export function getPartAudioUrl(sessionId, arrangement, part) {
  return `${BASE}/export/${sessionId}/audio/${arrangement}/${part}`
}

/**
 * Per-layer audio URL.  Serves the real audio stem when present on disk;
 * otherwise the backend synthesizes the layer's notes via the SF2 sampler.
 * @param {string} sessionId
 * @param {'bass'|'bass_pattern'|'bass_melody'|'pattern'|'melody'} layer
 */
export function getLayerAudioUrl(sessionId, layer) {
  return `${BASE}/export/${sessionId}/audio/layer/${layer}`
}

/**
 * Fetch per-stem cluster metadata (how many Pattern-N rows to render under
 * each pattern layer).  Returns {} if separation hasn't produced clusters.
 */
export async function getPatternClusters(sessionId) {
  const { data } = await api.get(`/export/${sessionId}/pattern-clusters`)
  return data
}

/**
 * Audio URL for one cluster of one pattern stem.
 * @param {string} sessionId
 * @param {'pattern'|'bass_pattern'} layer
 * @param {number} clusterIdx
 */
export function getPatternClusterAudioUrl(sessionId, layer, clusterIdx) {
  return `${BASE}/export/${sessionId}/audio/pattern-cluster/${layer}/${clusterIdx}`
}

export function getCdefUrl(sessionId) {
  return `${BASE}/export/${sessionId}/cdef`
}

export async function cleanupSession(sessionId) {
  await api.delete(`/cleanup/${sessionId}`)
}
