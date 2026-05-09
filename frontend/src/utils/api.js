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

export async function transformArrangement(sessionId, targetOctaveRange = 5) {
  const { data } = await api.post(`/transform/${sessionId}`, { target_octave_range: targetOctaveRange })
  return data
}

export async function scoreFidelity(sessionId) {
  const { data } = await api.post(`/score/${sessionId}`)
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

export function getCdefUrl(sessionId) {
  return `${BASE}/export/${sessionId}/cdef`
}

export async function cleanupSession(sessionId) {
  await api.delete(`/cleanup/${sessionId}`)
}
