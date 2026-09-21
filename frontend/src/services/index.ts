import type {
  AcousticFeatures,
  AnalyzeAndSaveResponse,
  AuthResponse,
  CheckInRecord,
  TrendInsight,
} from '../types'

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/$/, '')
export const ACCESS_TOKEN_STORAGE_KEY = 'mindtrace_access_token'

let unauthorizedHandler: (() => void) | null = null

export function setUnauthorizedHandler(handler: (() => void) | null): void {
  unauthorizedHandler = handler
}

export function getStoredToken(): string | null {
  return sessionStorage.getItem(ACCESS_TOKEN_STORAGE_KEY)
}

export function extractApiError(errorBody: unknown): string {
  if (typeof errorBody !== 'object' || errorBody === null) return 'Request failed.'

  const detail = (errorBody as { detail?: unknown }).detail
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    return detail
      .map((entry) => {
        if (typeof entry === 'object' && entry !== null && 'msg' in entry) {
          return String((entry as { msg: unknown }).msg)
        }
        return 'Invalid request'
      })
      .join(', ')
  }

  return 'Request failed.'
}

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const token = getStoredToken()
  const headers = new Headers(options?.headers)
  headers.set('Accept', 'application/json')
  if (token) headers.set('Authorization', `Bearer ${token}`)

  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers,
  })

  if (!response.ok) {
    const errorBody = await response.json().catch(() => null)
    if (response.status === 401) unauthorizedHandler?.()
    throw new Error(response.status === 401 ? 'Authentication failed.' : extractApiError(errorBody))
  }

  return (await response.json()) as T
}

export function loginUser(email: string, password: string): Promise<AuthResponse> {
  return apiFetch<AuthResponse>('/api/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  })
}

export function registerUser(
  email: string,
  displayName: string,
  password: string,
): Promise<AuthResponse> {
  return apiFetch<AuthResponse>('/api/auth/register', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, display_name: displayName, password }),
  })
}

export function fetchCurrentUser() {
  return apiFetch<AuthResponse['user']>('/api/auth/me')
}

export async function fetchCheckIns(): Promise<CheckInRecord[]> {
  return apiFetch<CheckInRecord[]>('/api/check-ins')
}

export async function fetchTrendInsights(windowSize?: number): Promise<TrendInsight> {
  const query = windowSize === undefined ? '' : `?window_size=${windowSize}`
  return apiFetch<TrendInsight>(`/api/insights/trends${query}`)
}

export async function createCheckIn(payload: {
  emotion: string
  confidence: number
  duration_seconds: number
  model_version?: string
  probabilities?: Record<string, number>
  acoustic_features?: AcousticFeatures
}): Promise<CheckInRecord> {
  return apiFetch<CheckInRecord>('/api/check-ins', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  })
}

function recordingExtension(blob: Blob): string {
  if (blob.type.includes('webm')) return 'webm'
  if (blob.type.includes('mp4')) return 'mp4'
  if (blob.type.includes('ogg')) return 'ogg'
  if (blob.type.includes('wav')) return 'wav'
  throw new Error('This browser produced an unsupported audio format. Please try again.')
}

export async function analyzeAndSaveCheckIn(
  blob: Blob,
  clientDurationSeconds: number,
): Promise<AnalyzeAndSaveResponse> {
  const formData = new FormData()
  formData.append(
    'file',
    blob,
    `check-in-${Date.now()}.${recordingExtension(blob)}`,
  )
  formData.append('client_duration_seconds', String(clientDurationSeconds))

  const idempotencyKey =
    typeof crypto.randomUUID === 'function'
      ? crypto.randomUUID()
      : `check-in-${Date.now()}-${Math.random().toString(36).slice(2)}`

  return apiFetch<AnalyzeAndSaveResponse>('/api/check-ins/analyze-and-save', {
    method: 'POST',
    headers: {
      'Idempotency-Key': idempotencyKey,
    },
    body: formData,
  })
}
