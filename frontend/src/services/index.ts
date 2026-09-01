import type { CheckInRecord } from '../types'

const API_BASE_URL = 'http://127.0.0.1:8000'

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      Accept: 'application/json',
      ...options?.headers,
    },
    ...options,
  })

  if (!response.ok) {
    const errorBody = await response.json().catch(() => null)
    const details = Array.isArray(errorBody?.detail)
      ? errorBody.detail.map((entry: { msg?: string }) => entry.msg ?? 'Invalid request').join(', ')
      : (errorBody?.detail ?? 'Request failed.')

    throw new Error(details)
  }

  return (await response.json()) as T
}

export async function fetchCheckIns(): Promise<CheckInRecord[]> {
  return apiFetch<CheckInRecord[]>('/api/check-ins')
}

export async function createCheckIn(payload: {
  emotion: string
  confidence: number
  duration_seconds: number
}): Promise<CheckInRecord> {
  return apiFetch<CheckInRecord>('/api/check-ins', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  })
}
