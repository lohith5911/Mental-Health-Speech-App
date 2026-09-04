export type AppRoutePath =
  | '/'
  | '/login'
  | '/register'
  | '/dashboard'
  | '/check-in'
  | '/results'
  | '/history'
  | '/profile'
  | '/resources'

export type Emotion =
  | 'angry'
  | 'disgust'
  | 'fear'
  | 'happy'
  | 'neutral'
  | 'sad'

export type TrendStatus = 'stable' | 'change_detected' | 'significant_change'

export interface TrendInsight {
  status: string
  sample_size: number
  window_size: number
  baseline: Record<Emotion, number>
  recent: Record<Emotion, number>
  change: Record<Emotion, number>
  change_score: number
  trend: TrendStatus
  persistent_emotions: Emotion[]
}

export type NavItem = {
  label: string
  path: AppRoutePath
}

export const publicNavItems: NavItem[] = [
  { label: 'Home', path: '/' },
  { label: 'Daily check-in', path: '/check-in' },
  { label: 'Log in', path: '/login' },
  { label: 'Register', path: '/register' },
]

export type CheckInStatus =
  | 'idle'
  | 'recording'
  | 'recorded'
  | 'uploading'
  | 'analyzing'
  | 'success'
  | 'error'

export type EmotionAnalysisResult = {
  emotion: string
  confidence: number
  model_version: string
  probabilities: Record<string, number>
}

export type CheckInRecord = {
  id: number
  created_at: string
  emotion: string
  confidence: number
  duration_seconds: number
  model_version: string | null
  probabilities: Record<string, number> | null
}

export const MAX_CHECK_IN_SECONDS = 60

export const appNavItems: NavItem[] = [
  { label: 'Dashboard', path: '/dashboard' },
  { label: 'Daily check-in', path: '/check-in' },
  { label: 'Results', path: '/results' },
  { label: 'History', path: '/history' },
  { label: 'Profile', path: '/profile' },
  { label: 'Resources', path: '/resources' },
]
