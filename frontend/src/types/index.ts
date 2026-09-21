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

export type User = {
  id: number
  email: string
  display_name: string
  created_at: string
}

export type AuthResponse = {
  access_token: string
  token_type: string
  user: User
}

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
  acoustic_features: AcousticFeatures
}

export type CheckInQualityStatus =
  | 'usable'
  | 'low_signal'
  | 'mostly_silent'
  | 'noisy_signal'
  | 'insufficient_audio'

export type CheckInQuality = {
  status: CheckInQualityStatus
  reasons: string[]
}

export type AcousticFeatures = {
  duration_seconds: number
  rms_mean: number
  rms_std: number
  zcr_mean: number
  zcr_std: number
  pitch_mean_hz: number | null
  pitch_std_hz: number | null
  pitch_range_hz: number | null
  silence_ratio: number
  speaking_rate_proxy: number | null
  is_silent: boolean
  is_noisy: boolean
}

export type CheckInRecord = {
  id: number
  created_at: string
  emotion: string
  confidence: number
  duration_seconds: number
  model_version: string | null
  probabilities: Record<string, number> | null
  acoustic_features: AcousticFeatures | null
  quality?: CheckInQuality | null
}

export type AnalyzeAndSaveResponse = CheckInRecord & {
  model_version: string
  probabilities: Record<string, number>
  acoustic_features: AcousticFeatures
  quality: CheckInQuality
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
