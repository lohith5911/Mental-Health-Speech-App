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
