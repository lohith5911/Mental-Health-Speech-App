import { useCallback, useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  ACCESS_TOKEN_STORAGE_KEY,
  fetchCurrentUser,
  getStoredToken,
  loginUser,
  registerUser,
  setUnauthorizedHandler,
} from '../services'
import type { User } from '../types'
import { AuthContext } from './AuthContextDefinition'

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const navigate = useNavigate()
  const [token, setToken] = useState<string | null>(() => getStoredToken())
  const [user, setUser] = useState<User | null>(null)
  const [loading, setLoading] = useState(() => getStoredToken() !== null)
  const initializationStarted = useRef(false)

  const clearAuthentication = useCallback((redirect = true) => {
    sessionStorage.removeItem(ACCESS_TOKEN_STORAGE_KEY)
    setToken(null)
    setUser(null)
    if (redirect && window.location.pathname !== '/login') navigate('/login', { replace: true })
  }, [navigate])

  useEffect(() => {
    setUnauthorizedHandler(() => clearAuthentication())
    return () => setUnauthorizedHandler(null)
  }, [clearAuthentication])

  useEffect(() => {
    if (initializationStarted.current) return
    initializationStarted.current = true

    const storedToken = getStoredToken()
    if (!storedToken) return

    fetchCurrentUser()
      .then((currentUser) => setUser(currentUser))
      .catch(() => clearAuthentication(false))
      .finally(() => setLoading(false))
  }, [clearAuthentication])

  async function login(email: string, password: string) {
    const response = await loginUser(email, password)
    sessionStorage.setItem(ACCESS_TOKEN_STORAGE_KEY, response.access_token)
    setToken(response.access_token)
    setUser(response.user)
  }

  async function register(email: string, displayName: string, password: string) {
    const response = await registerUser(email, displayName, password)
    sessionStorage.setItem(ACCESS_TOKEN_STORAGE_KEY, response.access_token)
    setToken(response.access_token)
    setUser(response.user)
  }

  function logout() {
    clearAuthentication(false)
    navigate('/login', { replace: true })
  }

  return (
    <AuthContext.Provider value={{ user, token, loading, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  )
}
