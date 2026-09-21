import { Navigate, Outlet, Route, Routes } from 'react-router-dom'
import AppLayout from './components/AppLayout'
import { useAuth } from './context/useAuth'
import DailyCheckIn from './pages/DailyCheckIn'
import Dashboard from './pages/Dashboard'
import History from './pages/History'
import Landing from './pages/Landing'
import Login from './pages/Login'
import Profile from './pages/Profile'
import Register from './pages/Register'
import Resources from './pages/Resources'
import Results from './pages/Results'

function ProtectedRoute() {
  const { user, loading } = useAuth()
  if (loading) return <div className="loading-state">Loading your account…</div>
  return user ? <AppLayout variant="app" /> : <Navigate to="/login" replace />
}

function PublicOnlyRoute() {
  const { user, loading } = useAuth()
  if (loading) return <div className="loading-state">Loading your account…</div>
  return user ? <Navigate to="/dashboard" replace /> : <Outlet />
}

function App() {
  return (
    <Routes>
      <Route element={<AppLayout variant="public" />}>
        <Route path="/" element={<Landing />} />
        <Route element={<PublicOnlyRoute />}>
          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register />} />
        </Route>
      </Route>

      <Route element={<ProtectedRoute />}>
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/check-in" element={<DailyCheckIn />} />
        <Route path="/results" element={<Results />} />
        <Route path="/history" element={<History />} />
        <Route path="/profile" element={<Profile />} />
        <Route path="/resources" element={<Resources />} />
      </Route>

      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}

export default App
