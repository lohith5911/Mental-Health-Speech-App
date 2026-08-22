import { Outlet } from 'react-router-dom'
import Navbar from './Navbar'

type AppLayoutProps = {
  variant: 'public' | 'app'
}

function AppLayout({ variant }: AppLayoutProps) {
  return (
    <div className={`app-shell ${variant}`}>
      <Navbar variant={variant} />
      <main className="main-content">
        <Outlet />
      </main>
      <footer className="app-footer">
        <p>
          Screening support only. This app is not a medical diagnosis or a
          replacement for professional care.
        </p>
      </footer>
    </div>
  )
}

export default AppLayout
