import { useState } from 'react'
import { NavLink } from 'react-router-dom'
import { appNavItems, publicNavItems } from '../types'

type NavbarProps = {
  variant: 'public' | 'app'
}

function Navbar({ variant }: NavbarProps) {
  const [isMenuOpen, setIsMenuOpen] = useState(false)
  const items = variant === 'public' ? publicNavItems : appNavItems

  function closeMenu() {
    setIsMenuOpen(false)
  }

  return (
    <header className="top-nav">
      <NavLink
        to={variant === 'app' ? '/dashboard' : '/'}
        className="brand"
        onClick={closeMenu}
      >
        <span className="brand-mark" aria-hidden="true">
          MH
        </span>
        <span className="brand-text">MindScreen</span>
      </NavLink>

      <button
        type="button"
        className="menu-toggle"
        aria-expanded={isMenuOpen}
        aria-controls="primary-nav"
        onClick={() => setIsMenuOpen((open) => !open)}
      >
        {isMenuOpen ? 'Close' : 'Menu'}
      </button>

      <nav id="primary-nav" className={isMenuOpen ? 'nav-links open' : 'nav-links'}>
        {items.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            className={({ isActive }) => (isActive ? 'nav-link active' : 'nav-link')}
            end={item.path === '/'}
            onClick={closeMenu}
          >
            {item.label}
          </NavLink>
        ))}
      </nav>
    </header>
  )
}

export default Navbar
