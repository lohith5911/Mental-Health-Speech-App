import { Link } from 'react-router-dom'
import PageHeader from '../components/PageHeader'

function Login() {
  return (
    <section className="page narrow-page">
      <PageHeader
        title="Log in"
        description="Account access will be connected later. This page is a layout placeholder only."
      />

      <form
        className="form-card"
        onSubmit={(event) => {
          event.preventDefault()
        }}
      >
        <label htmlFor="login-email">Email</label>
        <input id="login-email" name="email" type="email" autoComplete="email" />

        <label htmlFor="login-password">Password</label>
        <input
          id="login-password"
          name="password"
          type="password"
          autoComplete="current-password"
        />

        <button className="button primary" type="submit" disabled>
          Log in (coming later)
        </button>

        <p className="form-note">
          Need an account? <Link to="/register">Register</Link>
        </p>
      </form>
    </section>
  )
}

export default Login
