import { Link } from 'react-router-dom'
import PageHeader from '../components/PageHeader'

function Register() {
  return (
    <section className="page narrow-page">
      <PageHeader
        title="Register"
        description="New accounts will be created in a later step. This page is a layout placeholder only."
      />

      <form
        className="form-card"
        onSubmit={(event) => {
          event.preventDefault()
        }}
      >
        <label htmlFor="register-name">Full name</label>
        <input id="register-name" name="name" type="text" autoComplete="name" />

        <label htmlFor="register-email">Email</label>
        <input
          id="register-email"
          name="email"
          type="email"
          autoComplete="email"
        />

        <label htmlFor="register-password">Password</label>
        <input
          id="register-password"
          name="password"
          type="password"
          autoComplete="new-password"
        />

        <button className="button primary" type="submit" disabled>
          Create account (coming later)
        </button>

        <p className="form-note">
          Already registered? <Link to="/login">Log in</Link>
        </p>
      </form>
    </section>
  )
}

export default Register
