import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import PageHeader from '../components/PageHeader'
import { useAuth } from '../context/useAuth'

function Register() {
  const navigate = useNavigate()
  const { register } = useAuth()
  const [displayName, setDisplayName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setError('')
    if (!displayName.trim() || !email.trim() || !password) {
      setError('Enter your name, email, and password.')
      return
    }
    if (password.length < 8) {
      setError('Your password must be at least 8 characters.')
      return
    }

    setIsSubmitting(true)
    try {
      await register(email.trim(), displayName.trim(), password)
      navigate('/dashboard', { replace: true })
    } catch (requestError) {
      setError(requestError instanceof Error && requestError.message.includes('already')
        ? 'An account with this email already exists.'
        : 'We could not create your account. Check your details and try again.')
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <section className="page narrow-page">
      <PageHeader
        title="Register"
        description="Create a private account for your check-ins and insights."
      />

      <form
        className="form-card"
        onSubmit={handleSubmit}
      >
        <label htmlFor="register-name">Full name</label>
        <input id="register-name" name="name" type="text" autoComplete="name" value={displayName} onChange={(event) => setDisplayName(event.target.value)} />

        <label htmlFor="register-email">Email</label>
        <input
          id="register-email"
          name="email"
          type="email"
          autoComplete="email"
          value={email}
          onChange={(event) => setEmail(event.target.value)}
        />

        <label htmlFor="register-password">Password</label>
        <input
          id="register-password"
          name="password"
          type="password"
          autoComplete="new-password"
          value={password}
          onChange={(event) => setPassword(event.target.value)}
        />

        {error && <p className="form-error" role="alert">{error}</p>}

        <button className="button primary" type="submit" disabled={isSubmitting}>
          {isSubmitting ? 'Creating account…' : 'Create account'}
        </button>

        <p className="form-note">
          Already registered? <Link to="/login">Log in</Link>
        </p>
      </form>
    </section>
  )
}

export default Register
