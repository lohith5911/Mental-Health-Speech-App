import { Link } from 'react-router-dom'
import PageHeader from '../components/PageHeader'

function Landing() {
  return (
    <section className="page landing-page">
      <PageHeader
        title="AI-powered mental health screening through daily speech"
        description="A calm space to check in with your voice, review speech-derived signals, and notice personal patterns over time."
      />

      <div className="card-grid">
        <article className="info-card">
          <h2>Daily check-in</h2>
          <p>
            Speak naturally for a short daily check-in. The app analyzes
            emotion signals, acoustic measurements, and recording quality.
          </p>
        </article>
        <article className="info-card">
          <h2>Private by design</h2>
          <p>
            Your authenticated account keeps your check-ins and longitudinal
            history separate from other users.
          </p>
        </article>
        <article className="info-card">
          <h2>Support, not diagnosis</h2>
          <p>
            Use the results for self-monitoring and reflection. They do not
            diagnose depression or any other mental-health condition.
          </p>
        </article>
      </div>

      <div className="cta-row">
        <Link className="button primary" to="/register">
          Create an account
        </Link>
        <Link className="button secondary" to="/login">
          Log in
        </Link>
        <Link className="button ghost" to="/dashboard">
          View app pages
        </Link>
      </div>
    </section>
  )
}

export default Landing
