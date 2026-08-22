import { Link } from 'react-router-dom'
import PageHeader from '../components/PageHeader'

function Landing() {
  return (
    <section className="page landing-page">
      <PageHeader
        title="AI-powered mental health screening through daily speech"
        description="A calm space to check in with your voice over time. Screens and history will appear here after later features are connected."
      />

      <div className="card-grid">
        <article className="info-card">
          <h2>Daily check-in</h2>
          <p>
            Speak a short update each day. Recording and analysis are not
            enabled in this foundation build.
          </p>
        </article>
        <article className="info-card">
          <h2>Private by design</h2>
          <p>
            Sign-in, storage, and screening results will be added in later
            steps. Nothing is saved yet.
          </p>
        </article>
        <article className="info-card">
          <h2>Support, not diagnosis</h2>
          <p>
            The app will help you notice patterns and find resources. It will
            not replace a clinician.
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
