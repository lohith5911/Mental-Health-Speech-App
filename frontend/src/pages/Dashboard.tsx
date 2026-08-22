import { Link } from 'react-router-dom'
import PageHeader from '../components/PageHeader'

function Dashboard() {
  return (
    <section className="page">
      <PageHeader
        title="Dashboard"
        description="Your overview of check-ins and screens will live here. No health data is shown yet."
      />

      <div className="card-grid">
        <article className="info-card">
          <h2>Today</h2>
          <p>Daily speech check-in is not connected yet.</p>
          <Link className="text-link" to="/check-in">
            Open daily check-in
          </Link>
        </article>
        <article className="info-card">
          <h2>Latest screen</h2>
          <p>Results will appear after screening is added.</p>
          <Link className="text-link" to="/results">
            Go to results
          </Link>
        </article>
        <article className="info-card">
          <h2>Past check-ins</h2>
          <p>History will list earlier sessions once storage is ready.</p>
          <Link className="text-link" to="/history">
            View history
          </Link>
        </article>
      </div>
    </section>
  )
}

export default Dashboard
