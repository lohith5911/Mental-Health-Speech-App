import { useEffect, useState } from 'react'
import PageHeader from '../components/PageHeader'
import { fetchCheckIns } from '../services'
import type { CheckInRecord } from '../types'

function formatCheckInDate(value: string) {
  return new Intl.DateTimeFormat('en-US', {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(new Date(value))
}

function Results() {
  const [checkIns, setCheckIns] = useState<CheckInRecord[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let active = true

    async function loadResults() {
      setLoading(true)
      setError(null)

      try {
        const records = await fetchCheckIns()
        if (!active) {
          return
        }
        setCheckIns(records)
      } catch (loadError) {
        if (!active) {
          return
        }
        setError(
          loadError instanceof Error
            ? loadError.message
            : 'Unable to load your latest check-in results.',
        )
      } finally {
        if (active) {
          setLoading(false)
        }
      }
    }

    void loadResults()

    return () => {
      active = false
    }
  }, [])

  const latest = checkIns[0] ?? null
  const averageConfidence = checkIns.length
    ? checkIns.reduce((sum, entry) => sum + entry.confidence, 0) / checkIns.length
    : 0

  return (
    <section className="page">
      <PageHeader
        title="Results"
        description="A quick summary of your most recent emotional screening, along with a trend snapshot."
      />

      {loading ? (
        <article className="info-card loading-state">
          <h2>Loading results</h2>
          <p>Preparing your latest screening summary…</p>
        </article>
      ) : null}

      {!loading && error ? (
        <article className="info-card error-state">
          <h2>Results unavailable</h2>
          <p>{error}</p>
        </article>
      ) : null}

      {!loading && !error && !latest ? (
        <article className="info-card empty-state">
          <h2>No check-in results yet</h2>
          <p>Record a brief voice check-in to see your latest emotion summary here.</p>
        </article>
      ) : null}

      {!loading && !error && latest ? (
        <>
          <article className="info-card results-summary">
            <h2>Latest screen</h2>
            <div className="results-emotion-wrap">
              <p className="results-label">Detected emotion</p>
              <p className="results-emotion">{latest.emotion}</p>
            </div>
            <p className="results-detail">
              Confidence: <strong>{Math.round(latest.confidence * 100)}%</strong>
            </p>
            <p className="results-detail">
              Recorded: <strong>{formatCheckInDate(latest.created_at)}</strong>
            </p>
          </article>

          <div className="results-grid">
            <article className="info-card result-stat">
              <p className="result-stat-label">Check-ins saved</p>
              <p className="result-stat-value">{checkIns.length}</p>
            </article>

            <article className="info-card result-stat">
              <p className="result-stat-label">Average confidence</p>
              <p className="result-stat-value">
                {Math.round(averageConfidence * 100)}%
              </p>
            </article>

            <article className="info-card result-stat">
              <p className="result-stat-label">Most recent duration</p>
              <p className="result-stat-value">{latest.duration_seconds}s</p>
            </article>
          </div>
        </>
      ) : null}
    </section>
  )
}

export default Results
