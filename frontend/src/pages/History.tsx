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

function History() {
  const [checkIns, setCheckIns] = useState<CheckInRecord[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let active = true

    async function loadHistory() {
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
            : 'Unable to load your check-in history right now.',
        )
      } finally {
        if (active) {
          setLoading(false)
        }
      }
    }

    void loadHistory()

    return () => {
      active = false
    }
  }, [])

  return (
    <section className="page">
      <PageHeader
        title="History"
        description="A timeline of your saved emotional check-ins. Each one is kept locally for trend tracking."
      />

      {loading ? (
        <article className="info-card loading-state">
          <h2>Loading history</h2>
          <p>Fetching your recent check-ins…</p>
        </article>
      ) : null}

      {!loading && error ? (
        <article className="info-card error-state">
          <h2>History unavailable</h2>
          <p>{error}</p>
        </article>
      ) : null}

      {!loading && !error && checkIns.length === 0 ? (
        <article className="info-card empty-state">
          <h2>No saved check-ins yet</h2>
          <p>Complete your first daily check-in to start tracking patterns.</p>
        </article>
      ) : null}

      {!loading && !error && checkIns.length > 0 ? (
        <article className="info-card">
          <h2>Past sessions</h2>
          <div className="history-list">
            {checkIns.map((checkIn) => (
              <div key={checkIn.id} className="history-item">
                <div>
                  <p className="history-date">{formatCheckInDate(checkIn.created_at)}</p>
                  <p className="history-emotion">{checkIn.emotion}</p>
                </div>
                <div className="history-meta">
                  <span>{Math.round(checkIn.confidence * 100)}% confidence</span>
                  <span>{checkIn.duration_seconds}s</span>
                </div>
              </div>
            ))}
          </div>
        </article>
      ) : null}
    </section>
  )
}

export default History
