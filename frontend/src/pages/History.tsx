import { useEffect, useState } from 'react'
import PageHeader from '../components/PageHeader'
import { fetchCheckIns, fetchTrendInsights } from '../services'
import type { CheckInRecord, Emotion, TrendInsight } from '../types'

const emotions: Emotion[] = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']

const emotionLabels: Record<Emotion, string> = {
  angry: 'Angry',
  disgust: 'Disgust',
  fear: 'Fear',
  happy: 'Happy',
  neutral: 'Neutral',
  sad: 'Sad',
}

const trendDescriptions = {
  stable: 'Your recent speech-derived pattern is broadly consistent with your baseline.',
  change_detected: 'A change has appeared in your recent speech-derived pattern compared with your baseline.',
  significant_change: 'A more substantial change has appeared in your recent speech-derived pattern compared with your baseline.',
} as const

function displayPercentage(value: number) {
  return Math.min(100, Math.max(0, value * 100))
}

function formatPercentage(value: number) {
  return `${displayPercentage(value).toFixed(1)}%`
}

function trendStatusLabel(trend: TrendInsight['trend']) {
  if (trend === 'stable') {
    return 'Pattern appears stable'
  }
  if (trend === 'change_detected') {
    return 'Change detected'
  }
  return 'Significant pattern change'
}

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
  const [trendInsights, setTrendInsights] = useState<TrendInsight | null>(null)
  const [trendLoading, setTrendLoading] = useState(true)
  const [trendError, setTrendError] = useState(false)

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

  useEffect(() => {
    let active = true

    async function loadTrendInsights() {
      setTrendLoading(true)
      setTrendError(false)

      try {
        const insights = await fetchTrendInsights()
        if (active) {
          setTrendInsights(insights)
        }
      } catch {
        if (active) {
          setTrendError(true)
        }
      } finally {
        if (active) {
          setTrendLoading(false)
        }
      }
    }

    void loadTrendInsights()

    return () => {
      active = false
    }
  }, [])

  const changedEmotions = trendInsights
    ? emotions
        .map((emotion) => ({ emotion, change: trendInsights.change[emotion] }))
        .sort((left, right) => Math.abs(right.change) - Math.abs(left.change))
    : []
  const chronologicalCheckIns = [...checkIns].reverse()

  return (
    <section className="page">
      <PageHeader
        title="Wellbeing History"
        description="Track changes in your speech-derived wellbeing patterns over time."
      />

      {trendLoading && checkIns.length > 0 ? (
        <article className="info-card loading-state">
          <h2>Loading pattern insights...</h2>
          <p>Preparing your speech-derived pattern summary.</p>
        </article>
      ) : null}

      {!trendLoading && trendError && checkIns.length > 0 ? (
        <article className="info-card error-state">
          <h2>Pattern insights unavailable</h2>
          <p>Pattern insights are temporarily unavailable.</p>
        </article>
      ) : null}

      {!loading && !error && checkIns.length > 0 && trendInsights ? (
        <>
          <article className="info-card trend-status-card">
            <div className="section-heading-row">
              <div>
                <p className="eyebrow">Speech-derived pattern</p>
                <h2>{trendStatusLabel(trendInsights.trend)}</h2>
              </div>
              <span className="trend-status-label">{trendInsights.trend.replace('_', ' ')}</span>
            </div>
            <p>{trendDescriptions[trendInsights.trend]}</p>
            <div className="trend-stats" aria-label="Pattern analysis details">
              <div>
                <span className="trend-stat-label">Check-ins analyzed</span>
                <strong>{trendInsights.sample_size}</strong>
              </div>
              <div>
                <span className="trend-stat-label">Change score</span>
                <strong>{trendInsights.change_score.toFixed(2)}</strong>
              </div>
              <div>
                <span className="trend-stat-label">Analysis window</span>
                <strong>{trendInsights.window_size} check-ins</strong>
              </div>
            </div>
          </article>

          <article className="info-card">
            <div>
              <h2>Baseline and recent distribution</h2>
              <p>Compare each emotion signal with your personal baseline.</p>
            </div>
            <div className="distribution-list">
              {emotions.map((emotion) => (
                <div key={emotion} className="distribution-item">
                  <h3>{emotionLabels[emotion]}</h3>
                  <div className="distribution-row">
                    <span>Baseline</span>
                    <div
                      className="distribution-track"
                      role="progressbar"
                      aria-label={`${emotionLabels[emotion]} baseline`}
                      aria-valuemin={0}
                      aria-valuemax={100}
                      aria-valuenow={displayPercentage(trendInsights.baseline[emotion])}
                    >
                      <span
                        className="distribution-fill baseline-fill"
                        style={{ width: `${displayPercentage(trendInsights.baseline[emotion])}%` }}
                      />
                    </div>
                    <strong>{formatPercentage(trendInsights.baseline[emotion])}</strong>
                  </div>
                  <div className="distribution-row">
                    <span>Recent</span>
                    <div
                      className="distribution-track"
                      role="progressbar"
                      aria-label={`${emotionLabels[emotion]} recent distribution`}
                      aria-valuemin={0}
                      aria-valuemax={100}
                      aria-valuenow={displayPercentage(trendInsights.recent[emotion])}
                    >
                      <span
                        className="distribution-fill recent-fill"
                        style={{ width: `${displayPercentage(trendInsights.recent[emotion])}%` }}
                      />
                    </div>
                    <strong>{formatPercentage(trendInsights.recent[emotion])}</strong>
                  </div>
                </div>
              ))}
            </div>
          </article>

          <div className="insights-grid">
            <article className="info-card">
              <h2>Change from baseline</h2>
              <div className="change-list">
                {changedEmotions.map(({ emotion, change }) => (
                  <div key={emotion} className="change-item">
                    <span>{emotionLabels[emotion]}</span>
                    <span>
                      <strong>{change >= 0 ? '+' : ''}{(change * 100).toFixed(1)}%</strong>{' '}
                      <small>{change >= 0 ? 'increased' : 'decreased'}</small>
                    </span>
                  </div>
                ))}
              </div>
            </article>

            {trendInsights.persistent_emotions.length > 0 ? (
              <article className="info-card">
                <h2>Patterns appearing consistently</h2>
                <p>These emotions have shown a sustained change across multiple recent check-ins.</p>
                <ul className="persistent-list">
                  {trendInsights.persistent_emotions.map((emotion) => (
                    <li key={emotion}>{emotionLabels[emotion]}</li>
                  ))}
                </ul>
              </article>
            ) : null}
          </div>

          <article className="info-card">
            <h2>Emotion timeline</h2>
            <p>Recent check-in patterns, shown from oldest to newest.</p>
            <ol className="emotion-timeline">
              {chronologicalCheckIns.map((checkIn) => (
                <li key={checkIn.id} className="timeline-item">
                  <div className="timeline-marker" aria-hidden="true" />
                  <div className="timeline-content">
                    <p className="history-date">{formatCheckInDate(checkIn.created_at)}</p>
                    <p className="history-emotion">{checkIn.emotion}</p>
                    <p className="timeline-confidence">{Math.round(checkIn.confidence * 100)}% confidence</p>
                  </div>
                </li>
              ))}
            </ol>
          </article>
        </>
      ) : null}

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
          <h2>No check-ins yet</h2>
          <p>Complete a daily voice check-in to start building your personal pattern.</p>
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
