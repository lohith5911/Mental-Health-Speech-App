import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import PageHeader from '../components/PageHeader'
import { useAuth } from '../context/useAuth'
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

const trendLabels: Record<TrendInsight['trend'], string> = {
  stable: 'Pattern appears stable',
  change_detected: 'Change detected',
  significant_change: 'Significant pattern change',
}

function formatDate(value: string) {
  return new Intl.DateTimeFormat('en-US', {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(new Date(value))
}

function formatPercentage(value: number) {
  return `${Math.min(100, Math.max(0, value * 100)).toFixed(1)}%`
}

function formatEmotion(value: string) {
  return value.charAt(0).toUpperCase() + value.slice(1)
}

function Dashboard() {
  const { user } = useAuth()
  const [checkIns, setCheckIns] = useState<CheckInRecord[]>([])
  const [trendInsights, setTrendInsights] = useState<TrendInsight | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [currentTime] = useState(() => Date.now())

  async function loadDashboard() {
    setLoading(true)
    setError(null)

    try {
      const [records, insights] = await Promise.all([fetchCheckIns(), fetchTrendInsights()])
      setCheckIns(records)
      setTrendInsights(insights)
    } catch {
      setError('Unable to load your dashboard right now. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    let active = true

    async function loadInitialDashboard() {
      try {
        const [records, insights] = await Promise.all([fetchCheckIns(), fetchTrendInsights()])
        if (active) {
          setCheckIns(records)
          setTrendInsights(insights)
        }
      } catch {
        if (active) {
          setError('Unable to load your dashboard right now. Please try again.')
        }
      } finally {
        if (active) {
          setLoading(false)
        }
      }
    }

    void loadInitialDashboard()

    return () => {
      active = false
    }
  }, [])

  const latest = checkIns[0] ?? null
  const recentCheckIns = checkIns.slice(0, 4)
  const recentPeriodStart = currentTime - 30 * 24 * 60 * 60 * 1000
  const recentPeriodCheckIns = checkIns.filter(
    (checkIn) => new Date(checkIn.created_at).getTime() >= recentPeriodStart,
  )
  const recentDays = new Set(
    recentPeriodCheckIns.map((checkIn) => new Date(checkIn.created_at).toLocaleDateString('en-CA')),
  )
  const qualityRecords = recentCheckIns.filter((checkIn) => checkIn.quality)
  const usableQualityCount = qualityRecords.filter((checkIn) => checkIn.quality?.status === 'usable').length
  const largestChange = trendInsights
    ? emotions.reduce((largest, emotion) => (
        Math.abs(trendInsights.change[emotion]) > Math.abs(largest.change)
          ? { emotion, change: trendInsights.change[emotion] }
          : largest
      ), { emotion: 'neutral' as Emotion, change: trendInsights.change.neutral })
    : null

  return (
    <section className="page dashboard-page">
      <PageHeader
        title="Dashboard"
        description={user ? `${user.display_name}'s speech check-in overview.` : 'Your speech check-in overview.'}
      />

      {loading ? (
        <article className="info-card loading-state" aria-live="polite">
          <h2>Loading your overview</h2>
          <p>Preparing your recent check-ins and speech-derived patterns.</p>
        </article>
      ) : null}

      {!loading && error ? (
        <article className="info-card error-state" role="alert">
          <h2>Dashboard unavailable</h2>
          <p>{error}</p>
          <button className="button secondary" type="button" onClick={() => void loadDashboard()}>
            Try again
          </button>
        </article>
      ) : null}

      {!loading && !error && checkIns.length === 0 ? (
        <article className="info-card empty-state dashboard-empty-state">
          <h2>Start your personal overview</h2>
          <p>No check-ins have been recorded yet. Complete a brief voice check-in to begin tracking your speech-derived signals over time.</p>
          <Link className="button primary" to="/check-in">
            Start daily check-in
          </Link>
        </article>
      ) : null}

      {!loading && !error && latest && trendInsights ? (
        <>
          <div className="dashboard-summary-grid">
            <article className="info-card dashboard-latest-card">
              <div className="section-heading-row">
                <div>
                  <p className="eyebrow">Latest check-in</p>
                  <h2>Speech-derived result</h2>
                </div>
                <Link className="text-link" to="/results">View results</Link>
              </div>
              <p className="dashboard-emotion">{formatEmotion(latest.emotion)}</p>
              <dl className="dashboard-details">
                <div><dt>Confidence</dt><dd>{Math.round(latest.confidence * 100)}%</dd></div>
                <div><dt>Recorded</dt><dd>{formatDate(latest.created_at)}</dd></div>
                <div><dt>Duration</dt><dd>{latest.duration_seconds}s</dd></div>
                {latest.model_version ? <div><dt>Model</dt><dd>{latest.model_version}</dd></div> : null}
              </dl>
              <p className="dashboard-note">This result describes signals from one recording and is intended for self-monitoring.</p>
            </article>

            <article className="info-card dashboard-trend-card">
              <div className="section-heading-row">
                <div>
                  <p className="eyebrow">Speech-derived trend</p>
                  <h2>{trendLabels[trendInsights.trend]}</h2>
                </div>
                <span className="trend-status-label">{trendInsights.trend.replace('_', ' ')}</span>
              </div>
              <p>Recent emotion signals compared with your available personal baseline.</p>
              <div className="trend-stats dashboard-trend-stats">
                <div><span className="trend-stat-label">Analyzed</span><strong>{trendInsights.sample_size}</strong></div>
                <div><span className="trend-stat-label">Window</span><strong>{trendInsights.window_size}</strong></div>
                <div><span className="trend-stat-label">Change score</span><strong>{trendInsights.change_score.toFixed(2)}</strong></div>
              </div>
              {largestChange ? (
                <p className="dashboard-change">
                  Largest distribution change: <strong>{emotionLabels[largestChange.emotion]}</strong>{' '}
                  {largestChange.change >= 0 ? 'increased' : 'decreased'} by {formatPercentage(Math.abs(largestChange.change))}.
                </p>
              ) : null}
              <Link className="text-link" to="/history">Explore history and trends</Link>
            </article>
          </div>

          <div className="dashboard-content-grid">
            <article className="info-card">
              <div className="section-heading-row">
                <div>
                  <h2>Recent activity</h2>
                  <p>Your latest saved check-ins.</p>
                </div>
                <Link className="text-link" to="/history">View all</Link>
              </div>
              <div className="dashboard-activity-list">
                {recentCheckIns.map((checkIn) => (
                  <div key={checkIn.id} className="dashboard-activity-item">
                    <div>
                      <p className="history-date">{formatDate(checkIn.created_at)}</p>
                      <p className="history-emotion">{formatEmotion(checkIn.emotion)}</p>
                    </div>
                    <div className="history-meta">
                      <span>{Math.round(checkIn.confidence * 100)}% confidence</span>
                      <span>{checkIn.duration_seconds}s</span>
                    </div>
                  </div>
                ))}
              </div>
            </article>

            <article className="info-card">
              <h2>Recent emotion distribution</h2>
              <p>Average speech-derived emotion probabilities from the recent analysis window.</p>
              <div className="dashboard-distribution-list">
                {emotions.map((emotion) => (
                  <div key={emotion} className="dashboard-distribution-row">
                    <span>{emotionLabels[emotion]}</span>
                    <div
                      className="distribution-track"
                      role="progressbar"
                      aria-label={`${emotionLabels[emotion]} recent distribution`}
                      aria-valuemin={0}
                      aria-valuemax={100}
                      aria-valuenow={trendInsights.recent[emotion] * 100}
                      aria-valuetext={formatPercentage(trendInsights.recent[emotion])}
                    >
                      <span className="distribution-fill recent-fill" style={{ width: formatPercentage(trendInsights.recent[emotion]) }} />
                    </div>
                    <strong>{formatPercentage(trendInsights.recent[emotion])}</strong>
                  </div>
                ))}
              </div>
            </article>
          </div>

          <div className="dashboard-metrics-grid">
            <article className="info-card dashboard-metric-card">
              <span className="result-stat-label">Total check-ins</span>
              <strong>{checkIns.length}</strong>
              <p>All saved check-ins in your account.</p>
            </article>
            <article className="info-card dashboard-metric-card">
              <span className="result-stat-label">Last 30 days</span>
              <strong>{recentPeriodCheckIns.length}</strong>
              <p>{recentDays.size} day{recentDays.size === 1 ? '' : 's'} with a check-in.</p>
            </article>
            <article className="info-card dashboard-metric-card">
              <span className="result-stat-label">Recent signal quality</span>
              <strong>{qualityRecords.length ? `${Math.round((usableQualityCount / qualityRecords.length) * 100)}%` : 'Not available'}</strong>
              <p>{qualityRecords.length ? 'Recent recordings marked usable.' : 'Quality data is not available yet.'}</p>
            </article>
          </div>

          <div className="cta-row dashboard-actions">
            <Link className="button primary" to="/check-in">Record a check-in</Link>
            <Link className="button ghost" to="/history">Open full history</Link>
          </div>
        </>
      ) : null}
    </section>
  )
}

export default Dashboard
