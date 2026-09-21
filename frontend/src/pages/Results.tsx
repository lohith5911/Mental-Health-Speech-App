import { useEffect, useState } from 'react'
import PageHeader from '../components/PageHeader'
import { fetchCheckIns } from '../services'
import type { AcousticFeatures, CheckInQualityStatus, CheckInRecord, Emotion } from '../types'

const emotions: Emotion[] = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']

const emotionLabels: Record<Emotion, string> = {
  angry: 'Angry',
  disgust: 'Disgust',
  fear: 'Fear',
  happy: 'Happy',
  neutral: 'Neutral',
  sad: 'Sad',
}

const qualityStatusLabels: Record<CheckInQualityStatus, string> = {
  usable: 'Usable',
  low_signal: 'Low signal',
  mostly_silent: 'Mostly silent',
  noisy_signal: 'Noisy signal',
  insufficient_audio: 'Insufficient audio',
}

const qualityReasonLabels: Record<string, string> = {
  frontend_duration_mismatch: 'The recording duration differed slightly from the browser estimate.',
  low_signal: 'The recording had a low signal level.',
  mostly_silent: 'The recording contained mostly silence.',
  noisy_signal: 'The recording contained a high level of background noise.',
  insufficient_audio: 'There was not enough usable audio to assess the signal clearly.',
}

const acousticMeasurements: Array<{
  key: keyof AcousticFeatures
  label: string
  unit?: string
}> = [
  { key: 'duration_seconds', label: 'Acoustic duration', unit: 's' },
  { key: 'rms_mean', label: 'RMS mean' },
  { key: 'rms_std', label: 'RMS standard deviation' },
  { key: 'zcr_mean', label: 'Zero-crossing rate mean' },
  { key: 'zcr_std', label: 'Zero-crossing rate standard deviation' },
  { key: 'pitch_mean_hz', label: 'Pitch mean', unit: 'Hz' },
  { key: 'pitch_std_hz', label: 'Pitch standard deviation', unit: 'Hz' },
  { key: 'pitch_range_hz', label: 'Pitch range', unit: 'Hz' },
  { key: 'silence_ratio', label: 'Silence ratio', unit: '%' },
  { key: 'speaking_rate_proxy', label: 'Speaking-rate proxy' },
]

function formatCheckInDate(value: string) {
  return new Intl.DateTimeFormat('en-US', {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(new Date(value))
}

function formatMeasurement(value: number | null | undefined, unit?: string) {
  if (value === null || value === undefined) {
    return 'Not available'
  }
  const formatted = unit === '%' ? `${(value * 100).toFixed(1)}%` : value.toFixed(3)
  return unit && unit !== '%' ? `${formatted} ${unit}` : formatted
}

function formatQualityReason(reason: string) {
  return qualityReasonLabels[reason] ?? reason.replaceAll('_', ' ')
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
            <div className="section-heading-row">
              <div>
                <p className="eyebrow">Latest check-in</p>
                <h2>Analysis summary</h2>
              </div>
              <span className="results-model">{latest.model_version ?? 'Model not available'}</span>
            </div>
            <div className="results-emotion-wrap">
              <p className="results-label">Detected emotion</p>
              <p className="results-emotion">{latest.emotion}</p>
            </div>
            <dl className="results-summary-details">
              <div><dt>V4 confidence</dt><dd>{Math.round(latest.confidence * 100)}%</dd></div>
              <div><dt>Recording duration</dt><dd>{latest.duration_seconds}s</dd></div>
              <div><dt>Date and time</dt><dd>{formatCheckInDate(latest.created_at)}</dd></div>
            </dl>
          </article>

          <article className="info-card">
            <div>
              <h2>Emotion probability distribution</h2>
              <p>Probabilities produced by the V4 emotion classifier for this recording.</p>
            </div>
            {latest.probabilities ? (
              <div className="result-probability-list">
                {emotions.map((emotion) => {
                  const probability = latest.probabilities?.[emotion]
                  const percentage = probability === undefined ? null : Math.min(100, Math.max(0, probability * 100))
                  return (
                    <div key={emotion} className="result-probability-row">
                      <span className="result-probability-label">{emotionLabels[emotion]}</span>
                      <div
                        className="result-probability-track"
                        role="progressbar"
                        aria-label={`${emotionLabels[emotion]} probability`}
                        aria-valuemin={0}
                        aria-valuemax={100}
                        aria-valuenow={percentage ?? 0}
                        aria-valuetext={percentage === null ? 'Not available' : `${percentage.toFixed(1)} percent`}
                      >
                        <span className="result-probability-fill" style={{ width: `${percentage ?? 0}%` }} />
                      </div>
                      <strong>{percentage === null ? 'Not available' : `${percentage.toFixed(1)}%`}</strong>
                    </div>
                  )
                })}
              </div>
            ) : <p className="not-available">Not available for this check-in.</p>}
          </article>

          <article className="info-card">
            <div>
              <h2>Speech-derived acoustic measurements</h2>
              <p>Measurements calculated from the recorded speech signal.</p>
            </div>
            {latest.acoustic_features ? (
              <dl className="acoustic-measurements">
                {acousticMeasurements.map(({ key, label, unit }) => (
                  <div key={key}>
                    <dt>{label}</dt>
                    <dd>{formatMeasurement(latest.acoustic_features?.[key] as number | null | undefined, unit)}</dd>
                  </div>
                ))}
              </dl>
            ) : <p className="not-available">Not available for this check-in.</p>}
          </article>

          <article className="info-card result-quality-card">
            <div>
              <h2>Signal quality</h2>
              <p>Quality checks help describe how usable this recording was for analysis.</p>
            </div>
            {latest.quality ? (
              <>
                <p className="quality-status"><strong>{qualityStatusLabels[latest.quality.status]}</strong></p>
                {latest.quality.reasons.length > 0 ? (
                  <ul className="quality-reasons">
                    {latest.quality.reasons.map((reason) => <li key={reason}>{formatQualityReason(reason)}</li>)}
                  </ul>
                ) : <p className="not-available">No quality concerns were recorded.</p>}
              </>
            ) : <p className="not-available">Not available for this check-in.</p>}
          </article>

          <aside className="results-disclaimer">
            This result summarizes speech-derived emotion probabilities and acoustic signal characteristics from this check-in. These signals are intended for self-monitoring and are not a medical diagnosis.
          </aside>

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

            {latest.acoustic_features ? (
              <article className="info-card result-stat">
                <p className="result-stat-label">Silence ratio</p>
                <p className="result-stat-value">
                  {Math.round(latest.acoustic_features.silence_ratio * 100)}%
                </p>
              </article>
            ) : null}
          </div>
        </>
      ) : null}
    </section>
  )
}

export default Results
