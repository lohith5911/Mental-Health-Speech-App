import type { CheckInStatus, EmotionAnalysisResult } from '../../types'
import MicrophoneIcon from './MicrophoneIcon'

type RecordingCardProps = {
  status: CheckInStatus
  formattedTime: string
  audioUrl: string | null
  errorMessage: string | null
  successMessage: string | null
  isRequestingMic: boolean
  onStart: () => void
  onStop: () => void
  onRecordAgain: () => void
  onContinue: () => void
  analysisResult: EmotionAnalysisResult | null
}

const STATUS_COPY: Record<
  CheckInStatus,
  { title: string; detail: string }
> = {
  idle: {
    title: 'Ready to record',
    detail: 'Press start when you are ready to speak.',
  },
  recording: {
    title: 'Recording',
    detail: 'Speak naturally. You can stop at any time.',
  },
  recorded: {
    title: 'Recording complete',
    detail: 'Listen to your recording, then continue or record again.',
  },
  uploading: {
    title: 'Uploading your recording',
    detail: 'Your recording is being sent securely for analysis.',
  },
  analyzing: {
    title: 'Analyzing your recording',
    detail: 'Your recording is being analyzed for emotional cues.',
  },
  success: {
    title: 'Analysis complete',
    detail: 'Review the result below or record another check-in.',
  },
  error: {
    title: 'Analysis could not be completed',
    detail: 'Check your connection and try the analysis again.',
  },
}

function RecordingCard({
  status,
  formattedTime,
  audioUrl,
  errorMessage,
  successMessage,
  isRequestingMic,
  onStart,
  onStop,
  onRecordAgain,
  onContinue,
  analysisResult,
}: RecordingCardProps) {
  const copy = STATUS_COPY[status]
  const idleDetail = isRequestingMic
    ? 'Waiting for microphone permission…'
    : copy.detail

  return (
    <article className={`recording-card status-${status}`}>
      <div className="mic-wrap">
        {status === 'idle' || status === 'recording' ? (
          <button
            type="button"
            className="mic-button"
            onClick={status === 'idle' ? onStart : onStop}
            aria-label={status === 'idle' ? 'Start Recording' : 'Stop Recording'}
            disabled={isRequestingMic}
          >
            <MicrophoneIcon className="mic-icon" />
          </button>
        ) : (
          <MicrophoneIcon className="mic-icon" />
        )}
        {status === 'recording' ? (
          <span className="recording-dot" aria-hidden="true" />
        ) : null}
      </div>

      <p className="recording-status">{copy.title}</p>
      <p className="recording-detail">
        {status === 'idle' ? idleDetail : copy.detail}
      </p>
      <p className="recording-timer" aria-live="polite">
        {formattedTime}
      </p>

      {status === 'recording' ? (
        <p className="recording-indicator">
          <span className="pulse-dot" aria-hidden="true" />
          Recording in progress
        </p>
      ) : null}

      {audioUrl && status !== 'idle' && status !== 'recording' ? (
        <audio className="audio-preview" controls src={audioUrl} preload="metadata">
          Your browser cannot play this audio preview.
        </audio>
      ) : null}

      {analysisResult && status === 'success' ? (
        <div className="analysis-result" aria-live="polite">
          <p className="analysis-label">Detected Emotion</p>
          <p className="analysis-emotion">{analysisResult.emotion}</p>
          <p className="analysis-confidence">
            Confidence: <strong>{Math.round(analysisResult.confidence * 100)}%</strong>
          </p>
        </div>
      ) : null}

      {successMessage ? (
        <p className="recording-success" role="status">
          {successMessage}
        </p>
      ) : null}

      {errorMessage ? (
        <p className="recording-error" role="alert">
          {errorMessage}
        </p>
      ) : null}

      <div className="recording-actions">
        {status === 'idle' ? (
          <button
            className="button primary"
            type="button"
            onClick={onStart}
            disabled={isRequestingMic}
          >
            Start Recording
          </button>
        ) : null}

        {status === 'recording' ? (
          <button className="button stop" type="button" onClick={onStop}>
            Stop Recording
          </button>
        ) : null}

        {status === 'recorded' ? (
          <>
            <button className="button ghost" type="button" onClick={onRecordAgain}>
              Record Again
            </button>
            <button className="button primary" type="button" onClick={onContinue}>
              Continue
            </button>
          </>
        ) : null}

        {status === 'uploading' || status === 'analyzing' ? (
          <>
            <div className="processing-row">
              <span className="spinner" aria-hidden="true" />
              <span>
                {status === 'uploading'
                  ? 'Uploading your recording…'
                  : 'Analyzing your recording…'}
              </span>
            </div>
            <button className="button ghost" type="button" onClick={onRecordAgain}>
              Record Again
            </button>
          </>
        ) : null}

        {status === 'success' || status === 'error' ? (
          <button className="button ghost" type="button" onClick={onRecordAgain}>
            Record Again
          </button>
        ) : null}
      </div>
    </article>
  )
}

export default RecordingCard
