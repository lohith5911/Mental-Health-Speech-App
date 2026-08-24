import RecordingCard from '../components/check-in/RecordingCard'
import PageHeader from '../components/PageHeader'
import useRecording from '../hooks/useRecording'

function DailyCheckIn() {
  const recording = useRecording()

  return (
    <section className="page check-in-page">
      <PageHeader
        title="Daily Check-In"
        description="Speak naturally about your day for approximately 30–60 seconds. There is no need to perform or use special language."
      />

      <article className="info-card prompt-card">
        <p className="prompt-label">Today’s prompt</p>
        <h2>
          How was your day today? Tell us about anything that affected how you
          felt.
        </h2>
      </article>

      <RecordingCard
        status={recording.status}
        formattedTime={recording.formattedTime}
        audioUrl={recording.audioUrl}
        errorMessage={recording.errorMessage}
        successMessage={recording.successMessage}
        isRequestingMic={recording.isRequestingMic}
        onStart={() => {
          void recording.startRecording()
        }}
        onStop={recording.stopRecording}
        onRecordAgain={recording.recordAgain}
        onContinue={recording.continueToProcessing}
      />

      <p className="privacy-notice">
        Your voice recording will only be used for the screening analysis and
        will not be shared without your permission.
      </p>
      <p className="screening-note">
        This tool provides screening insights, not a medical diagnosis.
      </p>
    </section>
  )
}

export default DailyCheckIn
