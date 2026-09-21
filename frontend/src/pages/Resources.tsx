import PageHeader from '../components/PageHeader'

function Resources() {
  return (
    <section className="page">
      <PageHeader
        title="Resources"
        description="Practical guidance for using speech check-ins responsibly. MindTrace is a non-diagnostic self-monitoring tool."
      />

      <div className="card-grid">
        <article className="info-card">
          <h2>Understanding speech signals</h2>
          <p>
            Emotion labels, probabilities, acoustic measurements, and signal
            quality describe one recording. They are not a diagnosis and can
            be affected by context, microphone quality, and background noise.
          </p>
        </article>
        <article className="info-card">
          <h2>Making a useful check-in</h2>
          <p>
            Choose a quiet place, keep the microphone unobstructed, and speak
            naturally for approximately 30–60 seconds. Repeat a recording if
            it is too short, mostly silent, or noticeably noisy.
          </p>
        </article>
        <article className="info-card">
          <h2>Reading confidence and trends</h2>
          <p>
            Confidence is the model's relative preference among its supported
            emotion classes, not certainty about how you feel. History trends
            compare your recent speech-derived signals with your own records.
          </p>
        </article>
        <article className="info-card">
          <h2>Privacy and account safety</h2>
          <p>
            Sign out on shared devices and use a unique password. Check-ins
            are associated with your account and are returned only to that
            authenticated account by the backend.
          </p>
        </article>
        <article className="info-card">
          <h2>When to seek help</h2>
          <p>
            If you feel unsafe or are in immediate danger, contact local
            emergency services. In the U.S., call or text 988. A qualified
            mental-health professional can provide personal assessment and
            support; this app is not a substitute for care.
          </p>
        </article>
      </div>
    </section>
  )
}

export default Resources
