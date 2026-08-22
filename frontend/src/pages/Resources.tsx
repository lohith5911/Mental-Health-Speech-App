import PageHeader from '../components/PageHeader'

function Resources() {
  return (
    <section className="page">
      <PageHeader
        title="Resources"
        description="Support links and guidance will be collected here. This list is a starting point, not a clinical recommendation."
      />

      <div className="card-grid">
        <article className="info-card">
          <h2>Crisis support</h2>
          <p>
            If you are in immediate danger, contact local emergency services.
            In the U.S., you can call or text 988 for the Suicide &amp; Crisis
            Lifeline.
          </p>
        </article>
        <article className="info-card">
          <h2>Professional care</h2>
          <p>
            Licensed clinicians can provide assessment and treatment. This app
            is meant as a screening companion, not a substitute for care.
          </p>
        </article>
        <article className="info-card">
          <h2>How this app will help</h2>
          <p>
            Later versions will explain how speech check-ins work and how to
            read screens. Education copy will be added with the screening
            feature.
          </p>
        </article>
      </div>
    </section>
  )
}

export default Resources
