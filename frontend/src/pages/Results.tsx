import PageHeader from '../components/PageHeader'

function Results() {
  return (
    <section className="page">
      <PageHeader
        title="Results"
        description="Screening output from your speech check-ins will be shown here. No scores or labels are displayed yet."
      />

      <article className="info-card">
        <h2>Latest screen</h2>
        <p>
          When the model and API are connected, this page will present a clear
          summary. It will stay empty of sample or fake results until then.
        </p>
      </article>
    </section>
  )
}

export default Results
