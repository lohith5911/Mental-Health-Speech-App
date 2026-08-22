import PageHeader from '../components/PageHeader'

function History() {
  return (
    <section className="page">
      <PageHeader
        title="History"
        description="A timeline of past check-ins will appear here. No past sessions are listed yet."
      />

      <article className="info-card">
        <h2>Past sessions</h2>
        <p>
          After storage is added, you will be able to open earlier dates from
          this list. There is no sample history in this build.
        </p>
      </article>
    </section>
  )
}

export default History
