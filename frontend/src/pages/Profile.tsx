import PageHeader from '../components/PageHeader'

function Profile() {
  return (
    <section className="page narrow-page">
      <PageHeader
        title="Profile"
        description="Account details and preferences will live here. Profile data is not loaded yet."
      />

      <article className="info-card">
        <h2>Your account</h2>
        <p>
          Name, email, and notification settings will be editable after
          authentication is connected. No personal details are stored in this
          foundation build.
        </p>
      </article>
    </section>
  )
}

export default Profile
