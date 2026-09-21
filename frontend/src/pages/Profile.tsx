import PageHeader from '../components/PageHeader'
import { useAuth } from '../context/useAuth'

function Profile() {
  const { user } = useAuth()

  return (
    <section className="page narrow-page">
      <PageHeader
        title="Profile"
        description="Your account details."
      />

      <article className="info-card">
        <h2>Your account</h2>
        <p><strong>Name:</strong> {user?.display_name}</p>
        <p><strong>Email:</strong> {user?.email}</p>
        <p><strong>Account created:</strong> {user && new Date(user.created_at).toLocaleDateString()}</p>
      </article>
    </section>
  )
}

export default Profile
