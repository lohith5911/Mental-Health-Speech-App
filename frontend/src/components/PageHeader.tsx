type PageHeaderProps = {
  title: string
  description: string
}

function PageHeader({ title, description }: PageHeaderProps) {
  return (
    <header className="page-header">
      <h1>{title}</h1>
      <p>{description}</p>
    </header>
  )
}

export default PageHeader
