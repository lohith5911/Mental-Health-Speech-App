type MicrophoneIconProps = {
  className?: string
}

function MicrophoneIcon({ className }: MicrophoneIconProps) {
  return (
    <svg
      className={className}
      viewBox="0 0 64 64"
      role="img"
      aria-hidden="true"
    >
      <rect x="24" y="8" width="16" height="28" rx="8" fill="currentColor" />
      <path
        d="M16 30c0 8.8 7.2 16 16 16s16-7.2 16-16"
        fill="none"
        stroke="currentColor"
        strokeWidth="4"
        strokeLinecap="round"
      />
      <path
        d="M32 46v8M22 54h20"
        fill="none"
        stroke="currentColor"
        strokeWidth="4"
        strokeLinecap="round"
      />
    </svg>
  )
}

export default MicrophoneIcon
