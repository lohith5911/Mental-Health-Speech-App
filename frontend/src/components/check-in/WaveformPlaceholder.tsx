const BAR_HEIGHTS = [28, 46, 62, 38, 74, 52, 34, 68, 44, 58, 30, 64, 42, 76, 36]

function WaveformPlaceholder() {
  return (
    <div className="waveform" role="img" aria-label="Placeholder audio waveform">
      {BAR_HEIGHTS.map((height, index) => (
        <span
          key={index}
          className="waveform-bar"
          style={{ height: `${height}%` }}
        />
      ))}
    </div>
  )
}

export default WaveformPlaceholder
