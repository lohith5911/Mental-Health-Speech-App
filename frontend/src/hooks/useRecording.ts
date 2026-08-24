import { useCallback, useEffect, useRef, useState } from 'react'
import { MAX_CHECK_IN_SECONDS, type CheckInStatus } from '../types'

const PREFERRED_MIME_TYPES = [
  'audio/webm;codecs=opus',
  'audio/webm',
  'audio/mp4',
  'audio/ogg;codecs=opus',
  'audio/ogg',
] as const

function formatTimer(totalSeconds: number) {
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`
}

function pickSupportedMimeType(): string | undefined {
  if (typeof MediaRecorder === 'undefined') {
    return undefined
  }

  return PREFERRED_MIME_TYPES.find((type) => MediaRecorder.isTypeSupported(type))
}

function stopStreamTracks(stream: MediaStream | null) {
  if (!stream) {
    return
  }

  for (const track of stream.getTracks()) {
    track.stop()
  }
}

function getRecordingErrorMessage(error: unknown): string {
  if (typeof MediaRecorder === 'undefined') {
    return 'This browser does not support audio recording. Try the latest Chrome, Edge, or Firefox.'
  }

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    return 'Microphone access is not available in this browser. Use a modern browser over http://localhost or https.'
  }

  if (error instanceof DOMException) {
    if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
      return 'Microphone permission was denied. Allow microphone access in your browser settings, then try again.'
    }

    if (error.name === 'NotFoundError' || error.name === 'DevicesNotFoundError') {
      return 'No microphone was found. Connect a microphone and try again.'
    }

    if (error.name === 'NotReadableError' || error.name === 'AbortError') {
      return 'The microphone is not available right now. Close other apps using it, then try again.'
    }

    if (error.name === 'SecurityError') {
      return 'This page is not allowed to use the microphone. Open the app on localhost or https.'
    }
  }

  return 'Recording could not be started. Check your microphone and try again.'
}

function useRecording() {
  const [status, setStatus] = useState<CheckInStatus>('idle')
  const [elapsedSeconds, setElapsedSeconds] = useState(0)
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null)
  const [audioUrl, setAudioUrl] = useState<string | null>(null)
  const [mimeType, setMimeType] = useState<string | null>(null)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [successMessage, setSuccessMessage] = useState<string | null>(null)
  const [isRequestingMic, setIsRequestingMic] = useState(false)

  const statusRef = useRef<CheckInStatus>('idle')
  const streamRef = useRef<MediaStream | null>(null)
  const recorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const previewUrlRef = useRef<string | null>(null)
  const mimeTypeRef = useRef<string | undefined>(undefined)
  const timerEnabledRef = useRef(false)
  const isMountedRef = useRef(true)
  const startLockRef = useRef(false)
  const stoppingRef = useRef(false)

  useEffect(() => {
    statusRef.current = status
  }, [status])

  const revokePreviewUrl = useCallback(() => {
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current)
      previewUrlRef.current = null
    }
  }, [])

  const clearLiveRecorder = useCallback(() => {
    const recorder = recorderRef.current
    recorderRef.current = null

    if (recorder && recorder.state !== 'inactive') {
      recorder.ondataavailable = null
      recorder.onerror = null
      recorder.onstop = null
      recorder.stop()
    }

    stopStreamTracks(streamRef.current)
    streamRef.current = null
    chunksRef.current = []
    timerEnabledRef.current = false
    stoppingRef.current = false
  }, [])

  const resetRecordingData = useCallback(() => {
    revokePreviewUrl()
    setAudioBlob(null)
    setAudioUrl(null)
    setMimeType(null)
    setErrorMessage(null)
    setSuccessMessage(null)
    mimeTypeRef.current = undefined
  }, [revokePreviewUrl])

  const finishRecording = useCallback(() => {
    stoppingRef.current = false
    stopStreamTracks(streamRef.current)
    streamRef.current = null
    recorderRef.current = null
    timerEnabledRef.current = false

    if (!isMountedRef.current) {
      chunksRef.current = []
      return
    }

    const chunks = chunksRef.current
    chunksRef.current = []
    const type = mimeTypeRef.current || chunks[0]?.type || 'audio/webm'
    const blob = new Blob(chunks, { type })

    if (blob.size === 0) {
      setErrorMessage('Recording failed. Please try again.')
      setStatus('idle')
      resetRecordingData()
      return
    }

    const objectUrl = URL.createObjectURL(blob)
    previewUrlRef.current = objectUrl
    setAudioBlob(blob)
    setAudioUrl(objectUrl)
    setMimeType(type)
    setStatus('recorded')
  }, [resetRecordingData])

  const stopRecording = useCallback(() => {
    if (statusRef.current !== 'recording' || stoppingRef.current) {
      return
    }

    stoppingRef.current = true
    timerEnabledRef.current = false
    const recorder = recorderRef.current

    if (!recorder || recorder.state === 'inactive') {
      finishRecording()
      return
    }

    recorder.stop()
  }, [finishRecording])

  const startRecording = useCallback(async () => {
    if (startLockRef.current || statusRef.current === 'recording') {
      return
    }

    startLockRef.current = true
    setErrorMessage(null)

    if (!navigator.mediaDevices || typeof navigator.mediaDevices.getUserMedia !== 'function') {
      startLockRef.current = false
      setErrorMessage(
        'Microphone access is not available in this browser. Use a modern browser over http://localhost or https.',
      )
      return
    }

    if (typeof MediaRecorder === 'undefined') {
      startLockRef.current = false
      setErrorMessage(
        'This browser does not support audio recording. Try the latest Chrome, Edge, or Firefox.',
      )
      return
    }

    setIsRequestingMic(true)

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })

      if (!isMountedRef.current) {
        stopStreamTracks(stream)
        return
      }

      resetRecordingData()
      chunksRef.current = []
      stoppingRef.current = false

      const supportedType = pickSupportedMimeType()
      const recorder = supportedType
        ? new MediaRecorder(stream, { mimeType: supportedType })
        : new MediaRecorder(stream)

      mimeTypeRef.current = recorder.mimeType || supportedType
      streamRef.current = stream
      recorderRef.current = recorder

      recorder.ondataavailable = (event: BlobEvent) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data)
        }
      }

      recorder.onerror = () => {
        stoppingRef.current = false
        clearLiveRecorder()
        if (!isMountedRef.current) {
          return
        }
        setErrorMessage('Recording failed. Please try again.')
        setStatus('idle')
      }

      recorder.onstop = () => {
        finishRecording()
      }

      try {
        recorder.start(250)
      } catch {
        recorder.start()
      }

      timerEnabledRef.current = true
      setElapsedSeconds(0)
      setStatus('recording')
    } catch (error) {
      stoppingRef.current = false
      clearLiveRecorder()
      if (isMountedRef.current) {
        setErrorMessage(getRecordingErrorMessage(error))
        setStatus('idle')
      }
    } finally {
      startLockRef.current = false
      if (isMountedRef.current) {
        setIsRequestingMic(false)
      }
    }
  }, [clearLiveRecorder, finishRecording, resetRecordingData])

  const recordAgain = useCallback(() => {
    clearLiveRecorder()
    resetRecordingData()
    setElapsedSeconds(0)
    setErrorMessage(null)
    setStatus('idle')
  }, [clearLiveRecorder, resetRecordingData])

  const continueToProcessing = useCallback(async () => {
    if (statusRef.current !== 'recorded' || !audioBlob) {
      return
    }

    setErrorMessage(null)
    setSuccessMessage(null)
    setStatus('processing')

    try {
      const formData = new FormData()
      const extension = audioBlob.type.includes('webm')
        ? 'webm'
        : audioBlob.type.includes('mp4')
          ? 'mp4'
          : audioBlob.type.includes('ogg')
            ? 'ogg'
            : audioBlob.type.includes('wav')
              ? 'wav'
              : audioBlob.type.includes('mpeg') || audioBlob.type.includes('mp3')
                ? 'mp3'
                : 'webm'

      formData.append('file', audioBlob, `check-in-${Date.now()}.${extension}`)

      const response = await fetch('http://127.0.0.1:8000/api/check-ins', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errorBody = await response.json().catch(() => null)
        throw new Error(
          errorBody?.detail || 'The recording could not be uploaded. Please try again.',
        )
      }

      const payload = (await response.json()) as {
        check_in_id?: string
        status?: string
        filename?: string
      }

      if (!payload.check_in_id) {
        throw new Error('The backend did not return a valid check-in ID.')
      }

      setSuccessMessage(
        'Voice recording uploaded successfully. Analysis will be added in the next milestone.',
      )
      setStatus('recorded')
    } catch (error) {
      setStatus('recorded')
      setErrorMessage(
        error instanceof Error
          ? error.message
          : 'The recording could not be uploaded. Check your connection and try again.',
      )
    }
  }, [audioBlob])

  useEffect(() => {
    if (status !== 'recording') {
      return
    }

    const timerId = window.setInterval(() => {
      if (!timerEnabledRef.current) {
        return
      }

      setElapsedSeconds((current) => Math.min(current + 1, MAX_CHECK_IN_SECONDS))
    }, 1000)

    return () => {
      window.clearInterval(timerId)
    }
  }, [status])

  useEffect(() => {
    if (status === 'recording' && elapsedSeconds >= MAX_CHECK_IN_SECONDS) {
      stopRecording()
    }
  }, [elapsedSeconds, status, stopRecording])

  useEffect(() => {
    isMountedRef.current = true

    return () => {
      isMountedRef.current = false
      timerEnabledRef.current = false
      clearLiveRecorder()
      revokePreviewUrl()
    }
  }, [clearLiveRecorder, revokePreviewUrl])

  return {
    status,
    elapsedSeconds,
    formattedTime: formatTimer(elapsedSeconds),
    audioBlob,
    audioUrl,
    mimeType,
    errorMessage,
    successMessage,
    isRequestingMic,
    startRecording,
    stopRecording,
    recordAgain,
    continueToProcessing,
  }
}

export default useRecording
