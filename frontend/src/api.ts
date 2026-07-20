import type { ApiError, ChatResponse } from './types'

const normalizeBaseUrl = (value: string): string => value.trim().replace(/\/$/, '')

export const apiBaseUrl = normalizeBaseUrl(import.meta.env.VITE_API_BASE_URL || '')

const apiUrl = (path: string): string => `${apiBaseUrl}${path}`

const parseError = async (response: Response): Promise<Error> => {
  try {
    const data = (await response.json()) as ApiError
    return new Error(data.message || `${response.status} ${response.statusText}`)
  } catch {
    return new Error(`${response.status} ${response.statusText}`)
  }
}

export const checkHealth = async (): Promise<boolean> => {
  try {
    const response = await fetch(apiUrl('/api/health'), {
      headers: { Accept: 'application/json' },
    })
    return response.ok
  } catch {
    return false
  }
}

export interface SendMessageInput {
  query: string
  userId: string
  projectId: string
  sessionId: string
}

export const sendMessage = async (input: SendMessageInput): Promise<ChatResponse> => {
  const response = await fetch(apiUrl('/api/chat'), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
    },
    body: JSON.stringify({
      query: input.query,
      user_id: input.userId || null,
      project_id: input.projectId || null,
      session_id: input.sessionId || null,
      include_memory_status: true,
      options: {
        include_citations: true,
      },
    }),
  })

  if (!response.ok) {
    throw await parseError(response)
  }

  return (await response.json()) as ChatResponse
}
