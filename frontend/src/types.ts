export interface Citation {
  evidence_id?: string
  source?: string
  text?: string
  score?: number
  [key: string]: unknown
}

export interface MemoryStatus {
  loaded: boolean
  written: boolean
  recent_message_count: number
  retrieved_memory_count: number
  estimated_tokens: number
  summary_used: boolean
  written_count: number
  skipped_count: number
}

export interface ChatResponse {
  answer: string
  request_id: string
  session_id: string
  user_id: string
  project_id: string
  route: string
  answerability: string
  semantic_score: number
  citations: Citation[]
  traces: Record<string, unknown>[]
  warnings: string[]
  has_error: boolean
  error_message: string
  memory_status?: MemoryStatus | null
}

export interface ApiError {
  error_code?: string
  message?: string
  request_id?: string
  details?: Record<string, unknown>
}

export interface Message {
  id: string
  role: 'user' | 'assistant' | 'error'
  content: string
  response?: ChatResponse
}
