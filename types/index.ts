export interface Message {
  id: string
  content: string
  role: 'user' | 'assistant'
  timestamp: Date
}

export interface Document {
  id: string
  backendFileId?: string // Backend file ID for deletion
  name: string
  size: number
  type: string
  uploadedAt: Date
  processed: boolean
}

export interface ChatResponse {
  response: string
  sources?: string[]
}

export interface UploadResponse {
  file_paths: string[]
  message: string
}

export interface ProcessResponse {
  document_ids: string[]
  message: string
}
