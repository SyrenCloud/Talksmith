'use client'

import { useState, useRef, useEffect } from 'react'
import Sidebar from '@/components/Sidebar'
import ChatArea from '@/components/ChatArea'
import { Message, Document } from '@/types'

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([])
  const [documents, setDocuments] = useState<Document[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [isSidebarOpen, setIsSidebarOpen] = useState(true)
  const [uploadProgress, setUploadProgress] = useState({
    isUploading: false,
    progress: 0,
    stage: '' as 'uploading' | 'processing' | 'completed' | '',
    currentFile: '',
    processedFiles: 0,
    totalFiles: 0,
    detailedMessage: ''
  })
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSendMessage = async (content: string) => {
    if (!content.trim() || isLoading) return

    // Check if documents are uploaded
    if (documents.length === 0) {
      alert('Please upload and process documents first!')
      return
    }

    // Filter out any documents with null/undefined IDs
    const validDocuments = documents.filter(doc => doc.id != null && doc.id !== undefined)
    if (validDocuments.length === 0) {
      alert('No valid documents found. Please re-upload and process your documents.')
      return
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      content,
      role: 'user',
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setIsLoading(true)

    try {
      const response = await fetch('/api/ai/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: content,
          documents: validDocuments.map(doc => doc.id),
          conversationHistory: messages.slice(-6).map(msg => ({
            role: msg.role,
            content: msg.content
          }))
        }),
      })

      if (!response.ok) {
        const errorText = await response.text()
        console.error('API Error Response:', response.status, errorText)
        throw new Error(`Failed to get response: ${response.status} - ${errorText}`)
      }

      const data = await response.json()
      
      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: data.response,
        role: 'assistant',
        timestamp: new Date()
      }

      setMessages(prev => [...prev, botMessage])
    } catch (error) {
      console.error('Error sending message:', error)
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: 'Sorry, I encountered an error. Please try again.',
        role: 'assistant',
        timestamp: new Date()
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const handleDocumentUpload = async (files: File[]) => {
    if (uploadProgress.isUploading) return // Prevent multiple uploads
    
    const formData = new FormData()
    files.forEach(file => {
      formData.append('files', file)
    })

    try {
      // Start upload progress
      setUploadProgress({
        isUploading: true,
        progress: 10,
        stage: 'uploading',
        currentFile: '',
        processedFiles: 0,
        totalFiles: files.length,
        detailedMessage: 'Uploading files...'
      })

      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error('Failed to upload documents')
      }

      // Upload completed
      setUploadProgress(prev => ({
        ...prev,
        progress: 30,
        stage: 'processing',
        detailedMessage: 'Files uploaded, starting processing...'
      }))

      const data = await response.json()
      
      // Start streaming processing with no timeout
      const streamResponse = await fetch('/api/ai/process-stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          file_paths: data.file_paths
        }),
        // No timeout - allow processing to run indefinitely
      })

      if (!streamResponse.ok) {
        throw new Error('Failed to start document processing')
      }

      // Handle streaming response
      const reader = streamResponse.body?.getReader()
      const decoder = new TextDecoder()
      let documentIds: (string | null)[] = []

      if (reader) {
        try {
          while (true) {
            const { done, value } = await reader.read()
            if (done) break

            const chunk = decoder.decode(value)
            const lines = chunk.split('\n')

            for (const line of lines) {
              if (line.startsWith('data: ')) {
                try {
                  const eventData = JSON.parse(line.slice(6))
                  
                  setUploadProgress(prev => ({
                    ...prev,
                    progress: eventData.progress || prev.progress,
                    currentFile: eventData.current_file || prev.currentFile,
                    processedFiles: eventData.processed_files || prev.processedFiles,
                    totalFiles: eventData.total_files || prev.totalFiles,
                    detailedMessage: eventData.message || prev.detailedMessage,
                    stage: eventData.type === 'complete' ? 'completed' : 'processing'
                  }))

                  if (eventData.type === 'complete') {
                    documentIds = eventData.document_ids || []
                    break
                  }
                  
                  if (eventData.type === 'error') {
                    throw new Error(eventData.message)
                  }
                } catch (parseError) {
                  console.warn('Failed to parse SSE data:', line)
                }
              }
            }
          }
        } finally {
          reader.releaseLock()
        }
      }

      console.log('Document IDs from streaming:', documentIds)
      console.log('File paths from upload:', data.file_paths)
      
      const newDocuments: Document[] = data.file_paths
        .map((path: string, index: number) => ({
          id: documentIds[index], // AI service document ID
          backendFileId: data.files[index].id, // Backend file ID for deletion
          name: files[index].name,
          size: files[index].size,
          type: files[index].type,
          uploadedAt: new Date(),
          processed: true
        }))
        .filter((doc: Document) => doc.id != null && doc.id !== undefined) // Filter out documents with null IDs

      setDocuments(prev => [...prev, ...newDocuments])
      
      // Show warning if some documents failed to process
      const failedCount = data.file_paths.length - newDocuments.length
      if (failedCount > 0) {
        console.warn(`${failedCount} documents failed to process`)
        alert(`Warning: ${failedCount} out of ${data.file_paths.length} documents failed to process. You can still chat with the successfully processed documents.`)
      }
      
      // Final completion state
      setUploadProgress(prev => ({
        ...prev,
        progress: 100,
        stage: 'completed',
        detailedMessage: `Successfully processed ${newDocuments.length} out of ${files.length} documents`
      }))

      // Reset after a short delay
      setTimeout(() => {
        setUploadProgress({
          isUploading: false,
          progress: 0,
          stage: '',
          currentFile: '',
          processedFiles: 0,
          totalFiles: 0,
          detailedMessage: ''
        })
      }, 3000)

    } catch (error) {
      console.error('Error uploading documents:', error)
      const errorMessage = error instanceof Error ? error.message : 'Failed to upload documents. Please try again.'
      alert(errorMessage)
      
      // Reset progress on error
      setUploadProgress({
        isUploading: false,
        progress: 0,
        stage: '',
        currentFile: '',
        processedFiles: 0,
        totalFiles: 0,
        detailedMessage: ''
      })
    }
  }

  const handleDeleteDocument = async (documentId: string) => {
    try {
      // Find the document to get both IDs
      const document = documents.find(doc => doc.id === documentId)
      if (!document) {
        throw new Error('Document not found')
      }

      // Delete from AI service first (remove from vector store)
      const aiResponse = await fetch(`/api/ai/documents/${documentId}`, {
        method: 'DELETE',
      })

      if (!aiResponse.ok) {
        console.warn('Failed to delete from AI service, continuing with backend deletion')
      }

      // Delete from backend (remove physical file) using the backend file ID
      if (document.backendFileId) {
        const backendResponse = await fetch(`/api/documents/${document.backendFileId}`, {
          method: 'DELETE',
        })

        if (!backendResponse.ok) {
          console.warn('Failed to delete from backend service')
        }
      }

      // Remove from frontend state
      setDocuments(prev => prev.filter(doc => doc.id !== documentId))
      
    } catch (error) {
      console.error('Error deleting document:', error)
      alert('Failed to delete document. Please try again.')
    }
  }

  const handleNewChat = () => {
    setMessages([])
  }

  const toggleSidebar = () => {
    setIsSidebarOpen(prev => !prev)
  }

  return (
    <div className="flex h-screen bg-gray-50">
      <Sidebar 
        documents={documents}
        onDocumentUpload={handleDocumentUpload}
        onDeleteDocument={handleDeleteDocument}
        onNewChat={handleNewChat}
        isOpen={isSidebarOpen}
        onToggle={toggleSidebar}
        uploadProgress={uploadProgress}
      />
      <div className="flex-1 flex flex-col">
        <ChatArea 
          messages={messages}
          onSendMessage={handleSendMessage}
          isLoading={isLoading}
          hasDocuments={documents.length > 0}
          isSidebarOpen={isSidebarOpen}
          onToggleSidebar={toggleSidebar}
        />
        <div ref={messagesEndRef} />
      </div>
    </div>
  )
}
