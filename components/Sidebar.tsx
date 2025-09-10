'use client'

import { useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, File, Trash2, FileText, FileImage, FileCode, Plus, X, PanelLeftClose } from 'lucide-react'
import { Document } from '@/types'

interface SidebarProps {
  documents: Document[]
  onDocumentUpload: (files: File[]) => void
  onDeleteDocument: (documentId: string) => void
  onNewChat?: () => void
  isOpen: boolean
  onToggle: () => void
  uploadProgress: {
    isUploading: boolean
    progress: number
    stage: 'uploading' | 'processing' | 'completed' | ''
    currentFile: string
    processedFiles: number
    totalFiles: number
    detailedMessage: string
  }
}

export default function Sidebar({ documents, onDocumentUpload, onDeleteDocument, uploadProgress, isOpen, onToggle, onNewChat }: SidebarProps) {
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop: (acceptedFiles) => {
      console.log('Files dropped:', acceptedFiles)
      if (acceptedFiles.length > 0) {
        onDocumentUpload(acceptedFiles)
      }
    },
    accept: {
      'application/pdf': ['.pdf'],
      'text/plain': ['.txt'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx']
    },
    multiple: true
  })

  // Function to get initials from full name (max 2 letters)
  const getInitials = (fullName: string): string => {
    const names = fullName.trim().split(' ').filter(name => name.length > 0)
    if (names.length === 0) return ''
    if (names.length === 1) return names[0].charAt(0).toUpperCase()
    return (names[0].charAt(0) + names[names.length - 1].charAt(0)).toUpperCase()
  }

  const getFileIcon = (type: string) => {
    if (type.includes('pdf')) return <FileText className="w-4 h-4 text-gray-500" />
    if (type.includes('word') || type.includes('document')) return <FileText className="w-4 h-4 text-gray-500" />
    if (type.includes('text')) return <FileCode className="w-4 h-4 text-gray-500" />
    return <File className="w-4 h-4 text-gray-500" />
  }

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  return (
    <div className={`${isOpen ? 'w-80' : 'w-16'} bg-white border-r border-gray-200 flex flex-col h-screen transition-all duration-300 shadow-sm ${isOpen ? '' : 'overflow-hidden'}`}>
      {/* Header */}
      <div className={`${isOpen ? 'px-6' : 'px-3'} py-4 border-b border-gray-200 flex-shrink-0`}>
        {isOpen ? (
          <>
            <div className="flex items-center justify-end mb-3">
              <button
                onClick={onToggle}
                className="p-1.5 hover:bg-gray-100 rounded-md transition-all duration-200 text-gray-500 hover:text-gray-700"
                title="Close sidebar"
              >
                <PanelLeftClose className="w-4 h-4" />
              </button>
            </div>
            <button
              onClick={onNewChat}
              className="w-full flex items-center justify-center space-x-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg px-4 py-2.5 transition-all duration-200 text-sm font-medium mb-3 shadow-sm"
            >
              <Plus className="w-4 h-4" />
              <span>New Chat</span>
            </button>
            <p className="text-gray-500 text-xs text-center">
              Upload documents to start chatting
            </p>
          </>
        ) : (
          <div className="flex flex-col items-center space-y-3">
            <button
              onClick={onToggle}
              className="p-2 hover:bg-gray-100 rounded-md transition-all duration-200 text-gray-500 hover:text-gray-700"
              title="Open sidebar"
            >
              <PanelLeftClose className="w-4 h-4 rotate-180" />
            </button>
            <button
              onClick={onNewChat}
              className="p-2 bg-blue-600 hover:bg-blue-700 rounded-md transition-all duration-200 text-white shadow-sm"
              title="New Chat"
            >
              <Plus className="w-4 h-4" />
            </button>
          </div>
        )}
      </div>

      {/* Upload Area */}
      {isOpen && (
        <div className="p-4 flex-shrink-0">
          <div
            {...getRootProps()}
            className={`upload-area ${isDragActive ? 'border-blue-400 bg-blue-50' : ''}`}
          >
            <input {...getInputProps()} />
            <Upload className="w-8 h-8 text-gray-400 mx-auto mb-2" />
            <p className="text-sm text-gray-700 font-medium mb-1">
              {isDragActive ? 'Drop files here' : 'Upload Documents'}
            </p>
            <p className="text-xs text-gray-500">
              PDF, TXT, DOCX supported
            </p>
          </div>

          {/* Upload Progress */}
          {uploadProgress.isUploading && (
            <div className="mt-4 p-3 bg-blue-50 rounded-lg border border-blue-200">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-700 font-medium">
                  {uploadProgress.stage === 'uploading' && 'Uploading...'}
                  {uploadProgress.stage === 'processing' && 'Processing...'}
                  {uploadProgress.stage === 'completed' && 'Completed!'}
                </span>
                <span className="text-xs text-gray-600">
                  {uploadProgress.progress}%
                </span>
              </div>
              
              {/* Detailed progress message */}
              {uploadProgress.detailedMessage && (
                <div className="mb-2">
                  <p className="text-xs text-gray-600 truncate">
                    {uploadProgress.detailedMessage}
                  </p>
                </div>
              )}
              
              {/* Current file being processed */}
              {uploadProgress.currentFile && uploadProgress.stage === 'processing' && (
                <div className="mb-2">
                  <p className="text-xs text-gray-700">
                    Processing: <span className="text-gray-600 truncate">{uploadProgress.currentFile}</span>
                  </p>
                </div>
              )}
              
              {/* Files counter */}
              {uploadProgress.totalFiles > 0 && uploadProgress.stage === 'processing' && (
                <div className="mb-2">
                  <p className="text-xs text-gray-600">
                    {uploadProgress.processedFiles} of {uploadProgress.totalFiles} files processed
                  </p>
                </div>
              )}
              
              <div className="w-full bg-gray-200 rounded-full h-1.5">
                <div 
                  className={`h-1.5 rounded-full transition-all duration-300 ${
                    uploadProgress.stage === 'completed' ? 'bg-green-500' : 'bg-blue-500'
                  }`}
                  style={{ width: `${uploadProgress.progress}%` }}
                />
              </div>
            </div>
          )}
        </div>
      )}

      {/* Documents List */}
      {isOpen && (
        <div className="flex-1 min-h-0 overflow-y-auto scrollbar-thin">
          <div className="p-4">
            <h3 className="text-sm font-semibold mb-3 text-gray-800">
              Documents ({documents.length})
            </h3>
            {documents.length === 0 ? (
              <div className="text-gray-500 text-sm text-center py-8">
                No documents uploaded yet
              </div>
            ) : (
              <div className="space-y-2">
                {documents.map((doc) => (
                  <div
                    key={doc.id}
                    className="document-item group"
                  >
                    <div className="flex items-center space-x-3">
                      <FileText className="w-4 h-4 text-gray-500 flex-shrink-0" />
                      <div className="flex-1 min-w-0">
                        <p className="text-sm text-gray-800 font-medium break-words">
                          {doc.name}
                        </p>
                        <p className="text-xs text-gray-500">
                          {(doc.size / 1024).toFixed(1)} KB
                        </p>
                      </div>
                      <button
                        onClick={() => onDeleteDocument(doc.id)}
                        className="opacity-0 group-hover:opacity-100 p-1 hover:bg-gray-100 rounded transition-all duration-200 text-gray-400 hover:text-red-500"
                        title="Delete document"
                      >
                        <Trash2 className="w-3 h-3" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Footer */}
      {isOpen && (
        <div className="px-6 py-4 border-t border-gray-200 flex-shrink-0">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center flex-shrink-0">
              <span className="text-sm font-medium text-white">{getInitials("Swarnava Dutta")}</span>
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-gray-800 truncate">Swarnava Dutta</p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
