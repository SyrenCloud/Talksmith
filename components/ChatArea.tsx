'use client'

import { useState, useRef, useEffect } from 'react'
import { Send, User, Loader2, PanelLeftOpen } from 'lucide-react'
import { Message } from '@/types'
import Image from 'next/image'
import SyrenImage from '@/assets/Syren.webp'

interface ChatAreaProps {
  messages: Message[]
  onSendMessage: (message: string) => void
  isLoading: boolean
  hasDocuments: boolean
  isSidebarOpen: boolean
  onToggleSidebar: () => void
}

export default function ChatArea({ messages, onSendMessage, isLoading, hasDocuments, isSidebarOpen, onToggleSidebar }: ChatAreaProps) {
  const [input, setInput] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (input.trim() && !isLoading) {
      onSendMessage(input.trim())
      setInput('')
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`
    }
  }, [input])

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    if (messagesEndRef.current) {
      // Small delay to ensure DOM has updated
      setTimeout(() => {
        messagesEndRef.current?.scrollIntoView({ 
          behavior: 'smooth',
          block: 'end',
          inline: 'nearest'
        })
      }, 100)
    }
  }, [messages, isLoading])

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  }

  return (
    <div className="flex-1 flex flex-col bg-gray-50 h-screen">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 px-6 py-4 flex-shrink-0 shadow-md">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 rounded-full overflow-hidden">
              <Image src={SyrenImage} alt="Talksmith AI" width={32} height={32} className="w-full h-full object-cover" />
            </div>
            <div>
              <h1 className="text-lg font-semibold text-gray-900">Talksmith AI</h1>
              <p className="text-sm text-gray-600">Your intelligent document assistant</p>
            </div>
          </div>
        </div>
      </div>

      {/* Chat Messages */}
      <div className="flex-1 min-h-0 overflow-y-auto scrollbar-thin px-6 py-8 bg-gradient-to-b from-gray-50 to-gray-100">
        {messages.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center max-w-md">
              <div className="w-16 h-16 mx-auto mb-4 rounded-full overflow-hidden">
                <Image src={SyrenImage} alt="Talksmith AI" width={64} height={64} className="w-full h-full object-cover" />
              </div>
              <h2 className="text-2xl font-bold text-gray-900 mb-3">
                Welcome to Talksmith AI
              </h2>
              <p className="text-gray-600 mb-6 text-base leading-relaxed">
                Upload documents in the sidebar and start asking questions about their content.
              </p>
              {!hasDocuments && (
                <div className="bg-amber-50 border-2 border-amber-300 rounded-xl p-4 shadow-sm">
                  <p className="text-amber-800 text-sm font-semibold">
                    Please upload documents first to start chatting
                  </p>
                </div>
              )}
            </div>
          </div>
        ) : (
          <div className="space-y-4 max-w-5xl mx-auto pt-4 pb-8 px-4">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex items-start gap-3 ${
                  message.role === 'user' ? 'justify-end' : 'justify-start'
                }`}
              >
                <div className={`flex items-start gap-3 max-w-[85%] ${
                  message.role === 'user' ? 'flex-row-reverse' : ''
                }`}>
                  <div className={`flex-shrink-0 w-8 h-8 rounded-full ${
                    message.role === 'user' 
                      ? 'bg-blue-600 shadow-md flex items-center justify-center' 
                      : 'overflow-hidden border-2 border-gray-300 shadow-sm'
                  }`}>
                    {message.role === 'user' ? (
                      <User className="w-4 h-4 text-white" />
                    ) : (
                      <Image src={SyrenImage} alt="AI" width={32} height={32} className="w-full h-full object-cover" />
                    )}
                  </div>
                  
                  <div className="flex flex-col gap-1">
                    <div className={`chat-message ${
                      message.role === 'user' ? 'user-message' : 'bot-message'
                    }`}>
                      <p className="whitespace-pre-wrap leading-relaxed">{message.content}</p>
                    </div>
                    <p className={`text-xs text-gray-500 ${
                      message.role === 'user' ? 'text-right pr-1' : 'text-left pl-1'
                    }`}>
                      {formatTime(message.timestamp)}
                    </p>
                  </div>
                </div>
              </div>
            ))}
            
            {isLoading && (
              <div className="flex items-start gap-3 justify-start">
                <div className="flex items-start gap-3 max-w-[85%]">
                  <div className="flex-shrink-0 w-8 h-8 rounded-full bg-white border-2 border-gray-300 shadow-sm overflow-hidden">
                    <Image src={SyrenImage} alt="AI" width={32} height={32} className="w-full h-full object-cover" />
                  </div>
                  <div className="bot-message flex items-center gap-3 px-4 py-3 min-w-[80px]">
                    <div className="flex items-center">
                      <span className="typing-dot"></span>
                      <span className="typing-dot"></span>
                      <span className="typing-dot"></span>
                    </div>
                  </div>
                </div>
              </div>
            )}
            
            {/* Invisible div to scroll to */}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input Area */}
      <div className="border-t border-gray-200 bg-white p-4 flex-shrink-0 shadow-lg">
        <form onSubmit={handleSubmit} className="max-w-4xl mx-auto">
          <div className="flex items-start space-x-3">
            <div className="flex-1 relative">
              <textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={hasDocuments ? "Ask a question about your documents..." : "Upload documents first to start chatting"}
                disabled={!hasDocuments || isLoading}
                className="w-full bg-white border-2 border-gray-300 rounded-xl px-5 py-3 text-gray-900 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-none h-12 max-h-32 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 scrollbar-none hover:border-gray-400 shadow-sm"
                rows={1}
              />
            </div>
            <button
              type="submit"
              disabled={!input.trim() || !hasDocuments || isLoading}
              className="flex-shrink-0 w-12 h-12 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed text-white rounded-xl flex items-center justify-center transition-all duration-200 disabled:opacity-50 shadow-md hover:shadow-lg transform hover:-translate-y-0.5"
            >
              {isLoading ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Send className="w-4 h-4" />
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
