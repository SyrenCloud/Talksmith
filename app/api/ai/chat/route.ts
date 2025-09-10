import { NextRequest, NextResponse } from 'next/server'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { message, documents, conversationHistory = [] } = body

    if (!message) {
      return NextResponse.json(
        { error: 'Message is required' },
        { status: 400 }
      )
    }

    if (!documents || documents.length === 0) {
      return NextResponse.json(
        { error: 'No documents available for chat' },
        { status: 400 }
      )
    }

    // Forward the request to the AI service
    const aiServiceUrl = process.env.AI_SERVICE_URL || 'http://localhost:8000'
    
    const response = await fetch(`${aiServiceUrl}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message,
        documents: documents,
        conversationHistory: conversationHistory
      }),
    })

    if (!response.ok) {
      const errorText = await response.text()
      console.error('AI service error:', response.status, errorText)
      console.error('Request details:', { message, documents, aiServiceUrl })
      return NextResponse.json(
        { error: `AI service error: ${response.status} - ${errorText}` },
        { status: response.status }
      )
    }

    const data = await response.json()
    
    return NextResponse.json({
      response: data.response
    })

  } catch (error) {
    console.error('Chat API error:', error)
    return NextResponse.json(
      { error: 'Failed to process chat message' },
      { status: 500 }
    )
  }
}
