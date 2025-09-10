import { NextRequest, NextResponse } from 'next/server'

// Disable timeout for this API route to allow long-running document processing
export const maxDuration = 300 // 5 minutes max (Vercel limit)
export const dynamic = 'force-dynamic'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { file_paths } = body

    if (!file_paths || !Array.isArray(file_paths) || file_paths.length === 0) {
      return NextResponse.json(
        { error: 'File paths are required' },
        { status: 400 }
      )
    }

    // Forward the request to the AI service
    const aiServiceUrl = process.env.AI_SERVICE_URL || 'http://localhost:8000'
    
    const response = await fetch(`${aiServiceUrl}/process`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        file_paths
      }),
    })

    if (!response.ok) {
      const errorText = await response.text()
      console.error('AI service error:', errorText)
      throw new Error(`AI service responded with status ${response.status}`)
    }

    const data = await response.json()
    
    return NextResponse.json({
      document_ids: data.document_ids
    })

  } catch (error) {
    console.error('Process API error:', error)
    return NextResponse.json(
      { error: 'Failed to process documents' },
      { status: 500 }
    )
  }
}
