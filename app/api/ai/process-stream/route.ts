import { NextRequest } from 'next/server'

// Disable timeout for this API route to allow long-running document processing
export const maxDuration = 300 // 5 minutes max (Vercel limit)
export const dynamic = 'force-dynamic'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { file_paths } = body

    if (!file_paths || !Array.isArray(file_paths) || file_paths.length === 0) {
      return new Response(
        JSON.stringify({ error: 'File paths are required' }),
        { status: 400, headers: { 'Content-Type': 'application/json' } }
      )
    }

    // Forward the request to the AI service streaming endpoint with no timeout
    const aiServiceUrl = process.env.AI_SERVICE_URL || 'http://localhost:8000'
    
    const response = await fetch(`${aiServiceUrl}/process-stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        file_paths
      }),
      // No timeout - let processing run as long as needed
    })

    if (!response.ok) {
      const errorText = await response.text()
      console.error('AI service error:', errorText)
      return new Response(
        JSON.stringify({ error: `AI service responded with status ${response.status}` }),
        { status: response.status, headers: { 'Content-Type': 'application/json' } }
      )
    }

    // Stream the response from AI service to client
    return new Response(response.body, {
      headers: {
        'Content-Type': 'text/plain',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': '*',
      },
    })

  } catch (error) {
    console.error('Process stream API error:', error)
    return new Response(
      JSON.stringify({ error: 'Failed to process documents' }),
      { status: 500, headers: { 'Content-Type': 'application/json' } }
    )
  }
}
