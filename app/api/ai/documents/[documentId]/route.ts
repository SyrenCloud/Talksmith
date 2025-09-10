import { NextRequest, NextResponse } from 'next/server'

export async function DELETE(
  request: NextRequest,
  { params }: { params: { documentId: string } }
) {
  try {
    const { documentId } = params

    if (!documentId) {
      return NextResponse.json(
        { error: 'Document ID is required' },
        { status: 400 }
      )
    }

    // Forward the request to the AI service
    const aiServiceUrl = process.env.AI_SERVICE_URL || 'http://localhost:8000'
    
    const response = await fetch(`${aiServiceUrl}/documents/${documentId}`, {
      method: 'DELETE',
    })

    if (!response.ok) {
      const errorText = await response.text()
      console.error('AI service error:', errorText)
      throw new Error(`AI service responded with status ${response.status}`)
    }

    return NextResponse.json({
      success: true,
      message: 'Document deleted successfully'
    })

  } catch (error) {
    console.error('Delete document API error:', error)
    return NextResponse.json(
      { error: 'Failed to delete document' },
      { status: 500 }
    )
  }
}
