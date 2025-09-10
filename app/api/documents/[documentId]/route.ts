import { NextRequest, NextResponse } from 'next/server';
import { FileStorage } from '@/lib/fileStorage';

export async function DELETE(
  request: NextRequest,
  { params }: { params: { documentId: string } }
) {
  try {
    const { documentId } = params;
    
    if (!FileStorage.fileExists(documentId)) {
      return NextResponse.json({ error: 'Document not found' }, { status: 404 });
    }

    const success = FileStorage.deleteFile(documentId);
    
    if (!success) {
      return NextResponse.json({ error: 'Failed to delete document' }, { status: 500 });
    }
    
    return NextResponse.json({ message: 'Document deleted successfully' });
    
  } catch (error) {
    console.error('Delete error:', error);
    return NextResponse.json({ error: 'Failed to delete document' }, { status: 500 });
  }
}
