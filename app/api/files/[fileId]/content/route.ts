import { NextRequest, NextResponse } from 'next/server';
import { FileStorage } from '@/lib/fileStorage';
import fs from 'fs';

export async function GET(
  request: NextRequest,
  { params }: { params: { fileId: string } }
) {
  try {
    const { fileId } = params;
    
    if (!FileStorage.fileExists(fileId)) {
      return NextResponse.json({ error: 'File not found' }, { status: 404 });
    }

    const fileMetadata = FileStorage.getFileMetadata(fileId);
    
    if (!fileMetadata || !fs.existsSync(fileMetadata.path)) {
      return NextResponse.json({ error: 'Physical file not found' }, { status: 404 });
    }

    return NextResponse.json({
      metadata: fileMetadata,
      path: fileMetadata.path
    });
    
  } catch (error) {
    console.error('Error fetching file content:', error);
    return NextResponse.json({ error: 'Failed to fetch file content' }, { status: 500 });
  }
}
