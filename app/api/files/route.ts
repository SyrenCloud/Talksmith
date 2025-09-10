import { NextResponse } from 'next/server';
import { FileStorage } from '@/lib/fileStorage';

export async function GET() {
  try {
    const files = FileStorage.getAllFiles();
    return NextResponse.json({ files });
  } catch (error) {
    console.error('Error fetching files:', error);
    return NextResponse.json({ error: 'Failed to fetch files' }, { status: 500 });
  }
}
