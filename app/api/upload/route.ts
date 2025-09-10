import { NextRequest, NextResponse } from 'next/server';
import { writeFile } from 'fs/promises';
import path from 'path';
import { FileStorage, FileMetadata } from '@/lib/fileStorage';
import { validateFileType, validateFileSize } from '@/lib/multerConfig';

export async function POST(request: NextRequest) {
  try {
    console.log('Upload API called');
    const formData = await request.formData();
    console.log('FormData entries:', Array.from(formData.entries()).map(([key, value]) => [key, value instanceof File ? `File: ${value.name} (${value.type})` : value]));
    
    const files = formData.getAll('files') as File[];
    console.log('Files extracted:', files.length, files.map(f => ({ name: f.name, type: f.type, size: f.size })));

    if (!files || files.length === 0) {
      console.log('No files found in request');
      return NextResponse.json({ error: 'No files uploaded' }, { status: 400 });
    }

    const filePaths: string[] = [];
    const fileMetadata: FileMetadata[] = [];

    for (const file of files) {
      console.log(`Processing file: ${file.name}, type: ${file.type}, size: ${file.size}`);
      
      // Validate file type
      if (!validateFileType(file.type)) {
        console.log(`File type validation failed for ${file.name}: ${file.type}`);
        return NextResponse.json({ 
          error: `Invalid file type for ${file.name}. Only PDF, DOCX, TXT, and MD files are allowed. Received: ${file.type}` 
        }, { status: 400 });
      }

      // Validate file size (10MB limit)
      if (!validateFileSize(file.size)) {
        console.log(`File size validation failed for ${file.name}: ${file.size} bytes`);
        return NextResponse.json({ 
          error: `File too large: ${file.name} (${Math.round(file.size / 1024 / 1024)}MB). Maximum size is 10MB.` 
        }, { status: 400 });
      }

      const uniqueFilename = FileStorage.generateUniqueFilename(file.name);
      const fileId = path.parse(uniqueFilename).name;
      const filePath = path.join(FileStorage.getUploadsDir(), uniqueFilename);

      console.log(`Generated unique filename: ${uniqueFilename}, path: ${filePath}`);

      // Convert file to buffer and write to disk
      const bytes = await file.arrayBuffer();
      const buffer = Buffer.from(bytes);
      console.log(`Writing file to disk: ${filePath}, buffer size: ${buffer.length}`);
      await writeFile(filePath, buffer);
      console.log(`File written successfully: ${filePath}`);

      // Create metadata
      const metadata: FileMetadata = {
        id: fileId,
        originalName: file.name,
        filename: uniqueFilename,
        path: filePath,
        size: file.size,
        mimetype: file.type,
        uploadedAt: new Date().toISOString()
      };

      // Save metadata
      FileStorage.saveFileMetadata(metadata);
      filePaths.push(filePath);
      fileMetadata.push(metadata);
    }

    return NextResponse.json({
      message: 'Files uploaded successfully',
      file_paths: filePaths,
      files: fileMetadata
    });

  } catch (error) {
    console.error('Upload error:', error);
    return NextResponse.json({ error: 'Failed to upload files' }, { status: 500 });
  }
}
