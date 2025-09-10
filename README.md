# Talksmith - Document Analysis Platform

<div align="center">
  <img src="assets/Syren.webp" alt="Talksmith Logo" width="200"/>
  
  [![Next.js](https://img.shields.io/badge/Next.js-14.0-black?style=for-the-badge&logo=next.js)](https://nextjs.org/)
  [![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
  [![OpenAI](https://img.shields.io/badge/OpenAI-GPT--5-412991?style=for-the-badge&logo=openai)](https://openai.com/)
  [![TypeScript](https://img.shields.io/badge/TypeScript-5.0-3178C6?style=for-the-badge&logo=typescript)](https://www.typescriptlang.org/)
  
  *RAG system for document analysis and conversations*
</div>

## 📋 Overview

Talksmith is a document analysis platform that allows users to interact with their documents through natural language conversations. It uses AI technology to process and query documents.

### ✨ Key Features

- **RAG Architecture**: Multi-stage retrieval system with cross-encoder reranking
- **Multi-Format Support**: Process PDF, DOCX, TXT, and Markdown files
- **Real-time Processing**: Stream-based document processing with progress updates
- **Context-aware Search**: Search with conversation history integration
- **Security**: Isolated vector stores with document-level access control
- **Monitoring**: Performance metrics and health checks


## 🛠️ Technical Stack

### Frontend
- **Framework**: Next.js 14 with App Router
- **Language**: TypeScript 5
- **Styling**: Tailwind CSS
- **State Management**: React Hooks
- **File Handling**: react-dropzone

### Backend
- **API Framework**: FastAPI with async support
- **AI Integration**: LangChain with custom chains
- **Vector Database**: ChromaDB with persistent storage
- **Document Processing**: Asynchronous pipeline
- **Reranking**: Cross-encoder MS-MARCO model

## 📋 Prerequisites

Before deployment, ensure you have:

- **Node.js** 18.0+ and npm/yarn
- **Python** 3.9+
- **OpenAI API Key** with GPT-5 access
- **Windows/Linux/macOS** environment
- **8GB+ RAM** recommended

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/talksmith.git
cd talksmith
```

### 2. Environment Configuration

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_api_key_here
MODEL_NAME=gpt-5-nano-2025-08-07
TEMPERATURE=1.0
```

### 3. Install Dependencies

**Windows Users:**
```batch
# Run the automated installer
install-fullstack-dependencies.bat
```

**Manual Installation:**
```bash
# Frontend dependencies
npm install

# AI Service dependencies
cd ai-service
pip install -r requirements.txt
```

### 4. Launch the Application

**Windows Users:**
```batch
# Start both services
start-fullstack.bat
```

**Manual Start:**
```bash
# Terminal 1: Frontend
npm run dev

# Terminal 2: AI Service
cd ai-service
python main.py
```

Access the application at `http://localhost:3000`



## 💡 Usage Guide

### 1. Document Upload
- Drag and drop files into the sidebar
- Support for batch uploads (up to 10 files)
- Real-time processing progress
- Automatic file organization

### 2. Intelligent Conversations
- Natural language queries
- Context-aware responses
- Conversation history tracking
- Source attribution

### 3. Document Management
- View processed documents
- Delete individual documents
- Monitor processing status
- Export conversation history

## 🔧 Advanced Features

### Cross-Encoder Reranking
The system uses a two-stage retrieval process:

1. **Initial Retrieval**: Similarity search using embeddings
2. **Reranking**: Cross-encoder scoring

### Conversation Context
- Maintains conversation history for contextual understanding
- Supports references like "the previous method" or "that approach"
- Context window management

### Stream Processing
- Server-Sent Events for real-time updates
- Non-blocking document processing
- Progress tracking with detailed status

## 🔍 Troubleshooting

### Common Issues

#### 1. OpenAI API Errors
```
Error: OPENAI_API_KEY not found
```
**Solution**: Ensure `.env` file is properly configured in `ai-service` directory

#### 2. Document Processing Failures
```
Error: Failed to process document
```
**Solution**: Check file format and size (max 10MB per file)

#### 3. Vector Store Issues
```
Error: ChromaDB connection failed
```
**Solution**: Delete `chroma_db` folder and restart the service

## 🔒 Security Considerations

- **API Keys**: Never commit `.env` files to version control
- **File Access**: Documents are isolated by session
- **Data Privacy**: All processing happens locally
- **Network Security**: Configure CORS settings for production

## ⚡ Performance Optimization

### Recommended Settings
- **Chunk Size**: 1000 characters
- **Overlap**: 200 characters
- **Batch Size**: 100 documents
- **Rerank Candidates**: 20 chunks

### Scaling Considerations
- Implement Redis for session management
- Use PostgreSQL for metadata storage
- Deploy with Docker for containerization
- Configure nginx for load balancing

## 🤝 Contributing

Contributions are welcome in the following areas:

1. **UI/UX Improvements**: Frontend enhancements
2. **Documentation**: Clarifications and examples
3. **Testing**: Unit and integration tests
4. **Performance**: Optimization suggestions

Please discuss major architectural changes before implementation.

## 📞 Support

For technical support:

- **Email**: swarnava.d@syrencloud.com
- **Issues**: GitHub Issues (for approved contributors)

## 📄 License

© 2025 - All Rights Reserved

---

<div align="center">

### 👋 Let's Connect  

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/swarnava-dutta)  

</div>

