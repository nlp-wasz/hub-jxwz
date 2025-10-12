# RAG System - Intelligent Q&A System

A Retrieval-Augmented Generation (RAG) based intelligent question answering system that supports both local knowledge base retrieval and real-time web search.

## ✨ Features

- **Smart Routing**: Automatically decides between local knowledge base or web search
- **Multi-source Retrieval**: Combines vector database with real-time web information
- **Beautiful Interface**: React frontend with Markdown rendering and code highlighting
- **API Support**: RESTful API interfaces for easy integration

## 🏗️ System Architecture
Frontend (React + Vite) → Backend (FastAPI) → Tongyi Qianwen API
                                              ↓                  ↓
                                           Vector Database       Web Search (Tavily)


## 📦 Installation & Deployment

### Backend Setup

```bash
# Clone project
git clone <your-repo-url>
cd rag-project/backend

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env file with your API keys

# Start service
python rag_test.py
```

### Frontend Setup
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev

### Environment Configuration
Configure in backend/.env file:
    DASHSCOPE_API_KEY=your_alibaba_cloud_api_key
    TAVILY_API_KEY=your_tavily_api_key
    USER_AGENT=rag_test

### Quick Start
Access Application: Open http://localhost:5173

Ask Questions: Enter your question in the input box

Get Answers: System automatically selects the best data source and provides answers

### Knowledge Base Content
System preloads the following technical documentation:

Agent Systems

Prompt Engineering

Adversarial Attacks on LLMs

### API Endpoints
- Ask Question:
POST /api/ask
Content-Type: application/json

{
  "question": "What is a language model?"
}

- Health Check
GET /api/health

- Response
{
  "status": "healthy",
  "service": "RAG System API"
}

### Technology Stack
Backend
FastAPI - Web framework
ChromaDB - Vector database
LangChain - LLM application framework
Tavily - Web search API
Tongyi Qianwen - Large language model

Frontend
React 18 - User interface
Vite - Build tool
Axios - HTTP client
ReactMarkdown - Markdown rendering
rehype-highlight - Code syntax highlighting

### Project Structure
rag-project/
├── backend/                 # Python backend
│   ├── rag_test.py         # Main application
│   ├── requirements.txt    # Dependencies
│   └── .env               # Environment config
├── frontend/               # React frontend
│   ├── src/
│   │   ├── App.jsx        # Main component
│   │   ├── main.jsx       # Entry point
│   │   └── App.css        # Styles
│   ├── package.json       # Frontend dependencies
│   └── vite.config.js     # Vite configuration
└── my_chroma_db/          # Vector database storage

### Usage Examples
🤖 Technical Questions: "What is prompt engineering?"
🌐 Real-time Information: "What are the latest AI news?"
🔍 Hybrid Queries: "How to defend against LLM adversarial attacks?"
💡 Programming Help: "Explain Python decorators with examples"

### Development Roadmap
Conversation history management

Multi-format file support (PDF, Word, Excel)

User authentication system

Admin dashboard interface

Performance monitoring and optimization

Streaming response support

Multi-language support

### Acknowledgements
LangChain community for excellent tooling

Alibaba Cloud Tongyi Qianwen for language model services

Tavily for real-time search API

React and Vite communities for frontend tools

