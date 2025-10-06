# 🚀 AI Financial Research Assistant

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Node.js 18+](https://img.shields.io/badge/node-18+-green.svg)](https://nodejs.org/)
[![TypeScript](https://img.shields.io/badge/typescript-%23007ACC.svg?style=flat&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![NestJS](https://img.shields.io/badge/nestjs-%23E0234E.svg?style=flat&logo=nestjs&logoColor=white)](https://nestjs.com/)
[![Next.js](https://img.shields.io/badge/Next-black?style=flat&logo=next.js&logoColor=white)](https://nextjs.org/)

A sophisticated AI-powered financial research platform featuring real-time market analysis, streaming responses, intelligent memory systems, and comprehensive source tracking. Built with modern tech stack for professional financial analysis and investment research.

## 🌟 Features

### 🎯 **Core Capabilities**
- **🧠 Intelligent AI Analysis**: GPT-4o powered financial research with reasoning traces
- **📊 Real-Time Market Data**: Live stock prices, market data, and financial metrics
- **🔄 Streaming Interface**: Real-time AI responses with thinking process visualization
- **💬 Advanced Chat System**: Multi-threaded conversations with persistent history
- **📈 Comprehensive Analysis**: Stock analysis, peer comparisons, and market insights
- **� Enhanced Source Tracking**: 25+ source types with reliability scoring
- **🔐 Enterprise Authentication**: JWT-based security with session management

### 🚀 **Advanced Features**
- **Thinking Trace Visualization**: See AI reasoning process in real-time
- **Enhanced Database Schema**: Comprehensive metadata tracking for all analyses
- **Multi-Agent Research**: LangGraph workflow with specialized financial agents
- **Source Classification**: Automatic categorization and reliability assessment
- **Memory Integration**: Redis-based conversation context and state management
- **Professional Reports**: Exportable analysis with citations and metadata
- **Responsive UI**: Material-UI components with mobile-optimized design

## 🏗️ System Architecture

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│     Frontend        │    │      Backend        │    │    AI Agents        │
│                     │    │                     │    │                     │
│  • Next.js 15.5.4   │◄──►│  • NestJS Framework │◄──►│  • FastAPI Server   │
│  • Material-UI      │    │  • JWT Auth         │    │  • LangGraph        │
│  • TypeScript       │    │  • Prisma ORM       │    │  • Redis Memory     │
│  • Streaming UI     │    │  • Enhanced Schema  │    │  • Yahoo Finance    │
│  • Thinking Traces  │    │  • Source Tracking  │    │  • Multi-Agents     │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
         │                           │                           │
         │                           │                           │
         ▼                           ▼                           ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                          Data & Infrastructure Layer                       │
│                                                                           │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐                │
│  │   SQLite DB   │  │     Redis     │  │  File System  │                │
│  │               │  │               │  │               │                │
│  │ • Users       │  │ • Sessions    │  │ • Logs        │                │
│  │ • Threads     │  │ • Memory      │  │ • Traces      │                │
│  │ • Messages    │  │ • State Mgmt  │  │ • Config      │                │
│  │ • Sources     │  │ • Streaming   │  │ • Cache       │                │
│  │ • Enhanced    │  │ • Checkpoints │  │               │                │
│  │   Metadata    │  │               │  │               │                │
│  └───────────────┘  └───────────────┘  └───────────────┘                │
└───────────────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- **Node.js 18+** (for frontend and backend)
- **Python 3.11+** (for AI agents)
- **OpenAI API Key** (required for AI analysis)
- **Git** (for cloning the repository)

### 1. Clone & Setup Environment
```bash
git clone <repository-url>
cd Project1

# Copy environment template
cp .env.example .env
```

### 2. Configure API Keys
Edit `.env` file with your credentials:
```bash
# Essential Configuration
OPENAI_API_KEY=your_openai_api_key_here
JWT_SECRET=your_secure_random_jwt_secret_32chars_minimum

# Optional Enhancements
TAVILY_API_KEY=your_tavily_web_search_key  # For web research
PINECONE_API_KEY=your_pinecone_vector_db_key  # For advanced memory
```

### 3. Install Dependencies & Start Services

#### Option A: Quick Start (All Services)
```bash
# Make setup script executable
chmod +x setup.sh

# Run setup and start all services
./setup.sh
```

#### Option B: Manual Setup
```bash
# Backend (Terminal 1)
cd backend
npm install
npx prisma generate
npx prisma db push
npm run start:dev

# AI Agents (Terminal 2)
cd agents
pip install -r requirements.txt
python main.py

# Frontend (Terminal 3)
cd frontend
npm install
npm run dev
```

### 4. Access Your Application
- **🌐 Web Interface**: http://localhost:3000
- **⚡ Backend API**: http://localhost:8000
- **🤖 AI Service**: http://localhost:9000
- **📚 API Documentation**: http://localhost:8000/api

## 💻 Development Guide

### 🔧 **Backend Development** (NestJS + Prisma + SQLite)
```bash
cd backend

# Install dependencies
npm install

# Setup database
npx prisma generate    # Generate Prisma client
npx prisma db push     # Create database schema
npx prisma studio      # Optional: View database

# Start development server
npm run start:dev      # Runs on http://localhost:8000
```

**Key Backend Features:**
- Enhanced Prisma schema with 25+ source types
- JWT authentication with session management  
- AI service integration with metadata tracking
- Source deduplication and reliability scoring
- Comprehensive message and thread management

### 🎨 **Frontend Development** (Next.js + TypeScript + Material-UI)
```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev           # Runs on http://localhost:3000
npm run build         # Production build
npm run type-check    # TypeScript validation
```

**Key Frontend Features:**
- Real-time streaming interface with thinking traces
- Multi-threaded conversation management
- Enhanced source panel with citations
- Responsive Material-UI design
- TypeScript for type safety

### 🤖 **AI Agents Development** (FastAPI + LangGraph + Redis)
```bash
cd agents

# Setup Python environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies  
pip install -r requirements.txt

# Start AI service
python main.py            # Runs on http://localhost:9000
```

**Key AI Features:**
- Multi-agent LangGraph workflow
- Real-time financial data integration
- Redis-based memory and state management
- Streaming response generation
- Enhanced source classification

## �️ Project Structure

```
Project1/
├── 📱 frontend/                 # Next.js React Application
│   ├── app/
│   │   ├── chat/page.tsx       # Main chat interface with streaming
│   │   ├── login/page.tsx      # Authentication page  
│   │   └── globals.css         # Global styles
│   ├── components/
│   │   ├── Navigation.tsx      # App navigation bar
│   │   └── ProtectedRoute.tsx  # Auth route protection
│   ├── contexts/
│   │   └── AuthContext.tsx     # Authentication state management
│   └── package.json
│
├── 🔧 backend/                  # NestJS API Server
│   ├── src/
│   │   ├── auth/               # Authentication module
│   │   ├── chat/               # Chat & messaging logic
│   │   ├── services/
│   │   │   └── ai.service.ts   # AI integration service
│   │   └── main.ts
│   ├── prisma/
│   │   └── schema.prisma       # Enhanced database schema
│   └── package.json
│
├── 🤖 agents/                   # Python AI Service
│   ├── main.py                 # FastAPI server entry point
│   ├── research_agent.py       # Core research agent
│   ├── enhanced_research_agent.py  # Advanced multi-agent workflow
│   ├── financial_data_service.py   # Yahoo Finance integration
│   ├── memory/                 # Memory management system
│   │   ├── redis_store.py      # Redis-based short-term memory
│   │   ├── vector_store.py     # Vector database integration
│   │   └── memory_pipeline.py  # Unified memory interface
│   └── requirements.txt
│
├── 📄 Configuration
│   ├── .env.example           # Environment template
│   ├── docker-compose.yml     # Container orchestration
│   ├── setup.sh              # Quick setup script
│   └── README.md             # This file
│
└── 📊 Data & Logs
    ├── backend/prisma/dev.db  # SQLite database
    └── logs/                  # Application logs
```

## 🎯 Key Components Deep Dive

### 1. **Enhanced Database Schema** (`backend/prisma/schema.prisma`)
Comprehensive data modeling for financial research:
- **Users & Authentication**: JWT-based session management
- **Threads & Messages**: Multi-conversation support with metadata
- **Enhanced Message Fields**: AI confidence, processing time, analysis type
- **Source Classification**: 25+ source types with reliability scoring
- **Financial Metadata**: Stock symbols, insights, real-time data tracking

### 2. **AI Research Service** (`agents/enhanced_research_agent.py`)
Multi-agent LangGraph workflow:
- **Financial Data Agent**: Real-time market data and stock analysis
- **Research Coordinator**: Query planning and execution orchestration  
- **Memory Integration**: Context-aware responses with conversation history
- **Streaming Handler**: Real-time response generation with thinking traces
- **Source Classification**: Automatic categorization and reliability assessment

### 3. **Chat Interface** (`frontend/app/chat/page.tsx`)
Advanced conversation interface:
- **Streaming UI**: Real-time AI response rendering
- **Thinking Traces**: Visualized AI reasoning process
- **Thread Management**: Multi-conversation history and navigation
- **Source Panel**: Enhanced citations with metadata display
- **Quick Actions**: Pre-configured financial analysis queries

### 4. **Memory System** (`agents/memory/`)
Intelligent state and context management:
- **Redis Store**: Short-term memory and streaming state
- **Vector Store**: Long-term conversation embeddings (optional)
- **Memory Pipeline**: Unified interface for all memory operations
- **Context Synthesis**: Intelligent conversation context building

## ⚙️ Configuration

### 📋 **Environment Variables** (`.env`)
```bash
# 🔑 Essential Configuration  
OPENAI_API_KEY=sk-your-openai-api-key-here
JWT_SECRET=your-secure-random-jwt-secret-32-characters-minimum

# 🤖 AI Service Configuration
OPENAI_MODEL=gpt-4o                    # AI model for analysis
OPENAI_TEMPERATURE=0.1                 # Response creativity (0.0-1.0)
MAX_TOKENS=4000                        # Maximum response length

# 🔄 Memory & State Management  
ENABLE_MEMORY=true                     # Enable conversation memory
REDIS_URL=redis://localhost:6379       # Redis connection string
ENABLE_LANGGRAPH=true                  # Enable multi-agent workflow
USE_CHECKPOINTING=true                 # Enable workflow state persistence

# 🔍 Research Enhancement Features
ENABLE_WEB_SEARCH=true                 # Enable web research (requires Tavily)
TAVILY_API_KEY=tvly-your-key-here     # Optional: Web search capability
MAX_SOURCES=15                         # Maximum sources per query
MIN_RELEVANCE_SCORE=0.7               # Source relevance threshold

# 📊 Financial Data Configuration  
YAHOO_FINANCE_ENABLED=true            # Enable Yahoo Finance integration
ALPHA_VANTAGE_API_KEY=your-key        # Optional: Enhanced market data
FINNHUB_API_KEY=your-key             # Optional: Additional financial data

# 💾 Advanced Memory (Optional)
VECTOR_BACKEND=memory                 # 'memory' or 'pinecone'
PINECONE_API_KEY=your-key            # Optional: Production vector DB
USE_OPENAI_EMBEDDINGS=false          # Use OpenAI for embeddings

# 🖥️ Frontend Configuration
NEXT_PUBLIC_ENABLE_STREAMING=true     # Enable real-time streaming UI
NEXT_PUBLIC_ENABLE_THINKING_TRACES=true  # Show AI reasoning process
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000  # Backend API URL
```

### 🎛️ **Feature Flags**
The system supports dynamic feature toggling through environment variables:

| Feature | Variable | Description |
|---------|----------|-------------|
| **Streaming UI** | `NEXT_PUBLIC_ENABLE_STREAMING` | Real-time response streaming |
| **Thinking Traces** | `NEXT_PUBLIC_ENABLE_THINKING_TRACES` | AI reasoning visualization |
| **Memory System** | `ENABLE_MEMORY` | Conversation context retention |
| **Web Research** | `ENABLE_WEB_SEARCH` | External information gathering |
| **Multi-Agent** | `ENABLE_LANGGRAPH` | Advanced research workflow |
| **Export Features** | `ENABLE_EXPORT_FEATURES` | PDF/JSON report generation |

## 📊 API Documentation

### 🔐 **Authentication Endpoints** (Backend - Port 8000)
```bash
POST   /auth/register              # Create new user account
POST   /auth/login                 # User login with email/password  
GET    /auth/me                    # Get current user profile
POST   /auth/logout                # Logout and invalidate session
GET    /auth/validate              # Validate JWT token
```

### 💬 **Chat & Messaging** (Backend - Port 8000)
```bash
# Thread Management
POST   /chat/threads               # Create new conversation thread
GET    /chat/threads               # List user's conversation threads
GET    /chat/threads/:threadId     # Get specific thread with messages
DELETE /chat/threads/:threadId     # Delete conversation thread

# Message Management  
POST   /chat/threads/:threadId/messages     # Send message to thread
GET    /chat/threads/:threadId/messages     # Get thread messages
GET    /chat/threads/:threadId/messages/:messageId  # Get specific message
```

### 🤖 **AI Research Service** (AI Agents - Port 9000)
```bash
# Research Endpoints
POST   /research                   # Standard financial analysis
POST   /research/stream            # Real-time streaming research
GET    /health                     # Service health check
GET    /memory/health              # Memory system status

# Advanced Features
POST   /research/compare           # Peer comparison analysis
POST   /research/sector            # Sector analysis
GET    /research/sources           # Available data sources
```

### 🔍 **Query Examples**

#### Basic Stock Analysis
```bash
curl -X POST http://localhost:9000/research \\
  -H "Content-Type: application/json" \\
  -d '{
    "query": "Analyze Apple stock performance",
    "analysis_depth": "comprehensive",
    "include_charts": false,
    "timeframe": "1y"
  }'
```

#### Streaming Research
```bash
curl -X POST http://localhost:9000/research/stream \\
  -H "Content-Type: application/json" \\
  -d '{
    "query": "Compare Tesla vs Ford financial metrics", 
    "session_id": "user_123_session",
    "include_thinking": true
  }'
```

## 🧪 Testing & Validation

### 🏥 **Health Checks**
Verify all services are running correctly:

```bash
# Service Health Status
curl http://localhost:3000          # Frontend (should show login page)
curl http://localhost:8000/health   # Backend API health  
curl http://localhost:9000/health   # AI Agents health

# Detailed System Check
curl http://localhost:9000/memory/health  # Memory system status
curl http://localhost:8000/api           # API documentation
```

**Expected Health Response:**
```json
{
  "status": "healthy",
  "services": {
    "ai_research": "operational",
    "streaming": "operational", 
    "financial_data": "operational"
  },
  "memory_system": {
    "status": "healthy",
    "type": "redis",
    "connection": "active"
  }
}
```

### 🧪 **Functional Testing**
Test core functionality through the UI:

1. **Authentication Flow**:
   - Visit http://localhost:3000
   - Register new account or login
   - Verify JWT token in developer tools

2. **Chat Functionality**:
   - Create new chat thread
   - Send message: "Analyze Apple stock"
   - Verify streaming response appears
   - Check thinking traces panel

3. **Advanced Features**:
   - Test peer comparison: "Compare AAPL vs MSFT"
   - Verify source panel shows citations
   - Check conversation history persistence

### 🔍 **Development Testing**
```bash
# Backend Testing
cd backend
npm run test              # Run unit tests
npm run test:e2e         # End-to-end tests
npm run lint             # Code quality check

# Frontend Testing  
cd frontend
npm run test             # React component tests
npm run build            # Production build test
npm run type-check       # TypeScript validation

# AI Service Testing
cd agents  
python -m pytest tests/        # Python unit tests
python test_research.py        # Research functionality test
python test_memory.py          # Memory system test
```

## � Troubleshooting

### 🔧 **Common Issues & Solutions**

#### 1. **Service Connection Issues**
```bash
# Problem: "Cannot connect to backend/AI service"
# Solution: Check if all services are running

# Check service status
curl http://localhost:8000/health   # Backend
curl http://localhost:9000/health   # AI Service

# Restart services if needed
cd backend && npm run start:dev
cd agents && python main.py
```

#### 2. **OpenAI API Errors**  
```bash
# Problem: "OpenAI API key invalid" or rate limits
# Solution: Verify API key and check usage

export OPENAI_API_KEY="your-valid-api-key"
curl -H "Authorization: Bearer $OPENAI_API_KEY" \\
     https://api.openai.com/v1/models
```

#### 3. **Database Issues**
```bash  
# Problem: Database connection or schema issues
# Solution: Reset and regenerate database

cd backend
rm prisma/dev.db           # Remove existing database
npx prisma generate        # Regenerate client
npx prisma db push         # Create fresh schema
```

#### 4. **Memory/Redis Connection**
```bash
# Problem: "Redis connection failed" 
# Solution: Check Redis service

# Install and start Redis (macOS)
brew install redis
brew services start redis

# Test Redis connection
redis-cli ping            # Should return "PONG"
```

#### 5. **Frontend Build Issues**
```bash
# Problem: TypeScript errors or build failures
# Solution: Clear cache and reinstall

cd frontend
rm -rf node_modules .next
npm install
npm run build
```

### � **Performance Monitoring**

Monitor system performance and resource usage:

```bash  
# Check service resource usage
ps aux | grep -E "(node|python|redis)"

# Monitor API response times
curl -w "@curl-format.txt" http://localhost:8000/health

# Check memory usage
curl http://localhost:9000/memory/health | jq '.memory_stats'
```

## 🎯 **Current Status & Features**

### ✅ **Production Ready Features**
- [x] **Core AI Analysis**: GPT-4o powered financial research
- [x] **Real-time Streaming**: Live AI responses with thinking traces
- [x] **Enhanced Database**: Comprehensive metadata and source tracking
- [x] **Memory System**: Redis-based conversation context
- [x] **Multi-threaded Chat**: Persistent conversation history
- [x] **Authentication**: JWT-based user management
- [x] **Source Classification**: 25+ categorized financial data sources
- [x] **Responsive UI**: Material-UI with mobile optimization

### 🔧 **Advanced Features**  
- [x] **LangGraph Workflow**: Multi-agent research coordination
- [x] **Yahoo Finance Integration**: Real-time market data
- [x] **Source Deduplication**: Intelligent caching system
- [x] **Thinking Trace UI**: Visual AI reasoning process
- [x] **Enhanced Metadata**: Processing time, confidence scores
- [x] **Error Handling**: Comprehensive fallback systems

### 🎯 **Development Roadmap**

#### **Immediate Priorities** (90%+ Complete)
- [ ] **Enhanced Field Population**: Fix database enhanced fields saving
- [ ] **Source Panel UI**: Complete source citation display
- [ ] **Export Features**: PDF report generation
- [ ] **Performance Optimization**: Response caching and speed improvements

#### **Next Phase** (Future Enhancements)
- [ ] **Advanced Analytics**: Usage statistics and insights
- [ ] **Mobile App**: React Native mobile application  
- [ ] **Real-time Notifications**: WebSocket-based alerts
- [ ] **Additional Data Sources**: Bloomberg, Reuters integration
- [ ] **Machine Learning**: Personalized recommendations

## 📋 **Quick Reference**

### **Development Commands**
```bash
# Start all services for development
./setup.sh                         # All-in-one setup

# Individual service commands  
cd backend && npm run start:dev    # Backend (Port 8000)
cd frontend && npm run dev         # Frontend (Port 3000)  
cd agents && python main.py       # AI Service (Port 9000)

# Database operations
cd backend
npx prisma studio                  # Database viewer
npx prisma generate               # Regenerate client
npx prisma db push                # Update schema
```

### **Useful URLs**
- **🌐 Main App**: http://localhost:3000
- **⚡ Backend API**: http://localhost:8000
- **🤖 AI Service**: http://localhost:9000  
- **📊 Database Studio**: http://localhost:5555 (after `npx prisma studio`)
- **📖 API Docs**: http://localhost:8000/api

### **Environment Quick Setup**
```bash
# Minimal .env for getting started
OPENAI_API_KEY=your_openai_key_here
JWT_SECRET=your_32_character_secret_here
ENABLE_MEMORY=true
ENABLE_LANGGRAPH=true
```


**� Built for Professional Financial Research** | **⚡ Powered by Modern AI Technology** | **💼 Production Ready**
