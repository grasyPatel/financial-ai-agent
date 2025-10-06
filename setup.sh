

echo "🚀 Setting up Deep Finance Research Chatbot..."
echo "================================================"

echo "🔍 Checking prerequisites..."


if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found. Please install Node.js 18+ from https://nodejs.org"
    exit 1
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python not found. Please install Python 3.9+ from https://python.org"
    exit 1
fi

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker from https://docker.com"
    exit 1
fi

echo "✅ All prerequisites found!"

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your API keys before running the application"
else
    echo "✅ .env file already exists"
fi

echo ""
echo "🏗️  Project Structure Created:"
echo "├── frontend/          (Next.js React app)"
echo "├── backend/           (NestJS API server)"  
echo "├── agents/            (Python AI service)"
echo "├── docs/              (Documentation)"
echo "└── docker-compose.yml (Run all services)"

echo ""
echo "📚 What you've built in Phase 1:"
echo "✅ Complete project structure"
echo "✅ Frontend: Basic chat interface"
echo "✅ Backend: NestJS API with health endpoints"
echo "✅ Agents: Python FastAPI service for AI"
echo "✅ Database: Prisma schema design"
echo "✅ Docker: Multi-service setup"

echo ""
echo "🎯 Next Steps:"
echo "1. Edit .env file with your API keys"
echo "2. Run: docker-compose up --build"
echo "3. Visit: http://localhost:3000"
echo "4. Check backend: http://localhost:8000/health"
echo "5. Check agents: http://localhost:8001/health"

echo ""
echo "📖 Learning Path:"
echo "📍 Current: Phase 1 - Foundation Complete ✅"
echo "🔄 Next: Phase 2 - Database & Authentication"
echo ""
echo "Ready to continue? Read docs/phase-1-guide.md for details!"