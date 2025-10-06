
"""
Startup script for the AI Agents Service
Runs the service with proper Python module path resolution
"""

import sys
import os
from pathlib import Path


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


if __name__ == "__main__":
   
    os.chdir(Path(__file__).parent / "agents")
  
    from agents.main import app
    import uvicorn
    
    print("🚀 Starting Deep Finance Research AI Agents Service...")
    print("📍 Running on: http://0.0.0.0:9000")
    print("📚 API Docs: http://localhost:9000/docs")
    print("🔍 Health Check: http://localhost:9000/health")
    
    uvicorn.run(
        "agents.main:app",
        host="0.0.0.0", 
        port=9000,
        reload=True,
        log_level="info"
    )