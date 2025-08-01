#!/bin/bash
# setup.sh - HackRx 6.0 Complete System Setup Script

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ASCII Art Header
echo -e "${CYAN}"
cat << "EOF"
  ██   ██  █████   ██████ ██   ██ ██████  ██   ██     █████        ████  
  ██   ██ ██   ██ ██      ██  ██  ██   ██  ██ ██     ██           ██  ██ 
  ███████ ███████ ██      █████   ██████    ███      ██████      ██    ██
  ██   ██ ██   ██ ██      ██  ██  ██   ██  ██ ██     ██    ██    ██  ██ 
  ██   ██ ██   ██  ██████ ██   ██ ██   ██ ██   ██     ██████  ██  ████  
                                                                         
        Intelligent Query-Retrieval System Setup
EOF
echo -e "${NC}"

echo -e "${PURPLE}================================================================${NC}"
echo -e "${YELLOW}🚀 Setting up HackRx 6.0 Intelligent Query-Retrieval System${NC}"
echo -e "${PURPLE}================================================================${NC}"
echo ""

# Function to print status
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on macOS (for conda setup)
if [[ "$OSTYPE" == "darwin"* ]]; then
    IS_MACOS=true
else
    IS_MACOS=false
fi

# Step 1: Check Prerequisites
print_status "Checking prerequisites..."

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | grep -o '[0-9]\+\.[0-9]\+')
    print_success "Python $PYTHON_VERSION found"
else
    print_error "Python 3 not found. Please install Python 3.11+"
    exit 1
fi

# Check pip
if command -v pip3 &> /dev/null; then
    print_success "pip3 found"
else
    print_error "pip3 not found. Please install pip"
    exit 1
fi

# Check conda (optional)
if command -v conda &> /dev/null; then
    print_success "Conda found"
    CONDA_AVAILABLE=true
else
    print_warning "Conda not found. Will use venv instead"
    CONDA_AVAILABLE=false
fi

echo ""

# Step 2: Create Project Directory
print_status "Setting up project structure..."

mkdir -p hackrx-intelligent-system
cd hackrx-intelligent-system

# Create directory structure
mkdir -p src/{api,core,reasoning,utils}
mkdir -p tests
mkdir -p data/sample_documents
mkdir -p models/embeddings
mkdir -p deploy
mkdir -p logs

# Create __init__.py files
touch src/__init__.py
touch src/api/__init__.py
touch src/core/__init__.py
touch src/reasoning/__init__.py
touch src/utils/__init__.py
touch tests/__init__.py

print_success "Project structure created"

# Step 3: Create Virtual Environment
print_status "Setting up Python environment..."

if [ "$CONDA_AVAILABLE" = true ]; then
    print_status "Creating and activating conda environment..."
    
    # Deactivate any existing environment
    conda deactivate 2>/dev/null || true
    
    # Create new environment
    conda create -n hackrx python=3.11 -y
    
    # Source conda's shell functions and activate the environment
    # This is the critical fix for the script's logic
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate hackrx
    
    print_success "Conda environment 'hackrx' created and activated"
else
    print_status "Creating virtual environment..."
    python3 -m venv hackrx-env
    
    # Activate virtual environment
    source hackrx-env/bin/activate
    print_success "Virtual environment created and activated"
fi

echo ""

# Step 4: Create Requirements File
print_status "Creating requirements.txt..."

cat > requirements.txt << 'EOF'
# Core FastAPI Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
python-multipart==0.0.6

# Document Processing
PyMuPDF==1.23.8
python-docx==1.1.0
requests==2.31.0

# ML/AI Stack
sentence-transformers==2.2.2
#faiss-cpu==1.7.4
# Updated LangChain packages for compatibility
langchain==0.1.20
langchain-community==0.0.38
langchain-core==0.1.52

# LLM APIs
google-generativeai==0.3.2
groq==0.4.1

# Utilities
python-dotenv==1.0.0
httpx==0.25.2
pydantic-settings==2.1.0
aiofiles==23.2.0

# Development & Testing
pytest==7.4.3
pytest-asyncio==0.21.1
EOF
print_success "requirements.txt created"

# Step 5: Install Dependencies
print_status "Installing Python dependencies..."
print_warning "This may take a few minutes..."

# First, install faiss-cpu using conda if available
if [ "$CONDA_AVAILABLE" = true ]; then
    print_status "Installing faiss-cpu with conda..."
    conda install -c pytorch faiss-cpu -y
    print_success "faiss-cpu installed successfully"
fi

pip install --upgrade pip
pip install -r requirements.txt

print_success "All other dependencies installed successfully"

echo ""

# Step 6: Create Environment Configuration
print_status "Creating environment configuration..."

cat > .env.example << 'EOF'
# HackRx 6.0 Environment Configuration
GEMINI_API_KEY=your_gemini_key_here
GROQ_API_KEY=your_groq_key_here
AUTH_TOKEN=0b2c1453ccb7985da0c04cd70bca63a5ed5145f8f1b6316b56c1dafabb3e95a7
ENVIRONMENT=development
LOG_LEVEL=INFO
MAX_DOCUMENT_SIZE_MB=50
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RESULTS=5
MAX_RETRIES=3
TIMEOUT_SECONDS=30
EOF

# Copy to actual .env file
cp .env.example .env

print_success "Environment configuration created"

# Step 7: Create Docker Configuration
print_status "Creating Docker configuration..."

cat > Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=production

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY .env.example .env

RUN mkdir -p logs models data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

print_success "Dockerfile created"

# Step 8: Create Git Configuration
print_status "Setting up Git repository..."

cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Environment
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/
hackrx-env/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Logs
logs/
*.log

# Models and Data
models/
data/
!data/sample_documents/.gitkeep

# OS
.DS_Store
Thumbs.db

# HackRx Specific
*.faiss
*.pkl
*.cache
EOF

# Initialize git if not already done
if [ ! -d ".git" ]; then
    git init
    print_success "Git repository initialized"
else
    print_success "Using existing Git repository"
fi

echo ""

# Step 9: Create Quick Start Script
print_status "Creating quick start scripts..."

cat > run_dev.sh << 'EOF'
#!/bin/bash
# Quick start script for development

echo "🚀 Starting HackRx 6.0 Development Server..."

# Check if environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "hackrx" ]] && [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Please activate your environment first:"
    echo "   conda activate hackrx  (if using conda)"
    echo "   source hackrx-env/bin/activate  (if using venv)"
    exit 1
fi

# Check if API keys are configured
if grep -q "your_gemini_key_here" .env; then
    echo "⚠️  Please configure your API keys in .env file"
    echo "   GEMINI_API_KEY - from https://makersuite.google.com/app/apikey"
    echo "   GROQ_API_KEY - from https://console.groq.com/keys"
    exit 1
fi

echo "✅ Starting server on http://localhost:8000"
echo "📚 API docs available at http://localhost:8000/docs"
echo "🏥 Health check at http://localhost:8000/health"
echo ""

python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
EOF

chmod +x run_dev.sh

cat > test_api.sh << 'EOF'
#!/bin/bash
# Quick API test script

echo "🧪 Testing HackRx 6.0 API..."

# Test health endpoint
echo "Testing health endpoint..."
curl -s http://localhost:8000/health | jq . || echo "Health check failed"

echo ""
echo "Testing main endpoint with sample data..."

# Test main endpoint
curl -X POST http://localhost:8000/hackrx/run \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer 0b2c1453ccb7985da0c04cd70bca63a5ed5145f8f1b6316b56c1dafabb3e95a7" \
  -d '{
    "documents": "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D",
    "questions": [
      "What is the grace period for premium payment?"
    ]
  }' | jq . || echo "API test failed"
EOF

chmod +x test_api.sh

print_success "Quick start scripts created"

echo ""

# Step 10: Final Setup Summary
print_status "Setup completed successfully! 🎉"

echo ""
echo -e "${PURPLE}================================================================${NC}"
echo -e "${GREEN}✅ HackRx 6.0 System Setup Complete!${NC}"
echo -e "${PURPLE}================================================================${NC}"
echo ""

echo -e "${CYAN}📋 NEXT STEPS:${NC}"
echo ""

if [ "$CONDA_AVAILABLE" = true ]; then
    echo -e "${YELLOW}1. Activate environment:${NC}"
    echo -e "   ${BLUE}conda activate hackrx${NC}"
else
    echo -e "${YELLOW}1. Activate environment:${NC}"
    echo -e "   ${BLUE}source hackrx-env/bin/activate${NC}"
fi

echo ""
echo -e "${YELLOW}2. Configure API keys:${NC}"
echo -e "   ${BLUE}nano .env${NC}  # Edit GEMINI_API_KEY and GROQ_API_KEY"
echo ""
echo -e "   Get keys from:"
echo -e "   • Gemini: ${BLUE}https://makersuite.google.com/app/apikey${NC}"
echo -e "   • Groq: ${BLUE}https://console.groq.com/keys${NC}"

echo ""
echo -e "${YELLOW}3. Start development server:${NC}"
echo -e "   ${BLUE}./run_dev.sh${NC}  # or: python -m uvicorn src.main:app --reload"

echo ""
echo -e "${YELLOW}4. Test the API:${NC}"
echo -e "   ${BLUE}./test_api.sh${NC}  # (in another terminal)"

echo ""
echo -e "${CYAN}📚 RESOURCES:${NC}"
echo -e "   • API Docs: ${BLUE}http://localhost:8000/docs${NC}"
echo -e "   • Health Check: ${BLUE}http://localhost:8000/health${NC}"
echo -e "   • Logs: ${BLUE}tail -f logs/hackrx_*.log${NC}"

echo ""
echo -e "${CYAN}🚀 DEPLOYMENT:${NC}"
echo -e "   • Render: Push to GitHub, connect to Render"
echo -e "   • Docker: ${BLUE}docker build -t hackrx-system .${NC}"
echo -e "   • Railway: ${BLUE}railway init && railway up${NC}"

echo ""
echo -e "${GREEN}🏆 Ready to win HackRx 6.0! Good luck! 🏆${NC}"
echo ""

# Optional: Auto-open documentation
if command -v open &> /dev/null && [ "$IS_MACOS" = true ]; then
    read -p "Open API documentation in browser? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Opening documentation..."
        # Start server in background for 30 seconds
        python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 &
        SERVER_PID=$!
        sleep 3
        open http://localhost:8000/docs
        sleep 27
        kill $SERVER_PID 2>/dev/null || true
    fi
fi

print_success "Setup script completed successfully! 🎉"