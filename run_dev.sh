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
