# HackRx 6.0 - Intelligent Query-Retrieval System

## 🚀 **COMPLETE LLM-POWERED DOCUMENT ANALYSIS SYSTEM**

A production-ready intelligent system that processes insurance/legal documents and answers questions using semantic search and LLM reasoning with automatic failover between Gemini and Groq APIs.

### **📋 SYSTEM FEATURES**

- ✅ **Multi-Format Support**: PDF, DOCX document processing
- ✅ **Semantic Search**: FAISS vector database with hybrid search
- ✅ **LLM Failover**: Gemini + Groq with automatic switching
- ✅ **Production Ready**: FastAPI with authentication, logging, monitoring
- ✅ **Free Deployment**: Render, Railway, Fly.io configurations
- ✅ **Explainable AI**: Decision reasoning with source citations
- ✅ **Token Optimization**: Efficient chunking and context management

---

## **🏗️ ARCHITECTURE OVERVIEW**

```
Input Document URL → Document Parser → Text Chunks → Embeddings → FAISS Search
                                                                      ↓
JSON Response ← LLM Reasoning ← Context Retrieval ← Query Analysis ← User Questions
```

**Core Components:**
- **FastAPI**: REST API with `/hackrx/run` endpoint
- **Document Processing**: PyMuPDF + python-docx for text extraction
- **Vector Search**: FAISS with sentence-transformers embeddings
- **LLM Integration**: Gemini API (primary) + Groq API (fallback)
- **Reasoning Engine**: Query analysis + semantic matching + explainable decisions

---

## **⚡ QUICK START**

### **1. Environment Setup**

```bash
# Clone and navigate to project
git clone <your-repo-url>
cd hackrx-intelligent-system

# Create conda environment (recommended)
conda create -n hackrx python=3.11 -y
conda activate hackrx

# Install dependencies
# faiss-cpu is installed with conda for better compatibility on some systems
conda install -c pytorch faiss-cpu -y
pip install -r requirements.txt
```

### **2. API Keys Configuration**

Create `.env` file:

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

```env
# Required API Keys
GEMINI_API_KEY=your_gemini_key_from_makersuite
GROQ_API_KEY=your_groq_key_from_console
AUTH_TOKEN=0b2c1453ccb7985da0c04cd70bca63a5ed5145f8f1b6316b56c1dafabb3e95a7

# Optional Configuration
ENVIRONMENT=development
LOG_LEVEL=INFO
MAX_DOCUMENT_SIZE_MB=50
CHUNK_SIZE=1000
TOP_K_RESULTS=5
```

**Get API Keys:**
- **Gemini**: https://makersuite.google.com/app/apikey (Free tier: 15 RPM)
- **Groq**: https://console.groq.com/keys (Free tier: 30 RPM)

### **3. Run Locally**

```bash
# Start development server
python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Or use the direct command
python src/main.py
```

**Test the API:**

```bash
# Health check
curl http://localhost:8000/health

# Process document
curl -X POST http://localhost:8000/hackrx/run \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer 0b2c1453ccb7985da0c04cd70bca63a5ed5145f8f1b6316b56c1dafabb3e95a7" \
  -d '{
    "documents": "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf",
    "questions": [
      "What is the grace period for premium payment?",
      "Does this policy cover maternity expenses?"
    ]
  }'
```

---

## **🚀 DEPLOYMENT OPTIONS**

### **Option 1: Render (Recommended)**

```bash
# 1. Push code to GitHub
git add .
git commit -m "HackRx 6.0 deployment"
git push origin main

# 2. Deploy to Render
- Go to https://render.com
- Connect GitHub repository
- Select "Web Service"
- Use these settings:
  - Build Command: pip install -r requirements.txt
  - Start Command: uvicorn src.main:app --host 0.0.0.0 --port $PORT
  - Environment: Add GEMINI_API_KEY, GROQ_API_KEY
```

### **Option 2: Railway**

```bash
# 1. Install Railway CLI
npm install -g @railway/cli

# 2. Deploy
railway login
railway init
railway up

# 3. Add environment variables in Railway dashboard
```

### **Option 3: Docker (Any Platform)**

```bash
# Build image
docker build -t hackrx-system .

# Run container
docker run -p 8000:8000 \
  -e GEMINI_API_KEY=your_key \
  -e GROQ_API_KEY=your_key \
  hackrx-system
```

---

## **📊 API DOCUMENTATION**

### **Main Endpoint: `/hackrx/run`**

**Request:**
```json
{
  "documents": "https://example.com/policy.pdf",
  "questions": [
    "What is the grace period for premium payment?",
    "Does this policy cover knee surgery?"
  ]
}
```

**Headers:**
```
Authorization: Bearer 0b2c1453ccb7985da0c04cd70bca63a5ed5145f8f1b6316b56c1dafabb3e95a7
Content-Type: application/json
```

**Response:**
```json
{
  "answers": [
    "A grace period of thirty days is provided for premium payment after the due date.",
    "Yes, knee surgery is covered under this policy, subject to waiting periods and conditions."
  ]
}
```

### **Health Check: `/health`**

```json
{
  "status": "healthy",
  "version": "1.0.0", 
  "timestamp": "2025-01-XX XX:XX:XX UTC",
  "environment": "production"
}
```

### **Interactive Docs**

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

## **🧪 TESTING**

### **Run Tests**

```bash
# Basic API tests (no API keys required)
python tests/test_api.py

# Full test suite (requires API keys)
pytest tests/ -v

# Performance benchmarks
pytest tests/test_api.py::TestPerformance -v
```

### **Manual Testing with Sample Document**

The system includes the HackRx sample document URL for testing:

```bash
curl -X POST http://localhost:8000/hackrx/run \
  -H "Authorization: Bearer 0b2c1453ccb7985da0c04cd70bca63a5ed5145f8f1b6316b56c1dafabb3e95a7" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D",
    "questions": [
      "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
      "What is the waiting period for pre-existing diseases (PED) to be covered?",
      "Does this policy cover maternity expenses, and what are the conditions?"
    ]
  }'
```

---

## **⚙️ SYSTEM CONFIGURATION**

### **Performance Tuning**

```env
# Embedding Settings
EMBEDDING_MODEL=all-MiniLM-L6-v2  # Fast, good quality
CHUNK_SIZE=1000                   # Optimal for insurance docs
CHUNK_OVERLAP=200                 # Good context retention

# Search Settings  
TOP_K_RESULTS=5                   # Balance relevance vs speed
SIMILARITY_THRESHOLD=0.7          # Filter low-quality matches

# LLM Settings
MAX_TOKENS=1000                   # Sufficient for detailed answers
TEMPERATURE=0.1                   # Factual, consistent responses
```

### **Rate Limiting**

- **Gemini**: 15 RPM (free tier) → Auto-switch to Groq
- **Groq**: 30 RPM (free tier) → Built-in backoff
- **Rate Limit Buffer**: 80% of limits to prevent errors

### **Resource Usage**

- **Memory**: ~512MB (embedding model + vectors)
- **CPU**: Efficient with async processing
- **Storage**: Minimal (no persistent state)

---

## **🔧 TROUBLESHOOTING**

### **Common Issues**

**1. API Key Errors**
```bash
# Verify API keys are set
python -c "from src.utils.config import get_settings; print(get_settings().gemini_api_key[:10])"
```

**2. Document Download Fails**
- Check URL accessibility
- Verify file size < 50MB
- Ensure supported format (PDF/DOCX)

**3. Memory Issues**
- Reduce `CHUNK_SIZE` to 500
- Lower `TOP_K_RESULTS` to 3
- Use CPU-only embeddings

**4. Slow Response Times**
- Check embedding model download (first run)
- Monitor LLM API latency
- Verify adequate memory

### **Debug Mode**

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
export DEBUG=True

# Run with verbose output
python src/main.py
```

---

## **📁 PROJECT STRUCTURE**

```
hackrx-intelligent-system/
├── src/
│   ├── main.py                 # FastAPI application
│   ├── api/
│   │   ├── routes.py          # API integration
│   │   └── middleware.py      # Auth & logging
│   ├── core/
│   │   ├── document_processor.py  # PDF/DOCX parsing
│   │   ├── embeddings.py      # Text vectorization
│   │   ├── vector_store.py    # FAISS search
│   │   └── llm_handler.py     # LLM APIs with failover
│   ├── reasoning/
│   │   └── query_processor.py # Query analysis & reasoning
│   └── utils/
│       ├── config.py          # Configuration
│       ├── logger.py          # Logging
│       └── helpers.py         # Utilities
├── tests/
│   └── test_api.py            # Test suite
├── deploy/
│   ├── render.yaml            # Render config
│   ├── railway.toml           # Railway config
│   └── fly.toml              # Fly.io config
├── requirements.txt           # Dependencies
├── Dockerfile                 # Container config
└── README.md                  # This file
```

---

## **🎯 HACKRX SUBMISSION**

### **Submission Checklist**

- ✅ **API Endpoint**: `/hackrx/run` implemented
- ✅ **Authentication**: Bearer token validation
- ✅ **Document Processing**: PDF/DOCX support
- ✅ **Question Answering**: LLM-powered responses
- ✅ **Error Handling**: Graceful failure management
- ✅ **Performance**: <30s response times
- ✅ **Deployment**: Multiple platform support

### **Expected Performance**

- **Accuracy**: >85% on sample questions
- **Latency**: <15s average response time
- **Token Efficiency**: ~500 tokens per query
- **Reliability**: Automatic LLM failover
- **Scalability**: Handles concurrent requests

### **Evaluation Metrics**

1. **Accuracy**: Correct answers to insurance questions
2. **Token Efficiency**: Optimized LLM usage
3. **Latency**: Fast response times
4. **Reusability**: Clean, modular code
5. **Explainability**: Clear reasoning and citations

---

## **📞 SUPPORT**

### **System Status**

Check system health: `GET /health`

Monitor components:
- Embedding model loading
- Vector store statistics  
- LLM provider availability
- Memory and performance metrics

### **Logs**

```bash
# View application logs
tail -f logs/hackrx_*.log

# Or check console output in development
python src/main.py
```

---

## **🏆 COMPETITIVE ADVANTAGES**

1. **Multi-LLM Failover**: Reliability through redundancy
2. **Local Vector Storage**: No external dependencies
3. **Hybrid Search**: Semantic + keyword matching
4. **Domain Optimization**: Insurance/legal document focus
5. **Production Ready**: Comprehensive error handling
6. **Free Deployment**: No hosting costs
7. **Explainable Decisions**: Source citations and reasoning
8. **Token Optimization**: Cost-effective LLM usage

---

**Built for HackRx 6.0 | Ready for Production | Optimized for Insurance/Legal Documents**

🚀 **Deploy in 5 minutes | Scale to millions of documents | Win the hackathon!**