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
