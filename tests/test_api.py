# tests/test_api.py
# HackRx 6.0 - API Testing Suite

import pytest
import asyncio
from httpx import AsyncClient
from fastapi.testclient import TestClient

from src.main import app
from src.utils.config import get_settings

# Test client
client = TestClient(app)

# Sample test data
SAMPLE_DOCUMENT_URL = "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D"

SAMPLE_QUESTIONS = [
    "What is the grace period for premium payment?",
    "What is the waiting period for pre-existing diseases?", 
    "Does this policy cover maternity expenses?",
    "What is the waiting period for cataract surgery?",
    "Are AYUSH treatments covered?"
]

AUTH_HEADERS = {
    "Authorization": "Bearer 0b2c1453ccb7985da0c04cd70bca63a5ed5145f8f1b6316b56c1dafabb3e95a7"
}


class TestHealthEndpoint:
    """Test health check endpoint"""
    
    def test_health_check(self):
        """Test health endpoint returns 200"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data


class TestRootEndpoint:
    """Test root endpoint"""
    
    def test_root_endpoint(self):
        """Test root endpoint returns system info"""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "endpoints" in data
        assert data["status"] == "operational"


class TestAuthenticationMiddleware:
    """Test authentication middleware"""
    
    def test_hackrx_endpoint_requires_auth(self):
        """Test that /hackrx/run requires authentication"""
        response = client.post("/hackrx/run", json={
            "documents": SAMPLE_DOCUMENT_URL,
            "questions": ["Test question"]
        })
        assert response.status_code == 401
        
    def test_invalid_auth_token(self):
        """Test invalid auth token is rejected"""
        response = client.post("/hackrx/run", 
            json={
                "documents": SAMPLE_DOCUMENT_URL,
                "questions": ["Test question"]
            },
            headers={"Authorization": "Bearer invalid_token"}
        )
        assert response.status_code == 401
    
    def test_missing_bearer_prefix(self):
        """Test missing Bearer prefix is rejected"""
        response = client.post("/hackrx/run",
            json={
                "documents": SAMPLE_DOCUMENT_URL, 
                "questions": ["Test question"]
            },
            headers={"Authorization": "invalid_format"}
        )
        assert response.status_code == 401


class TestHackRXEndpoint:
    """Test main HackRX endpoint"""
    
    def test_valid_request_structure(self):
        """Test that endpoint accepts valid request structure"""
        # This test checks request validation without full processing
        response = client.post("/hackrx/run",
            json={
                "documents": "https://example.com/test.pdf",
                "questions": ["Test question"]
            },
            headers=AUTH_HEADERS
        )
        
        # Should not be 400 (bad request) or 401 (unauthorized)
        assert response.status_code != 400
        assert response.status_code != 401
    
    def test_missing_documents_field(self):
        """Test missing documents field returns 400"""
        response = client.post("/hackrx/run",
            json={"questions": ["Test question"]},
            headers=AUTH_HEADERS
        )
        assert response.status_code == 422  # Validation error
    
    def test_missing_questions_field(self):
        """Test missing questions field returns 400"""
        response = client.post("/hackrx/run",
            json={"documents": SAMPLE_DOCUMENT_URL},
            headers=AUTH_HEADERS
        )
        assert response.status_code == 422  # Validation error
    
    def test_empty_questions_list(self):
        """Test empty questions list returns 400"""
        response = client.post("/hackrx/run",
            json={
                "documents": SAMPLE_DOCUMENT_URL,
                "questions": []
            },
            headers=AUTH_HEADERS
        )
        assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
class TestIntegrationFlow:
    """Integration tests for full workflow"""
    
    async def test_sample_document_processing(self):
        """Test processing of sample document (requires API keys)"""
        # Skip if no API keys configured
        settings = get_settings()
        if (not settings.gemini_api_key or 
            settings.gemini_api_key == "your_gemini_key_here"):
            pytest.skip("API keys not configured")
        
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post("/hackrx/run",
                json={
                    "documents": SAMPLE_DOCUMENT_URL,
                    "questions": SAMPLE_QUESTIONS[:2]  # Test with 2 questions
                },
                headers=AUTH_HEADERS,
                timeout=60.0  # Longer timeout for processing
            )
            
            assert response.status_code == 200
            
            data = response.json()
            assert "answers" in data
            assert len(data["answers"]) == 2
            
            # Verify answers are not empty
            for answer in data["answers"]:
                assert isinstance(answer, str)
                assert len(answer.strip()) > 0


# Performance benchmarks
class TestPerformance:
    """Performance benchmarking tests"""
    
    def test_health_endpoint_performance(self):
        """Test health endpoint response time"""
        import time
        
        start_time = time.time()
        response = client.get("/health")
        end_time = time.time()
        
        assert response.status_code == 200
        assert (end_time - start_time) < 1.0  # Should respond in under 1 second


# Utility functions for testing
def create_test_request(questions=None):
    """Create a test request payload"""
    if questions is None:
        questions = ["What is covered under this policy?"]
    
    return {
        "documents": SAMPLE_DOCUMENT_URL,
        "questions": questions
    }


def run_basic_tests():
    """Run basic tests that don't require API keys"""
    print("🧪 Running basic API tests...")
    
    # Test health endpoint
    response = client.get("/health")
    print(f"Health check: {response.status_code} - {'✅' if response.status_code == 200 else '❌'}")
    
    # Test authentication
    response = client.post("/hackrx/run", json=create_test_request())
    print(f"Auth required: {response.status_code} - {'✅' if response.status_code == 401 else '❌'}")
    
    # Test valid auth
    response = client.post("/hackrx/run", 
        json=create_test_request(),
        headers=AUTH_HEADERS
    )
    print(f"Valid request structure: {response.status_code} - {'✅' if response.status_code != 401 else '❌'}")
    
    print("✅ Basic tests completed")


if __name__ == "__main__":
    run_basic_tests()