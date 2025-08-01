# src/utils/config.py
# HackRx 6.0 - Configuration Management Module

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """
    Application configuration using Pydantic settings.
    Automatically loads from environment variables and .env file.
    """
    
    # API Configuration
    gemini_api_key: str = Field(..., env="GEMINI_API_KEY")
    groq_api_key: str = Field(..., env="GROQ_API_KEY")
    auth_token: str = Field(..., env="AUTH_TOKEN")
    
    # Application Settings
    environment: str = Field(default="development", env="ENVIRONMENT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Document Processing Settings
    max_document_size_mb: int = Field(default=50, env="MAX_DOCUMENT_SIZE_MB")
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")
    
    # Vector Search Settings
    top_k_results: int = Field(default=5, env="TOP_K_RESULTS")
    similarity_threshold: float = Field(default=0.7, env="SIMILARITY_THRESHOLD")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    
    # API Client Settings
    max_retries: int = Field(default=3, env="MAX_RETRIES")
    timeout_seconds: int = Field(default=30, env="TIMEOUT_SECONDS")
    rate_limit_buffer: float = Field(default=0.8, env="RATE_LIMIT_BUFFER")
    
    # LLM Settings
    primary_llm: str = Field(default="gemini", env="PRIMARY_LLM")
    fallback_llm: str = Field(default="groq", env="FALLBACK_LLM")
    max_tokens: int = Field(default=1000, env="MAX_TOKENS")
    temperature: float = Field(default=0.1, env="TEMPERATURE")
    
    # File Paths
    models_dir: str = Field(default="models", env="MODELS_DIR")
    data_dir: str = Field(default="data", env="DATA_DIR")
    logs_dir: str = Field(default="logs", env="LOGS_DIR")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


class LLMConfig:
    """LLM-specific configuration and rate limits"""
    
    GEMINI_LIMITS = {
        "requests_per_minute": 15,
        "tokens_per_minute": 32000,
        "context_window": 1000000,
        "max_output_tokens": 8192
    }
    
    GROQ_LIMITS = {
        "requests_per_minute": 30,  # Free tier
        "tokens_per_minute": 6000,
        "context_window": 4096,
        "max_output_tokens": 1024
    }
    
    MODELS = {
        "gemini": {
            "model_name": "gemini-1.5-flash",
            "provider": "google",
            "limits": GEMINI_LIMITS
        },
        "groq": {
            "model_name": "llama3-8b-8192",
            "provider": "groq", 
            "limits": GROQ_LIMITS
        }
    }


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings instance"""
    return settings


def validate_environment() -> bool:
    """
    Validate that all required environment variables are set
    Returns True if valid, False otherwise
    """
    try:
        settings = get_settings()
        
        # Check required API keys
        if not settings.gemini_api_key or settings.gemini_api_key == "your_gemini_key_here":
            print("❌ GEMINI_API_KEY not configured")
            return False
            
        if not settings.groq_api_key or settings.groq_api_key == "your_groq_key_here":
            print("❌ GROQ_API_KEY not configured")
            return False
            
        if not settings.auth_token:
            print("❌ AUTH_TOKEN not configured")
            return False
            
        print("✅ Environment validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Environment validation failed: {e}")
        return False


def create_directories():
    """Create necessary directories if they don't exist"""
    dirs_to_create = [
        settings.models_dir,
        settings.data_dir,
        settings.logs_dir,
        f"{settings.models_dir}/embeddings",
        f"{settings.data_dir}/sample_documents"
    ]
    
    for directory in dirs_to_create:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ Directories created/verified")