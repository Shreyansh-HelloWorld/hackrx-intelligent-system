# Dockerfile for HackRx HuggingFace Spaces Deployment
# Optimized for ML workloads with 1GB memory

FROM python:3.11-slim

# Create user for HuggingFace Spaces (required)
RUN useradd -m -u 1000 user
USER user

# Set environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/home/user/app \
    PYTHONUNBUFFERED=1 \
    ENVIRONMENT=production

# Set working directory
WORKDIR $HOME/app

# Copy requirements first for better caching
COPY --chown=user requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install faiss-cpu separately for better compatibility
RUN pip install --no-cache-dir faiss-cpu

# Copy application code
COPY --chown=user src/ ./src/
# Create basic .env file (real environment variables set via HF Spaces)
RUN echo "ENVIRONMENT=production" > .env

# Create necessary directories
RUN mkdir -p logs models data

# Expose HuggingFace Spaces standard port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start the FastAPI application on port 7860
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "7860"]