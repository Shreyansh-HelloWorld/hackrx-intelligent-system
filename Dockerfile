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
# First install faiss-cpu and then the rest of the requirements
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir faiss-cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY .env.example .env

RUN mkdir -p logs models data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
