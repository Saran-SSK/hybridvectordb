# HybridVectorDB Production Docker Image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HYBRIDVECTORDB_VERSION=0.5.0
ENV HYBRIDVECTORDB_PORT=8080

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    python3-dev \
    libopenblas-dev \
    liblapack-dev \
    libfaiss-dev \
    libfaiss-gpu-dev \
    cuda-toolkit-11-8 \
    nvidia-driver-470

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional production dependencies
RUN pip install --no-cache-dir \
    gunicorn==21.2.0 \
    flask==2.3.2 \
    prometheus-client==0.16.0 \
    psutil==5.9.5 \
    pydantic==2.0.3 \
    uvicorn==0.21.0 \
    fastapi==0.95.0 \
    python-multipart==0.0.6

# Create application directory
WORKDIR /app

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash hybridvectordb
USER hybridvectordb

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "4", "--timeout", "120", "hybridvectordb.app:app"]
