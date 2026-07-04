# Stage 1: Builder stage
# WHY python:3.10-slim: Uses a lightweight Debian-based Linux image that drastically reduces container image size compared to full python:3.10.
# WHY multi-stage: Compiles and installs Python dependencies into a temporary directory (/install).
# Heavy build utilities and wheel caches remain in Stage 1 and are discarded in the final image.
FROM python:3.10-slim AS builder

# Set the working directory for builder operations
WORKDIR /build

# Copy production requirements file first to take advantage of Docker layer caching
COPY requirements.txt .

# Upgrade pip to ensure latest wheel download verification algorithms are used
RUN pip install --no-cache-dir --upgrade pip

# Pre-install CPU-only PyTorch wheel
# WHY: Reduces PyTorch download size from 530MB (CUDA) to 150MB (CPU-only), dramatically accelerating Docker build time and keeping container footprint compact.
RUN pip install --prefix=/install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install production dependencies into /install prefix using CPU wheel index
# WHY --extra-index-url: Ensures sentence-transformers resolves CPU PyTorch dependencies instead of pulling 500MB CUDA PyTorch wheels.
RUN pip install --prefix=/install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt


# Stage 2: Runtime stage
# WHY python:3.10-slim: Minimal production base image containing only Python 3.10 runtime binaries.
FROM python:3.10-slim AS runtime

# Set environment variable to ensure Python outputs (print/logs) are sent directly to stdout/stderr without buffering
# WHY PYTHONUNBUFFERED=1: Essential for container log aggregators (e.g. Render/Docker logs) to receive real-time output.
ENV PYTHONUNBUFFERED=1

# Copy installed Python packages from the builder stage
# WHY: Eliminates compiler tools and build overhead, keeping final image sleek and secure.
COPY --from=builder /install /usr/local

# Set working directory inside container for FastAPI application
WORKDIR /app

# Copy application source code into container
COPY . /app

# Expose port 8000 for network communication
# WHY EXPOSE 8000: Documents that FastAPI application listens on port 8000.
EXPOSE 8000

# Set container entry command to launch FastAPI server via Uvicorn
# WHY --host 0.0.0.0: Binds Uvicorn to all network interfaces inside the container, allowing external traffic from Docker host and other containers to reach the server.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
