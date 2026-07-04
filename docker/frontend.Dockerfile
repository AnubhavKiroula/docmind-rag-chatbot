# Streamlit Frontend Dockerfile
# WHY python:3.10-slim: Provides a compact Linux base image containing Python 3.10 runtime environment.
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Disable output buffering so Streamlit logs stream immediately to console/log-collectors
ENV PYTHONUNBUFFERED=1

# Copy requirements file BEFORE app source code
# WHY layer ordering matters: Docker caches build steps. Copying requirements first ensures that pip install 
# is only re-executed when requirements.txt changes, saving build time when source code is updated.
COPY requirements.txt /app/requirements.txt

# Pre-install CPU-only PyTorch wheel
# WHY: Reduces PyTorch download size from 530MB to 150MB, significantly reducing Docker image size and build time.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install Python production dependencies using CPU wheel index
# WHY --extra-index-url: Ensures sentence-transformers resolves CPU PyTorch dependencies.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Copy full application repository into container working directory
COPY . /app

# Expose port 8501 (default Streamlit port) for network routing
EXPOSE 8501

# Run Streamlit server when container starts
# WHY --server.headless=true: Required when running Streamlit in headless container environments (prevents attempting to auto-open a browser window).
# WHY --server.address=0.0.0.0: Listens across all container network interfaces to accept web browser connections.
CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
