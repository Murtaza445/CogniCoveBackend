# CogniCove Backend — Docker image for Render / VPS / Cloud Run
# Uses Python 3.11 for compatibility with PyTorch + LangChain ecosystem

FROM python:3.11-slim

WORKDIR /app

# Install system libraries required by opencv, librosa, soundfile, faiss, and piper
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    libespeak-ng1 \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------
# Download Linux Piper binary (Windows piper.exe in repo won't work)
# If this URL breaks, update it from: https://github.com/rhasspy/piper/releases
# ------------------------------------------------------------------
RUN mkdir -p /opt/piper && \
    wget -qO- \
    "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz" | \
    tar xz -C /opt/piper --strip-components=1 && \
    chmod +x /opt/piper/piper

# Install Python deps first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application + model weights
COPY . .

# ------------------------------------------------------------------
# Fail fast if someone pushed Git LFS pointers instead of real weights
# ------------------------------------------------------------------
RUN python deploy_validate.py --fail-on-error

# Runtime env defaults (override in Render dashboard if needed)
ENV PYTHONUNBUFFERED=1
ENV VOSK_MODEL_PATH=/app/models/vosk-model
ENV PIPER_BIN=/opt/piper/piper
ENV PIPER_MODEL=/app/models/piper/en_US-lessac-medium.onnx
ENV VOSK_SAMPLE_RATE=16000

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
