# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Reduce pip's output and force UTF-8
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

# System deps: ffmpeg for splitting/processing audio, libsndfile for soundfile
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project metadata first for better layer caching
COPY pyproject.toml README.md /app/
COPY transcriber /app/transcriber

# Install package
RUN python -m pip install -U pip \
 && pip install --no-cache-dir .

# Pre-download the default model (medium.en) into the image layer cache
# This ensures first run doesn't have to download the model
RUN python - << 'EOH'
from transcriber.whisper_utils import load_model
# CPU device ensures the model downloads in a generic format
load_model('medium.en', device='cpu')
EOH

# Default entrypoint to the CLI
ENTRYPOINT ["vibe-transcriber"]
