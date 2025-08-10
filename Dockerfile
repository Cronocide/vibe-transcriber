# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Reduce pip's output and force UTF-8
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    WHISPER_MODEL_DIR=/models

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

# Create a volume location for models; operator can mount a persistent directory here
VOLUME ["/models"]

# Default entrypoint to the CLI (alias provided in pyproject)
ENTRYPOINT ["vibe-transcriber"]
