# =============================================================================
# Groksito Discord Bot — Multi-stage Production Dockerfile
# =============================================================================
# We use stages so that:
#   - `docker compose build` (or build target "bot") produces the full bot image
#     needed by the Discord bot (discord.py, openai, pydub, ffmpeg, etc.).
#   - `docker compose build web` (or build target "web") produces a much smaller
#     image (~280-400 MB range) containing only the FastAPI dashboard + the tiny
#     subset of bot modules it actually imports (env_utils + config).
#
# This fixes the previous situation where both services used the identical heavy
# single-stage build and both images reported ~1.04 GB.
# =============================================================================

# -----------------------------------------------------------------------------
# Base stage — common slim Python + minimal system tools
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS base

WORKDIR /app

# Common minimal system packages needed by both bot and web.
# We deliberately do NOT install ffmpeg here (bot-only).
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Bot stage — full runtime for the Discord agent + power tools
# -----------------------------------------------------------------------------
FROM base AS bot

ARG IMAGE_VERSION=dev
ARG IMAGE_REVISION=unknown
ARG IMAGE_TARGET=bot

LABEL org.opencontainers.image.title="groksito-discord-bot" \
      org.opencontainers.image.description="Standalone Groksito Discord Bot (Discord agent + media tools)" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      org.opencontainers.image.revision="${IMAGE_REVISION}" \
      org.opencontainers.image.source="https://github.com/lupintic/groksito-discord-bot" \
      org.opencontainers.image.licenses="MIT"

# System dependencies required only by the bot:
# - ffmpeg: audio (pydub) and video handling
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy packaging metadata + requirements first for better layer caching
COPY pyproject.toml README.md requirements.txt .

# Install deps then register the package (canonical import: groksito_discord)
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --no-deps -e .

# Copy dashboard (not part of the wheel)
COPY web ./web

# Create runtime directories (good for volume mounts even before first write)
RUN mkdir -p /app/data && chmod 755 /app/data \
 && mkdir -p /app/oauth && chmod 755 /app/oauth

# Environment defaults
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LOG_LEVEL=INFO \
    DATA_DIR=/app/data

# Default process for the bot service (compose can still override)
CMD ["groksito"]

# -----------------------------------------------------------------------------
# Web stage — slim dashboard-only image (FastAPI + the modules it actually uses)
# -----------------------------------------------------------------------------
FROM base AS web

ARG IMAGE_VERSION=dev
ARG IMAGE_REVISION=unknown
ARG IMAGE_TARGET=web

LABEL org.opencontainers.image.title="groksito-discord-bot-web" \
      org.opencontainers.image.description="Groksito FastAPI configuration dashboard (web target)" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      org.opencontainers.image.revision="${IMAGE_REVISION}" \
      org.opencontainers.image.source="https://github.com/lupintic/groksito-discord-bot" \
      org.opencontainers.image.licenses="MIT"

# Web-only minimal dependencies.
# See requirements-web.txt for the explicit list and rationale.
COPY pyproject.toml README.md requirements-web.txt .

COPY src ./src
COPY web ./web

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-web.txt && \
    pip install --no-cache-dir --no-deps -e .

# Same directories as the bot image so volume mounts and path logic in web/main.py
# (editable install + DATA_DIR) continue to work without surprises.
RUN mkdir -p /app/data && chmod 755 /app/data \
 && mkdir -p /app/oauth && chmod 755 /app/oauth

# Environment defaults (web service)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LOG_LEVEL=INFO \
    DATA_DIR=/app/data

# Sensible default for the web dashboard (compose overrides with the same command,
# but this makes `docker run <web-image>` work out of the box on port 8010).
CMD ["uvicorn", "web.main:app", "--host", "0.0.0.0", "--port", "8010"]
# Railway builds the last stage unless a target is set
FROM bot AS railway
