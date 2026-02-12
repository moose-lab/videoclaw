FROM python:3.12-slim AS base

# System deps for FFmpeg and build tools.
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv for fast dependency resolution.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project metadata first for layer caching.
COPY pyproject.toml ./
COPY src/ src/

# Install the project.
RUN uv pip install --system -e ".[server]"

# Default config via env vars.
ENV VIDEOCLAW_PROJECTS_DIR=/data/projects \
    VIDEOCLAW_MODELS_DIR=/data/models_cache \
    VIDEOCLAW_LOG_LEVEL=info

VOLUME /data

EXPOSE 8000

CMD ["uvicorn", "videoclaw.server.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
