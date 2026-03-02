# Base image
FROM python:3.12-slim

# Image metadata
LABEL maintainer="NVIDIA"
LABEL description="Voice Agent WebSocket example"
LABEL version="1.0"

# Environment setup
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN pip install --no-cache-dir --upgrade pip uv

# App directory setup
WORKDIR /app

# App files
COPY pyproject.toml uv.lock \
     LICENSE README.md NVIDIA_PIPECAT.md \
     ./
COPY src/ ./src/
COPY examples/voice_agent_websocket/ ./examples/voice_agent_websocket/

# Example app directory
WORKDIR /app/examples/voice_agent_websocket

# Dependencies
RUN uv venv && . .venv/bin/activate && uv sync --frozen

# Port configuration
EXPOSE 8100

# Start command
CMD ["uv", "run", "bot.py"]