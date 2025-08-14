# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

# Base image
FROM python:3.12-slim

# Image metadata
LABEL maintainer="NVIDIA"
LABEL description="Voice Agent WebRTC example"
LABEL version="1.0"

# Environment setup
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglx-mesa0 \
    curl \
    ffmpeg \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade pip uv

# App directory setup
WORKDIR /app

# App files
COPY pyproject.toml uv.lock \
     LICENSE README.md NVIDIA_PIPECAT.md \
     ./
COPY src/ ./src/
COPY examples/voice_agent_webrtc/ ./examples/voice_agent_webrtc/

# Example app directory
WORKDIR /app/examples/voice_agent_webrtc

# Dependencies
RUN uv sync --frozen

# Port configuration
EXPOSE 7860

# Start command
CMD ["uv", "run", "pipeline.py"]