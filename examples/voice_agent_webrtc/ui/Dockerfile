# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

# Build stage
FROM node:18-alpine AS builder

# Image metadata
LABEL stage="builder"
LABEL description="UI build environment"

# Set working directory
WORKDIR /app

# Copy package files and install dependencies
COPY ui/package*.json ./
RUN npm ci --frozen-lockfile \
    && npm cache clean --force

# Copy source code and build
COPY ui/ .
RUN npm run build \
    && rm -rf node_modules

# Production stage
FROM python:3.12-alpine AS production

# Image metadata
LABEL maintainer="NVIDIA"
LABEL description="Voice Agent WebRTC UI static server"
LABEL version="1.0"

# Set working directory and copy built assets
WORKDIR /app
COPY --from=builder /app/dist ./static

# Port configuration
EXPOSE 8000

# Start command
CMD ["python", "-m", "http.server", "8000", "--directory", "static"]