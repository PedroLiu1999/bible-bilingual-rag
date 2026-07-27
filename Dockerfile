FROM python:3.11-slim

# Install system build tools (C++ compiler required for compiling native dependencies)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt-get/lists/*

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

ENV UV_PYTHON=python3.11

WORKDIR /app

# Copy dependency configs
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-cache

# Copy application source code
COPY . .
RUN uv pip install -e .

EXPOSE 8000

# Start FastAPI application with uvicorn
CMD ["uv", "run", "python", "app.py"]
