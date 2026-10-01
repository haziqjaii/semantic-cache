# The semantic cache server.
#
# Configuration (GEMINI_API_KEY, REDIS_URL, ...) comes from environment
# variables at run time. No .env file or secret is copied into the image.

FROM python:3.13-slim

# uv installs the locked dependencies (the same versions as in development).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

# Dependencies first: this layer is rebuilt only when they change.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# Then the application itself.
COPY src ./src
RUN uv sync --frozen --no-dev

EXPOSE 8000

# One worker: the metrics are kept in the process's memory (see the README).
CMD ["uv", "run", "--no-dev", "uvicorn", "semcache.main:app", "--host", "0.0.0.0", "--port", "8000"]
