
FROM python:3.13-slim AS base

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app


COPY pyproject.toml uv.lock ./
# Install dependencies
RUN uv sync --frozen --no-dev

# Application code
COPY src/ src/
COPY api/ api/
COPY models/ models/
COPY data/processed/ data/processed/

# Logs directory
RUN mkdir -p logs

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
