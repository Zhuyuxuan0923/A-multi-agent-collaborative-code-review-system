FROM python:3.11-slim
WORKDIR /app

# System deps for building sentence-transformers and other native packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry==1.8.3

# Copy dependency files first (layer caching: deps only rebuild when lock changes)
COPY pyproject.toml poetry.lock ./

# Remove tsinghua mirror (not accessible from Docker) and install deps
RUN sed -i '/tsinghua/,/primary/d' pyproject.toml && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev --no-interaction --no-ansi --no-root && \
    rm -rf /root/.cache/pypoetry

# Copy source code
COPY src/ ./src/

# Create data directory for SQLite
RUN mkdir -p /app/data

EXPOSE 8000
CMD ["uvicorn", "study_agent.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
