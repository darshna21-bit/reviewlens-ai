# ── Build stage ───────────────────────────────────────────────────────────────
# Using slim to keep the image small. The full python:3.11 image adds ~400MB
# for things we don't need. slim is ~130MB base and works fine for inference.
FROM python:3.11-slim AS base

# system deps needed by tokenizers (Rust-based) and matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy requirements first so Docker layer cache is hit on code-only changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy the rest of the app
COPY . .

# make sure log and output dirs exist — Render's filesystem is ephemeral
# but we still want the dirs to be there when the app starts
RUN mkdir -p logs data/raw data/processed saved_models/sentiment saved_models/summarization

# ── Runtime stage ──────────────────────────────────────────────────────────────
# non-root user for security — standard practice for containerized web apps
RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app
USER appuser

# Gunicorn config: 2 workers is the sweet spot for Render's free tier (512MB RAM).
# 4 workers would be better for throughput but the two model weights in memory
# push us close to the limit. Set WEB_CONCURRENCY env var to override.
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 7860

CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120", "--chdir", "/app", "app:app"]
