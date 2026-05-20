# Backend Dockerfile — CPU-only inference service.
# Build context is the PROJECT ROOT (churn_project/), so this image can copy in
# both the backend code and the trained artifacts in outputs/.
#
#   docker build -f backend/Dockerfile -t churn-api .
#   docker run -p 8000:8000 churn-api
FROM python:3.12-slim

# Faster, quieter Python in containers.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first so Docker can cache this layer across code changes.
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy the application code, the trained model, the report charts, and the
# dataset (used by SHAP as a background sample at startup).
COPY backend/ ./backend/
COPY outputs/ ./outputs/
COPY data/ ./data/

EXPOSE 8000

# Render (and most PaaS) inject the port via $PORT; default to 8000 locally.
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
