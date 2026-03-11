# ---- Rio Cloud Service ----
# Lightweight image for the FastAPI + Gemini relay.

FROM python:3.11-slim

# Prevent Python from buffering stdout/stderr (important for Cloud Run logs)
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Run as non-root user for security (Cloud Run best practice)
RUN adduser --disabled-password --gecos "" riouser \
    && chown -R riouser:riouser /app
USER riouser

# Cloud Run default port
EXPOSE 8080

# Health check so Cloud Run knows when the container is ready
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Start the direct Live API server
CMD ["uvicorn", "adk_server:app", "--host", "0.0.0.0", "--port", "8080"]
