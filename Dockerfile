# ------------------------------------------------------------
# AGRARIAN Dockerfile Template
# ------------------------------------------------------------
# Purpose: Provides a language-agnostic starting point
#          for containerized applications within the
#          AGRARIAN ecosystem.
#
# Usage:   Adapt it for your preferred
#          runtime environment.
# ------------------------------------------------------------

# ============================================================
# 🐍 Python Example
# ============================================================

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN adduser --system --group appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost/health || exit 1

# Run application
CMD ["python", "src/app.py"]