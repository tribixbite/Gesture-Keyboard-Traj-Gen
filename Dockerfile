FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY *.py ./
COPY holokeyboard.csv ./
COPY docs/ ./docs/
COPY Jerk-Minimization/ ./Jerk-Minimization/

# Create directories for output
RUN mkdir -p synthetic_traces memory

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV NUM_WORKERS=4

# Expose port for API
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command - run API server
CMD ["python", "api_server.py"]