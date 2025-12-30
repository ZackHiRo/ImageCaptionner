FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install requests for model downloading
RUN pip install --no-cache-dir requests

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt', quiet=True)"

# Copy application files (new structure)
COPY app/ /app/app/
COPY training/ /app/training/
COPY scripts/ /app/scripts/
COPY templates/ /app/templates/
COPY static/ /app/static/
COPY app_production.py /app/
COPY main.py /app/

# Create models directory (models will be downloaded at runtime)
RUN mkdir -p /app/models/optimized_models

# Create uploads directory
RUN mkdir -p uploads

# Expose port
EXPOSE 5000

# Set environment variable for production
ENV FLASK_ENV=production

# Run the application (using production app with model caching)
# Use $PORT environment variable (Render provides this) or default to 5000
# Using shell form to allow environment variable expansion
CMD sh -c "gunicorn app_production:app --bind 0.0.0.0:${PORT:-5000} --workers 1 --timeout 120 --threads 2"

