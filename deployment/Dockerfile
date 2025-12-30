FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt', quiet=True)"

# Copy application files
COPY . .

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

