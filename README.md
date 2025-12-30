# Image Caption Generator

Production-ready image captioning application using EfficientNet and ResNet models.

## 🚀 Quick Start

### Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variable
export SESSION_SECRET=$(openssl rand -hex 32)  # Linux/Mac
# or
$env:SESSION_SECRET = -join ((48..57) + (97..102) | Get-Random -Count 64 | ForEach-Object {[char]$_})  # Windows

# Run development server
python main.py
```

### Production

```bash
# Using Gunicorn
gunicorn app_production:app --bind 0.0.0.0:5000 --workers 2

# Using Docker
docker build -f deployment/Dockerfile -t image-captioner .
docker run -p 5000:5000 -e SESSION_SECRET=your-secret image-captioner
```

## 📁 Project Structure

```
project/
├── app/                  # Application package
│   ├── __init__.py      # Flask app factory
│   ├── routes.py        # API routes
│   ├── config.py        # Configuration
│   └── utils/           # Utilities
│       └── model_cache.py
│
├── training/            # Training scripts
├── scripts/             # Utility scripts
├── deployment/          # Deployment configs
├── docs/                # Documentation
├── models/              # Saved models
│   └── optimized_models/
├── static/              # Static files
└── templates/           # HTML templates
```

## 📚 Documentation

- [Deployment Guide](docs/DEPLOYMENT_GUIDE.md)
- [Production Deployment](docs/PRODUCTION_DEPLOYMENT.md)
- [Render Deployment](docs/DEPLOY_RENDER.md)
- [DigitalOcean Deployment](docs/DEPLOY_DIGITALOCEAN_DROPLET.md)

## 🔧 Configuration

Set environment variables:
- `SESSION_SECRET` - Required for production
- `FLASK_ENV` - Set to `production` for production
- `USE_OPTIMIZED_MODELS` - Use quantized models (default: true)
- `LOAD_MODELS` - Load models on startup (default: true)

## 🎯 Features

- ✅ EfficientNet and ResNet model support
- ✅ Model caching for fast inference
- ✅ Production-ready with security headers
- ✅ Health check endpoints
- ✅ Optimized quantized models
- ✅ Docker support

## 📝 License

[Your License Here]

