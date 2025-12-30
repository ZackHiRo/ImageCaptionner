#!/bin/bash
# Quick setup script for DigitalOcean droplet deployment

set -e

echo "🚀 Setting up Image Captioner on DigitalOcean Droplet"
echo "=================================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    echo "✅ Docker installed. You may need to log out and back in."
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Installing Docker Compose..."
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    echo "✅ Docker Compose installed."
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    SECRET=$(openssl rand -hex 32)
    cat > .env << EOF
SESSION_SECRET=$SECRET
FLASK_ENV=production
PYTHONUNBUFFERED=1
EOF
    echo "✅ .env file created with secure SESSION_SECRET"
else
    echo "✅ .env file already exists"
fi

# Check if optimized model exists
if [ ! -f "optimized_models/efficientnet_efficient_best_model_quantized.pth" ]; then
    echo "⚠️  Warning: Optimized model not found!"
    echo "   Make sure optimized_models/efficientnet_efficient_best_model_quantized.pth exists"
    echo "   Or run: python optimize_models.py --model efficientnet --method quantize"
fi

# Check port availability
echo "🔍 Checking port availability..."
if lsof -Pi :5000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "⚠️  Port 5000 is already in use!"
    echo "   Update docker-compose.yml to use a different port (e.g., 5001:5000)"
    read -p "   Press Enter to continue anyway, or Ctrl+C to cancel..."
fi

# Build and start
echo "🔨 Building Docker image..."
docker-compose build

echo "🚀 Starting services..."
docker-compose up -d

echo ""
echo "✅ Setup complete!"
echo ""
echo "📊 Service Status:"
docker-compose ps

echo ""
echo "📋 Next steps:"
echo "   1. Check logs: docker-compose logs -f image-captioner"
echo "   2. Test health: curl http://localhost:5000/health"
echo "   3. Access app: http://your-droplet-ip:5000"
echo ""
echo "🔧 Useful commands:"
echo "   - View logs: docker-compose logs -f image-captioner"
echo "   - Restart: docker-compose restart image-captioner"
echo "   - Stop: docker-compose down"
echo "   - Update: git pull && docker-compose up -d --build"
echo ""

