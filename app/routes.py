"""
Application routes.
"""

import os
import logging
import time
from datetime import datetime
from flask import Blueprint, render_template, request, jsonify
from werkzeug.utils import secure_filename
from torchvision import transforms
from PIL import Image
import torch

from app.utils.model_cache import model_cache
from app.config import MAX_FILE_SIZE, ALLOWED_EXTENSIONS

# Import training functions (handle both old and new locations)
try:
    from training.resnet_train import visualize_attention
    from training.efficient_train import generate_caption
except ImportError:
    # Fallback for backward compatibility (before reorganization)
    from resnet_train import visualize_attention
    from efficient_train import generate_caption

logger = logging.getLogger(__name__)

bp = Blueprint('main', __name__)

# Image transformation for EfficientNet
efficientnet_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_file_type(file_path):
    """Validate file is actually an image (not just extension)."""
    try:
        img = Image.open(file_path)
        img.verify()
        return True
    except Exception:
        return False


@bp.before_request
def before_request():
    """Log request start time."""
    request.start_time = time.time()


@bp.after_request
def after_request(response):
    """Add security headers and log request duration."""
    # Security headers
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    
    # Log request
    duration = time.time() - request.start_time
    logger.info(f"{request.method} {request.path} - {response.status_code} - {duration:.3f}s")
    
    return response


@bp.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@bp.route('/health')
def health_check():
    """Health check endpoint for load balancers."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'models_loaded': {
            'resnet': model_cache.is_resnet_loaded(),
            'efficientnet': model_cache.is_efficientnet_loaded()
        }
    }), 200


@bp.route('/ready')
def readiness_check():
    """Readiness check - ensures models are loaded."""
    if not model_cache.is_resnet_loaded() and not model_cache.is_efficientnet_loaded():
        return jsonify({'status': 'not ready', 'reason': 'models not loaded'}), 503
    return jsonify({'status': 'ready'}), 200


@bp.route('/upload', methods=['POST'])
def upload_file():
    """Handle image upload and generate caption."""
    if 'image' not in request.files:
        logger.warning("Upload request missing 'image' field")
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['image']
    model_choice = request.form.get('model', 'efficientnet')  # Default to EfficientNet
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only PNG, JPG, JPEG allowed.'}), 400
    
    # Get upload folder from current app (set in __init__.py)
    from flask import current_app
    upload_folder = current_app.config['UPLOAD_FOLDER']
    
    # Save file temporarily
    filename = secure_filename(file.filename)
    filepath = os.path.join(upload_folder, filename)
    
    try:
        file.save(filepath)
        
        # Validate file size
        file_size = os.path.getsize(filepath)
        if file_size > MAX_FILE_SIZE:
            os.remove(filepath)
            return jsonify({'error': f'File too large. Maximum size: {MAX_FILE_SIZE / 1024 / 1024}MB'}), 400
        
        # Validate file is actually an image
        if not validate_file_type(filepath):
            os.remove(filepath)
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Generate caption based on model choice
        start_time = time.time()
        
        if model_choice == 'efficientnet':
            if not model_cache.is_efficientnet_loaded():
                return jsonify({'error': 'EfficientNet model not available'}), 503
            
            model, tokenizer = model_cache.get_efficientnet_model()
            
            # Load and preprocess image
            image = Image.open(filepath).convert('RGB')
            image_tensor = efficientnet_transform(image).to(model_cache._device)
            
            # Generate caption
            with torch.no_grad():
                caption = generate_caption(
                    model, 
                    image_tensor, 
                    tokenizer, 
                    model_cache._device, 
                    max_length=64
                )
        else:  # resnet50
            if not model_cache.is_resnet_loaded():
                return jsonify({'error': 'ResNet model not available'}), 503
            
            encoder, decoder, vocab = model_cache.get_resnet_models()
            
            # Generate caption
            with torch.no_grad():
                caption = visualize_attention(filepath, encoder, decoder, model_cache._device)
        
        inference_time = time.time() - start_time
        logger.info(f"Caption generated in {inference_time:.3f}s using {model_choice}")
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'caption': caption,
            'model': model_choice,
            'inference_time': round(inference_time, 3)
        })
        
    except Exception as e:
        logger.error(f"Error generating caption: {e}", exc_info=True)
        
        # Clean up file on error
        if os.path.exists(filepath):
            os.remove(filepath)
        
        return jsonify({'error': 'Failed to generate caption. Please try again.'}), 500

