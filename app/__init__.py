"""
Image Caption Generator - Flask Application
Production-ready application with model caching and security.
"""

from flask import Flask
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_app(config=None):
    """
    Application factory pattern.
    Creates and configures the Flask application.
    """
    # Get base directory (project root)
    import os
    from pathlib import Path
    base_dir = Path(__file__).resolve().parent.parent
    
    app = Flask(__name__, 
                template_folder=str(base_dir / 'templates'),
                static_folder=str(base_dir / 'static'))
    
    # Load configuration
    app.secret_key = os.environ.get("SESSION_SECRET")
    if not app.secret_key or app.secret_key == "default-secret-key":
        if os.environ.get("FLASK_ENV") == "production":
            raise ValueError("SESSION_SECRET must be set in production environment!")
        else:
            logger.warning("Using default secret key. Set SESSION_SECRET in production!")
            app.secret_key = "default-secret-key"
    
    # Configuration
    app.config['UPLOAD_FOLDER'] = os.environ.get('UPLOAD_FOLDER', 'uploads')
    app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get('MAX_FILE_SIZE', 10 * 1024 * 1024))
    app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
    
    # Create uploads directory
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    # Register blueprints/routes
    from app.routes import bp
    app.register_blueprint(bp)
    
    # Download model if needed (before loading)
    if os.environ.get("FLASK_ENV") == "production" or os.environ.get("LOAD_MODELS", "true").lower() == "true":
        try:
            import sys
            sys.path.insert(0, str(base_dir))
            from scripts.download_model import download_efficientnet_model
            download_efficientnet_model()
        except Exception as e:
            logger.warning(f"Could not download model: {e}. Will try to use existing model if available.")
    
    # Initialize models at startup (production)
    if os.environ.get("FLASK_ENV") == "production" or os.environ.get("LOAD_MODELS", "true").lower() == "true":
        logger.info("Initializing models...")
        try:
            from app.utils.model_cache import model_cache
            # Only load EfficientNet model
            model_cache.load_efficientnet_model_only(use_optimized=True)
            logger.info("EfficientNet model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load models: {e}", exc_info=True)
            # Don't raise here - let the app start and handle errors gracefully
    
    return app


# For backward compatibility
app = create_app()

