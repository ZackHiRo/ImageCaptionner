"""
Application configuration.
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Flask configuration
SECRET_KEY = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")
FLASK_ENV = os.environ.get("FLASK_ENV", "development")
DEBUG = FLASK_ENV != "production"

# Upload configuration
UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", str(BASE_DIR / "uploads"))
MAX_FILE_SIZE = int(os.environ.get("MAX_FILE_SIZE", 10 * 1024 * 1024))  # 10MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Model paths
MODELS_DIR = BASE_DIR / "models"
OPTIMIZED_MODELS_DIR = MODELS_DIR / "optimized_models"
RESNET_MODEL_PATH = MODELS_DIR / "resnet_best_model.pth"
EFFICIENTNET_MODEL_PATH = MODELS_DIR / "efficient_best_model.pth"
VOCAB_PATH = MODELS_DIR / "vocab.pkl"

# Model configuration
USE_OPTIMIZED_MODELS = os.environ.get("USE_OPTIMIZED_MODELS", "true").lower() == "true"
LOAD_MODELS_ON_STARTUP = os.environ.get("LOAD_MODELS", "true").lower() == "true"

