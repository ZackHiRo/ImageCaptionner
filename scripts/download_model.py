"""
Download EfficientNet model from cloud storage if not present.
This script runs at application startup to download the model if needed.
"""

import os
import sys
import logging
from pathlib import Path
import requests

logger = logging.getLogger(__name__)

def download_efficientnet_model():
    """
    Download EfficientNet optimized model if it doesn't exist.
    Model URL should be set in EFFICIENTNET_MODEL_URL environment variable.
    """
    # Get base directory
    base_dir = Path(__file__).resolve().parent.parent
    models_dir = base_dir / "models" / "optimized_models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = models_dir / "efficientnet_efficient_best_model_quantized.pth"
    
    # Check if model already exists
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        logger.info(f"EfficientNet model already exists ({size_mb:.1f}MB)")
        return True
    
    # Get download URL from environment
    model_url = os.environ.get("EFFICIENTNET_MODEL_URL")
    
    if not model_url:
        logger.warning("EFFICIENTNET_MODEL_URL not set. Model will not be downloaded.")
        logger.warning("Set EFFICIENTNET_MODEL_URL environment variable to download the model.")
        return False
    
    try:
        logger.info(f"Downloading EfficientNet model from {model_url}...")
        logger.info("This may take a few minutes (model is ~245MB)...")
        
        # Download with progress
        response = requests.get(model_url, stream=True, timeout=300)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        if downloaded % (10 * 1024 * 1024) == 0:  # Log every 10MB
                            logger.info(f"Downloaded {downloaded / (1024 * 1024):.1f}MB / {total_size / (1024 * 1024):.1f}MB ({percent:.1f}%)")
        
        size_mb = model_path.stat().st_size / (1024 * 1024)
        logger.info(f"EfficientNet model downloaded successfully ({size_mb:.1f}MB)")
        return True
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download model: {e}")
        # Clean up partial download
        if model_path.exists():
            model_path.unlink()
        return False
    except Exception as e:
        logger.error(f"Error downloading model: {e}", exc_info=True)
        # Clean up partial download
        if model_path.exists():
            model_path.unlink()
        return False

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    download_efficientnet_model()

