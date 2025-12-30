"""
Model Caching Module for Production
Loads models once at startup and reuses them for all requests.
This eliminates the overhead of loading models per-request.
"""

import torch
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Get base directory (project root)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"


class ModelCache:
    """Singleton class to cache loaded models in memory."""
    
    def __init__(self):
        self._resnet_encoder = None
        self._resnet_decoder = None
        self._resnet_vocab = None
        self._efficientnet_model = None
        self._efficientnet_tokenizer = None
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._models_loaded = False
        
        logger.info(f"ModelCache initialized on device: {self._device}")
    
    def load_all_models(self, 
                       resnet_path=None,
                       efficientnet_path=None,
                       use_optimized=True):
        """
        Load all models at startup.
        
        Args:
            resnet_path: Path to ResNet checkpoint (default: models/resnet_best_model.pth)
            efficientnet_path: Path to EfficientNet checkpoint (default: models/efficient_best_model.pth)
            use_optimized: If True, try to load optimized models first
        """
        if self._models_loaded:
            logger.warning("Models already loaded, skipping")
            return
        
        # Set default paths
        if resnet_path is None:
            resnet_path = str(MODELS_DIR / "resnet_best_model.pth")
        if efficientnet_path is None:
            efficientnet_path = str(MODELS_DIR / "efficient_best_model.pth")
        
        # Try optimized models first if requested
        if use_optimized:
            # Check multiple possible locations for optimized models
            optimized_resnet_paths = [
                str(MODELS_DIR / "optimized_models" / "resnet_resnet_best_model_quantized.pth"),
                str(BASE_DIR / "optimized_models" / "resnet_resnet_best_model_quantized.pth"),
                resnet_path.replace('.pth', '_quantized.pth'),
                resnet_path.replace('resnet_best_model.pth', 'resnet_resnet_best_model_quantized.pth'),
            ]
            
            optimized_efficient_paths = [
                str(MODELS_DIR / "optimized_models" / "efficientnet_efficient_best_model_quantized.pth"),
                str(BASE_DIR / "optimized_models" / "efficientnet_efficient_best_model_quantized.pth"),
                efficientnet_path.replace('.pth', '_quantized.pth'),
                efficientnet_path.replace('efficient_best_model.pth', 'efficientnet_efficient_best_model_quantized.pth'),
            ]
            
            # Find optimized ResNet model
            for opt_path in optimized_resnet_paths:
                if os.path.exists(opt_path):
                    resnet_path = opt_path
                    logger.info(f"Using optimized ResNet model: {resnet_path}")
                    break
            
            # Find optimized EfficientNet model
            for opt_path in optimized_efficient_paths:
                if os.path.exists(opt_path):
                    efficientnet_path = opt_path
                    logger.info(f"Using optimized EfficientNet model: {efficientnet_path}")
                    break
        
        # Load EfficientNet only (ResNet skipped)
        try:
            self.load_efficientnet_model(efficientnet_path)
            logger.info("EfficientNet model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load EfficientNet model: {e}", exc_info=True)
        
        self._models_loaded = True
    
    def load_efficientnet_model_only(self, use_optimized=True):
        """
        Load only EfficientNet model (skip ResNet).
        Useful when only EfficientNet is needed.
        """
        if self._models_loaded:
            logger.warning("Models already loaded, skipping")
            return
        
        efficientnet_path = str(MODELS_DIR / "efficient_best_model.pth")
        
        # Try optimized model first if requested
        if use_optimized:
            optimized_efficient_paths = [
                str(MODELS_DIR / "optimized_models" / "efficientnet_efficient_best_model_quantized.pth"),
                str(BASE_DIR / "optimized_models" / "efficientnet_efficient_best_model_quantized.pth"),
                efficientnet_path.replace('.pth', '_quantized.pth'),
                efficientnet_path.replace('efficient_best_model.pth', 'efficientnet_efficient_best_model_quantized.pth'),
            ]
            
            # Find optimized EfficientNet model
            for opt_path in optimized_efficient_paths:
                if os.path.exists(opt_path):
                    efficientnet_path = opt_path
                    logger.info(f"Using optimized EfficientNet model: {efficientnet_path}")
                    break
        
        # Load EfficientNet
        try:
            self.load_efficientnet_model(efficientnet_path)
            logger.info("EfficientNet model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load EfficientNet model: {e}", exc_info=True)
        
        self._models_loaded = True
    
    def load_resnet_models(self, checkpoint_path=None):
        """Load ResNet encoder and decoder models."""
        if self._resnet_encoder is not None:
            return self._resnet_encoder, self._resnet_decoder, self._resnet_vocab
        
        if checkpoint_path is None:
            checkpoint_path = str(MODELS_DIR / "resnet_best_model.pth")
        
        # Resolve path - try multiple locations
        checkpoint_path = self._resolve_model_path(checkpoint_path)
        
        logger.info(f"Loading ResNet models from {checkpoint_path}")
        
        # Import from training module (handles both old and new locations)
        # Need to do this BEFORE loading checkpoint to avoid pickle issues
        try:
            from training.resnet_train import EncoderCNN, DecoderRNN
            # Add to sys.modules to help with pickle loading
            import sys
            if 'resnet_train' not in sys.modules:
                sys.modules['resnet_train'] = sys.modules['training.resnet_train']
        except ImportError:
            try:
                # Fallback for backward compatibility
                import sys
                sys.path.insert(0, str(BASE_DIR))
                from resnet_train import EncoderCNN, DecoderRNN
            except ImportError:
                logger.error("Could not import ResNet model classes. Make sure resnet_train.py exists in training/ or root.")
                raise
        
        # Load checkpoint with proper module mapping
        import sys
        import importlib.util
        
        # Map old module names for pickle compatibility
        if 'resnet_train' not in sys.modules:
            try:
                spec = importlib.util.spec_from_file_location("resnet_train", str(BASE_DIR / "training" / "resnet_train.py"))
                if spec and spec.loader:
                    resnet_module = importlib.util.module_from_spec(spec)
                    sys.modules['resnet_train'] = resnet_module
                    spec.loader.exec_module(resnet_module)
            except Exception:
                pass
        
        checkpoint = torch.load(checkpoint_path, map_location=self._device, weights_only=False)
        
        # Initialize models
        self._resnet_encoder = EncoderCNN().to(self._device)
        self._resnet_decoder = DecoderRNN().to(self._device)
        
        # Load weights
        self._resnet_encoder.load_state_dict(checkpoint['encoder'])
        self._resnet_decoder.load_state_dict(checkpoint['decoder'])
        
        # Set to eval mode
        self._resnet_encoder.eval()
        self._resnet_decoder.eval()
        
        # Store vocabulary
        self._resnet_vocab = checkpoint.get('vocab')
        
        # Warm up models (first inference is slower)
        logger.info("Warming up ResNet models...")
        dummy_input = torch.randn(1, 3, 224, 224).to(self._device)
        with torch.no_grad():
            _ = self._resnet_encoder(dummy_input)
        logger.info("ResNet models warmed up")
        
        return self._resnet_encoder, self._resnet_decoder, self._resnet_vocab
    
    def load_efficientnet_model(self, checkpoint_path=None):
        """Load EfficientNet model."""
        if self._efficientnet_model is not None:
            return self._efficientnet_model, self._efficientnet_tokenizer
        
        if checkpoint_path is None:
            checkpoint_path = str(MODELS_DIR / "efficient_best_model.pth")
        
        # Resolve path - try multiple locations
        checkpoint_path = self._resolve_model_path(checkpoint_path)
        
        logger.info(f"Loading EfficientNet model from {checkpoint_path}")
        
        # Import from training module (handles both old and new locations)
        try:
            from training.efficient_train import Encoder, Decoder, ImageCaptioningModel
        except ImportError:
            try:
                # Fallback for backward compatibility
                import sys
                sys.path.insert(0, str(BASE_DIR))
                from efficient_train import Encoder, Decoder, ImageCaptioningModel
            except ImportError:
                logger.error("Could not import EfficientNet model classes. Make sure efficient_train.py exists in training/ or root.")
                raise
        
        from transformers import AutoTokenizer
        
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        special_tokens = {'additional_special_tokens': ['<start>', '<end>']}
        tokenizer.add_special_tokens(special_tokens)
        self._efficientnet_tokenizer = tokenizer
        
        # Initialize model
        encoder = Encoder(model_name='efficientnet_b3', embed_dim=512)
        decoder = Decoder(
            vocab_size=len(tokenizer),
            embed_dim=512,
            num_layers=8,
            num_heads=8,
            max_seq_length=64
        )
        self._efficientnet_model = ImageCaptioningModel(encoder, decoder).to(self._device)
        
        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location=self._device, weights_only=False)
        
        # Check if this is a quantized model (has _packed_params keys)
        is_quantized = any('_packed_params' in key for key in checkpoint.get('model_state', checkpoint).keys())
        
        if is_quantized:
            # For quantized models, we need to prepare the model for quantization first
            logger.info("Detected quantized model, preparing model for quantization...")
            try:
                # Prepare model for quantization
                import torch.quantization as quant
                self._efficientnet_model = quant.quantize_dynamic(
                    self._efficientnet_model, {torch.nn.Linear}, dtype=torch.qint8
                )
                logger.info("Model prepared for quantization")
            except Exception as e:
                logger.warning(f"Could not prepare model for quantization: {e}. Trying to load anyway...")
        
        if 'model_state' in checkpoint:
            try:
                self._efficientnet_model.load_state_dict(checkpoint['model_state'], strict=False)
            except Exception as e:
                logger.warning(f"Could not load quantized state dict: {e}. Trying regular model...")
                # Try loading non-quantized model instead
                regular_path = checkpoint_path.replace('_quantized.pth', '.pth').replace('efficientnet_efficient_best_model', 'efficient_best_model')
                if os.path.exists(regular_path) and regular_path != checkpoint_path:
                    logger.info(f"Trying regular model: {regular_path}")
                    checkpoint = torch.load(regular_path, map_location=self._device, weights_only=False)
                    if 'model_state' in checkpoint:
                        self._efficientnet_model.load_state_dict(checkpoint['model_state'])
                    else:
                        self._efficientnet_model.load_state_dict(checkpoint)
        else:
            # Fallback: try loading directly
            try:
                self._efficientnet_model.load_state_dict(checkpoint, strict=False)
            except Exception:
                logger.warning("Could not load state dict. Model may not work correctly.")
        
        self._efficientnet_model.eval()
        
        # Warm up
        logger.info("Warming up EfficientNet model...")
        dummy_input = torch.randn(1, 3, 224, 224).to(self._device)
        with torch.no_grad():
            _ = self._efficientnet_model.encoder(dummy_input)
        logger.info("EfficientNet model warmed up")
        
        return self._efficientnet_model, self._efficientnet_tokenizer
    
    def _resolve_model_path(self, checkpoint_path):
        """Resolve model path, trying multiple locations."""
        # If path exists, use it
        if os.path.exists(checkpoint_path):
            return checkpoint_path
        
        # Try in models directory
        alt_path = str(MODELS_DIR / os.path.basename(checkpoint_path))
        if os.path.exists(alt_path):
            logger.info(f"Found model at: {alt_path}")
            return alt_path
        
        # Try in root directory (backward compatibility)
        alt_path = str(BASE_DIR / os.path.basename(checkpoint_path))
        if os.path.exists(alt_path):
            logger.info(f"Found model at: {alt_path}")
            return alt_path
        
        # Return original path (will fail with clear error)
        return checkpoint_path
    
    def get_resnet_models(self):
        """Get cached ResNet models."""
        if self._resnet_encoder is None:
            raise RuntimeError("ResNet models not loaded. Call load_resnet_models() first.")
        return self._resnet_encoder, self._resnet_decoder, self._resnet_vocab
    
    def get_efficientnet_model(self):
        """Get cached EfficientNet model."""
        if self._efficientnet_model is None:
            raise RuntimeError("EfficientNet model not loaded. Call load_efficientnet_model() first.")
        return self._efficientnet_model, self._efficientnet_tokenizer
    
    def is_resnet_loaded(self):
        """Check if ResNet models are loaded."""
        return self._resnet_encoder is not None
    
    def is_efficientnet_loaded(self):
        """Check if EfficientNet model is loaded."""
        return self._efficientnet_model is not None


# Singleton instance
model_cache = ModelCache()
