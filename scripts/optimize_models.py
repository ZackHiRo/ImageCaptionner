"""
Model Optimization Script for Production Deployment
Reduces model size and improves inference speed through:
1. Quantization (INT8)
2. TorchScript compilation
3. Model pruning (optional)
4. State dict optimization
"""

import torch
import os
import argparse
from pathlib import Path

# Import model classes BEFORE loading checkpoints (needed for unpickling)
# This ensures PyTorch can find the class definitions when loading saved objects
# Note: resnet_train.py has module-level code that loads COCO data, which may fail
# if training files aren't present. We'll handle this in the functions.

def quantize_model(checkpoint_path, output_path, model_type='resnet'):
    """
    Quantize model to INT8 for 4x size reduction and faster inference.
    Note: Slight accuracy loss (usually <1%)
    """
    print(f"Quantizing {model_type} model...")
    
    device = torch.device('cpu')  # Quantization typically done on CPU
    
    # Import classes before loading (required for unpickling)
    # resnet_train.py now handles missing training data gracefully
    if model_type == 'resnet':
        # Import the module itself so we can update vocab later
        import resnet_train
        from resnet_train import EncoderCNN, DecoderRNN, Vocabulary
        
        # Make Vocabulary available in __main__ for unpickling
        # This handles cases where checkpoint was saved with Vocabulary from __main__
        import __main__
        if not hasattr(__main__, 'Vocabulary'):
            __main__.Vocabulary = Vocabulary
    elif model_type == 'efficientnet':
        from efficient_train import Encoder, Decoder, ImageCaptioningModel
        from transformers import AutoTokenizer
    
    # Load checkpoint (now all classes are available for unpickling)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if model_type == 'resnet':
        # For ResNet, quantize encoder and decoder separately
        
        # IMPORTANT: Update vocab from checkpoint before creating DecoderRNN
        # The decoder uses len(vocab.word2idx) in its __init__, so we need the full vocab
        if 'vocab' in checkpoint and checkpoint['vocab'] is not None:
            # Update the vocab in resnet_train module (DecoderRNN.__init__ references resnet_train.vocab)
            resnet_train.vocab = checkpoint['vocab']
            print(f"  Updated vocab size: {len(checkpoint['vocab'].word2idx)}")
        else:
            raise ValueError("Checkpoint does not contain 'vocab' key. Cannot proceed.")
        
        encoder = EncoderCNN()
        decoder = DecoderRNN()  # Now uses the correct vocab size from checkpoint
        
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
        
        # Set to eval mode
        encoder.eval()
        decoder.eval()
        
        # Prepare for quantization (dummy input)
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # Quantize encoder (only Linear and Conv2d layers)
        encoder_quantized = torch.quantization.quantize_dynamic(
            encoder, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
        )
        
        # Quantize decoder (only Linear layers - Embedding requires special config)
        # Embeddings are typically small and don't benefit much from quantization
        decoder_quantized = torch.quantization.quantize_dynamic(
            decoder, {torch.nn.Linear}, dtype=torch.qint8
        )
        
        # Save quantized model
        quantized_checkpoint = {
            'encoder': encoder_quantized.state_dict(),
            'decoder': decoder_quantized.state_dict(),
            'vocab': checkpoint.get('vocab'),
            'quantized': True
        }
        
    elif model_type == 'efficientnet':
        # Classes already imported above before loading checkpoint
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        special_tokens = {'additional_special_tokens': ['<start>', '<end>']}
        tokenizer.add_special_tokens(special_tokens)
        
        encoder = Encoder(model_name='efficientnet_b3', embed_dim=512)
        decoder = Decoder(
            vocab_size=len(tokenizer),
            embed_dim=512,
            num_layers=8,
            num_heads=8,
            max_seq_length=64
        )
        model = ImageCaptioningModel(encoder, decoder)
        
        # Load state dict - handle both 'model_state' key and direct state dict
        if 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        # Quantize the full model
        model_quantized = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
        )
        
        quantized_checkpoint = {
            'model_state': model_quantized.state_dict(),
            'quantized': True
        }
    
    torch.save(quantized_checkpoint, output_path)
    
    # Compare sizes
    original_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB
    quantized_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    reduction = (1 - quantized_size / original_size) * 100
    
    print(f"✓ Quantization complete!")
    print(f"  Original size: {original_size:.2f} MB")
    print(f"  Quantized size: {quantized_size:.2f} MB")
    print(f"  Size reduction: {reduction:.1f}%")
    
    return output_path


def optimize_state_dict(checkpoint_path, output_path):
    """
    Remove unnecessary metadata and optimize state dict for smaller size.
    """
    print(f"Optimizing state dict...")
    
    # Import classes before loading (required for unpickling)
    try:
        from resnet_train import Vocabulary
        # Make Vocabulary available in __main__ for unpickling
        import __main__
        if not hasattr(__main__, 'Vocabulary'):
            __main__.Vocabulary = Vocabulary
    except ImportError:
        pass
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Create optimized checkpoint with only essential data
    optimized = {}
    for key, value in checkpoint.items():
        if key not in ['optimizer', 'scheduler', 'epoch', 'loss', 'metrics']:
            optimized[key] = value
    
    # Save with highest compression
    torch.save(optimized, output_path, _use_new_zipfile_serialization=True)
    
    original_size = os.path.getsize(checkpoint_path) / (1024 * 1024)
    optimized_size = os.path.getsize(output_path) / (1024 * 1024)
    reduction = (1 - optimized_size / original_size) * 100
    
    print(f"✓ State dict optimized!")
    print(f"  Original: {original_size:.2f} MB")
    print(f"  Optimized: {optimized_size:.2f} MB")
    print(f"  Reduction: {reduction:.1f}%")
    
    return output_path


def create_torchscript(checkpoint_path, output_path, model_type='resnet'):
    """
    Convert model to TorchScript for faster loading and inference.
    Note: Requires example input for tracing.
    """
    print(f"Creating TorchScript model...")
    
    device = torch.device('cpu')
    
    # Import classes before loading (required for unpickling)
    if model_type == 'resnet':
        import resnet_train
        from resnet_train import EncoderCNN, DecoderRNN, Vocabulary
        
        # Make Vocabulary available in __main__ for unpickling
        import __main__
        if not hasattr(__main__, 'Vocabulary'):
            __main__.Vocabulary = Vocabulary
    elif model_type == 'efficientnet':
        from efficient_train import Encoder, Decoder, ImageCaptioningModel
        from transformers import AutoTokenizer
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if model_type == 'resnet':
        # Update vocab from checkpoint before creating DecoderRNN
        if 'vocab' in checkpoint and checkpoint['vocab'] is not None:
            resnet_train.vocab = checkpoint['vocab']
            print(f"  Updated vocab size: {len(checkpoint['vocab'].word2idx)}")
        else:
            raise ValueError("Checkpoint does not contain 'vocab' key. Cannot proceed.")
        
        encoder = EncoderCNN().eval()
        decoder = DecoderRNN().eval()  # Now uses the correct vocab size
        
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
        
        # Trace encoder
        dummy_image = torch.randn(1, 3, 224, 224)
        encoder_traced = torch.jit.trace(encoder, dummy_image)
        
        # For decoder, we need to trace with proper inputs
        # This is more complex due to RNN structure
        print("  ⚠ TorchScript for RNN decoder may require manual scripting")
        print("  ✓ Encoder traced successfully")
        
        torch.jit.save(encoder_traced, output_path.replace('.pth', '_encoder.pt'))
        
    elif model_type == 'efficientnet':
        # Classes already imported above
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        special_tokens = {'additional_special_tokens': ['<start>', '<end>']}
        tokenizer.add_special_tokens(special_tokens)
        
        encoder = Encoder(model_name='efficientnet_b3', embed_dim=512)
        decoder = Decoder(
            vocab_size=len(tokenizer),
            embed_dim=512,
            num_layers=8,
            num_heads=8,
            max_seq_length=64
        )
        model = ImageCaptioningModel(encoder, decoder).eval()
        
        model.load_state_dict(checkpoint['model_state'])
        
        # Trace encoder only (decoder has dynamic inputs)
        dummy_image = torch.randn(1, 3, 224, 224)
        encoder_traced = torch.jit.trace(model.encoder, dummy_image)
        
        torch.jit.save(encoder_traced, output_path.replace('.pth', '_encoder.pt'))
        print("  ✓ Encoder traced successfully")
    
    print(f"✓ TorchScript saved to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Optimize models for production deployment')
    parser.add_argument('--model', type=str, choices=['resnet', 'efficientnet', 'both'], 
                       default='both', help='Model to optimize')
    parser.add_argument('--method', type=str, choices=['quantize', 'optimize', 'torchscript', 'all'],
                       default='all', help='Optimization method')
    parser.add_argument('--resnet-path', type=str, default='resnet_best_model.pth',
                       help='Path to ResNet checkpoint')
    parser.add_argument('--efficientnet-path', type=str, default='efficient_best_model.pth',
                       help='Path to EfficientNet checkpoint')
    parser.add_argument('--output-dir', type=str, default='optimized_models',
                       help='Output directory for optimized models')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    models_to_process = []
    if args.model in ['resnet', 'both']:
        if os.path.exists(args.resnet_path):
            models_to_process.append(('resnet', args.resnet_path))
        else:
            print(f"⚠ Warning: {args.resnet_path} not found, skipping ResNet")
    
    if args.model in ['efficientnet', 'both']:
        if os.path.exists(args.efficientnet_path):
            models_to_process.append(('efficientnet', args.efficientnet_path))
        else:
            print(f"⚠ Warning: {args.efficientnet_path} not found, skipping EfficientNet")
    
    if not models_to_process:
        print("❌ No models found to optimize!")
        return
    
    for model_type, model_path in models_to_process:
        print(f"\n{'='*60}")
        print(f"Processing {model_type.upper()} model")
        print(f"{'='*60}")
        
        base_name = Path(model_path).stem
        output_base = os.path.join(args.output_dir, f"{model_type}_{base_name}")
        
        if args.method in ['quantize', 'all']:
            quantized_path = f"{output_base}_quantized.pth"
            quantize_model(model_path, quantized_path, model_type)
        
        if args.method in ['optimize', 'all']:
            optimized_path = f"{output_base}_optimized.pth"
            optimize_state_dict(model_path, optimized_path)
        
        if args.method in ['torchscript', 'all']:
            torchscript_path = f"{output_base}_torchscript.pt"
            create_torchscript(model_path, torchscript_path, model_type)
    
    print(f"\n{'='*60}")
    print("✓ Optimization complete!")
    print(f"Optimized models saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

