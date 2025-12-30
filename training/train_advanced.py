"""
Advanced Training Script with Best Practices
- Learning rate scheduling
- Mixed precision training
- Experiment tracking (W&B optional)
- Comprehensive evaluation
- Model checkpointing
"""

import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, LambdaLR
import math
from efficient_train import (
    create_dataloaders, Encoder, Decoder, ImageCaptioningModel,
    train_epoch, validate, generate_caption
)
from datetime import datetime

# Optional: Weights & Biases
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("W&B not available. Install with: pip install wandb")


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Create learning rate schedule with warmup and cosine annealing"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def train_advanced(args):
    """Advanced training with all best practices"""
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # GPU optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
        torch.backends.cudnn.deterministic = False  # Faster, but non-deterministic
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Initialize W&B
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project,
            name=f"{args.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=vars(args)
        )
    
    # Create dataloaders
    train_loader, val_loader, test_loader, tokenizer, train_set = create_dataloaders(args)
    
    # Initialize model
    encoder = Encoder(args.model_name, args.embed_dim)
    decoder = Decoder(
        vocab_size=tokenizer.vocab_size + 2,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_length=64,
        dropout=args.dropout
    )
    model = ImageCaptioningModel(encoder, decoder).to(device)
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_val_loss = float('inf')
    best_metrics = {}
    
    if args.resume_checkpoint:
        print(f"Loading checkpoint from {args.resume_checkpoint}")
        # Handle PyTorch 2.6+ security: allow tokenizer classes
        try:
            from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
            torch.serialization.add_safe_globals([GPT2TokenizerFast])
        except ImportError:
            pass
        
        checkpoint = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")
    
    # Optimizer with different learning rates for encoder/decoder
    encoder_params = [p for n, p in model.named_parameters() if 'encoder' in n]
    decoder_params = [p for n, p in model.named_parameters() if 'decoder' in n]
    
    if args.different_lr:
        # Lower learning rate for encoder (fine-tuning)
        optimizer = optim.AdamW([
            {'params': encoder_params, 'lr': args.lr * 0.1},
            {'params': decoder_params, 'lr': args.lr}
        ], weight_decay=args.weight_decay)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=args.epochs * len(train_loader),
            eta_min=args.min_lr
        )
    elif args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=args.patience
        )
    elif args.scheduler == 'warmup_cosine':
        num_training_steps = args.epochs * len(train_loader)
        num_warmup_steps = args.warmup_epochs * len(train_loader)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )
    else:
        scheduler = None
    
    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    # Mixed precision training - Use new API for PyTorch 2.6+
    if hasattr(torch.amp, 'GradScaler'):
        scaler = torch.amp.GradScaler('cuda', enabled=args.use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Training loop
    patience_counter = 0
    
    for epoch in range(start_epoch, args.epochs):
        args.epoch = epoch  # Set epoch for train_epoch function
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 60)
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, scaler, 
            scheduler if args.scheduler == 'cosine' or args.scheduler == 'warmup_cosine' else None,
            device, args
        )
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        if args.scheduler == 'plateau':
            scheduler.step(val_loss)
        elif args.scheduler in ['cosine', 'warmup_cosine']:
            # Already updated in train_epoch
            pass
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")
        
        # Log to W&B
        log_dict = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': current_lr
        }
        
        if args.use_wandb and WANDB_AVAILABLE:
            wandb.log(log_dict)
        
        # Checkpointing
        is_best = val_loss < best_val_loss
        
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            checkpoint = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict() if scheduler else None,
                'val_loss': val_loss,
                'train_loss': train_loss,
                'tokenizer': tokenizer,
                'config': vars(args)
            }
            
            best_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model (val_loss: {val_loss:.4f})")
            
        else:
            patience_counter += 1
            
            # Save periodic checkpoints
            if (epoch + 1) % args.save_every == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict() if scheduler else None,
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                    'tokenizer': tokenizer,
                    'config': vars(args)
                }
                checkpoint_path = os.path.join(
                    args.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'
                )
                torch.save(checkpoint, checkpoint_path)
                print(f"✓ Saved periodic checkpoint (epoch {epoch+1})")
        
        # Early stopping
        if patience_counter >= args.early_stopping_patience:
            print(f"\nEarly stopping triggered after {args.early_stopping_patience} epochs without improvement")
            break
    
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {os.path.join(args.checkpoint_dir, 'best_model.pth')}")
    print("="*60)
    
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='Advanced training with best practices')
    
    # Data arguments
    parser.add_argument('--train_image_dir', type=str, required=True)
    parser.add_argument('--train_ann_file', type=str, required=True)
    parser.add_argument('--val_image_dir', type=str, required=True)
    parser.add_argument('--val_ann_file', type=str, required=True)
    parser.add_argument('--test_image_dir', type=str, required=True)
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='efficientnet_b3')
    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=96)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_amp', action='store_true', help='Use mixed precision')
    parser.add_argument('--grad_accum', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--different_lr', action='store_true', 
                       help='Use different LR for encoder/decoder')
    
    # Scheduler arguments
    parser.add_argument('--scheduler', type=str, default='plateau',
                       choices=['cosine', 'plateau', 'warmup_cosine', 'none'])
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_epochs', type=int, default=2)
    
    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--resume_checkpoint', type=str, default=None)
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--early_stopping_patience', type=int, default=5)
    
    # Experiment tracking
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases')
    parser.add_argument('--wandb_project', type=str, default='image-captioning')
    
    # Additional args needed by create_dataloaders and train_epoch
    parser.add_argument('--distributed', action='store_true', help='Use distributed training')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    
    args = parser.parse_args()
    
    # Set epoch attribute (will be updated during training)
    args.epoch = 0
    
    train_advanced(args)


if __name__ == '__main__':
    main()

