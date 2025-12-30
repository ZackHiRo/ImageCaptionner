"""
Hyperparameter Optimization using Optuna
Run this to find the best hyperparameters for your model
"""

import optuna
import torch
import argparse
import os
import sys
from efficient_train import create_dataloaders, Encoder, Decoder, ImageCaptioningModel
from efficient_train import train_epoch, validate, generate_caption
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

def train_with_config(trial, args):
    """Train model with suggested hyperparameters from Optuna"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Suggest hyperparameters
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 96, 128])
    embed_dim = trial.suggest_categorical('embed_dim', [256, 512, 768])
    num_layers = trial.suggest_int('num_layers', 4, 12)
    num_heads = trial.suggest_categorical('num_heads', [4, 8, 12, 16])
    dropout = trial.suggest_uniform('dropout', 0.1, 0.5)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-2)
    warmup_epochs = trial.suggest_int('warmup_epochs', 0, 3)
    
    # Update args with suggested values
    args.lr = lr
    args.batch_size = batch_size
    args.embed_dim = embed_dim
    args.num_layers = num_layers
    args.num_heads = num_heads
    args.epochs = 5  # Fewer epochs for hyperparameter search
    
    # Create dataloaders
    train_loader, val_loader, test_loader, tokenizer, train_set = create_dataloaders(args)
    
    # Initialize model
    encoder = Encoder(args.model_name, embed_dim)
    decoder = Decoder(
        vocab_size=tokenizer.vocab_size + 2,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        max_seq_length=64,
        dropout=dropout
    )
    model = ImageCaptioningModel(encoder, decoder).to(device)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # Loss
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    # Mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    
    # Training loop (fewer epochs for hyperparameter search)
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler, 
                                scheduler, device, args)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Report to Optuna
        trial.report(val_loss, epoch)
        
        # Prune trial if not promising
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    
    return best_val_loss


def objective(trial):
    """Optuna objective function"""
    
    # Create minimal args object
    args = argparse.Namespace(
        train_image_dir='Data/train2017/train2017',
        train_ann_file='Data/annotations_trainval2017/annotations/captions_train2017.json',
        val_image_dir='Data/val2017',
        val_ann_file='Data/annotations_trainval2017/annotations/captions_val2017.json',
        test_image_dir='Data/test2017/test2017',
        model_name='efficientnet_b3',
        embed_dim=512,  # Will be overridden
        num_layers=8,   # Will be overridden
        num_heads=8,    # Will be overridden
        batch_size=96,  # Will be overridden
        lr=3e-4,        # Will be overridden
        epochs=5,
        seed=42,
        use_amp=True,
        grad_accum=1,
        checkpoint_dir='checkpoints',
        early_stopping_patience=3,
        distributed=False,
        local_rank=0,
        resume_checkpoint=None
    )
    
    try:
        val_loss = train_with_config(trial, args)
        return val_loss
    except Exception as e:
        print(f"Trial failed: {e}")
        return float('inf')


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter optimization with Optuna')
    parser.add_argument('--n_trials', type=int, default=50, help='Number of trials')
    parser.add_argument('--timeout', type=int, default=3600*24, help='Timeout in seconds')
    parser.add_argument('--study_name', type=str, default='efficientnet_captioning', 
                       help='Study name')
    parser.add_argument('--storage', type=str, default='sqlite:///optuna_study.db',
                       help='Storage URL for study')
    
    args = parser.parse_args()
    
    # Create or load study
    study = optuna.create_study(
        direction='minimize',
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    )
    
    print(f"Starting optimization with {args.n_trials} trials...")
    print(f"Study: {args.study_name}")
    
    # Optimize
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)
    
    # Print results
    print("\n" + "="*60)
    print("Optimization Complete!")
    print("="*60)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best validation loss: {study.best_value:.4f}")
    print("\nBest parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save results
    import json
    with open('best_hyperparameters.json', 'w') as f:
        json.dump(study.best_params, f, indent=2)
    
    print("\nBest hyperparameters saved to best_hyperparameters.json")
    
    # Visualize (optional, requires plotly)
    try:
        import optuna.visualization as vis
        
        # Optimization history
        fig = vis.plot_optimization_history(study)
        fig.write_image("optimization_history.png")
        print("Saved optimization_history.png")
        
        # Parameter importances
        fig = vis.plot_param_importances(study)
        fig.write_image("param_importances.png")
        print("Saved param_importances.png")
        
        # Parallel coordinate plot
        fig = vis.plot_parallel_coordinate(study)
        fig.write_image("parallel_coordinate.png")
        print("Saved parallel_coordinate.png")
        
    except ImportError:
        print("Install plotly to generate visualizations: pip install plotly")


if __name__ == '__main__':
    main()

