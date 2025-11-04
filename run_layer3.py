#!/usr/bin/env python3
"""
Layer 3: Model Training

This script trains the TCN model and validates:
- Train loss goes down
- Val loss doesn't immediately diverge
- Predictions aren't degenerate (spread of 0.45–0.55+)

Success criteria:
- Model learns signal (val AUC > 0.5)
- No severe overfitting (val loss doesn't explode)
- Predictions have reasonable variance
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn

# Project imports
from utils import load_config, get_paths, set_seed
from models import (
    TCNDataset,
    create_dataloaders,
    load_processed_data_for_training,
    create_tcn_from_config,
)
from models.train import train_model, validate, load_checkpoint
from visualization.layer3_diagnostics import generate_layer3_report


def main():
    """Run Layer 3: Model Training."""
    
    print("\n" + "="*70)
    print("LAYER 3: MODEL TRAINING")
    print("="*70 + "\n")
    
    # 1. Load configuration
    print("[1/7] Loading configuration...")
    config = load_config()
    paths = get_paths()
    set_seed(config['experiment']['seed'])
    
    symbol = config['data']['symbol']
    frequency = config['data']['frequency']
    device = torch.device(config['experiment']['device'])
    
    print(f"  Symbol: {symbol}")
    print(f"  Frequency: {frequency}")
    print(f"  Device: {device}")
    
    # 2. Load datasets
    print("\n[2/7] Loading datasets...")
    
    train_dataset, val_dataset, test_dataset = load_processed_data_for_training(
        processed_data_dir=paths['processed_data'],
        symbol=symbol,
        frequency=frequency,
        seq_len=config['data']['seq_len'],
    )
    
    # 3. Create dataloaders
    print("\n[3/7] Creating dataloaders...")
    
    dataloaders = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=config['training']['batch_size'],
        num_workers=0,
        shuffle_train=False,  # Keep temporal order for time series
    )
    
    # 4. Create model
    print("\n[4/7] Creating TCN model...")
    
    model = create_tcn_from_config(config)
    model = model.to(device)
    
    # 5. Setup training
    print("\n[5/7] Setting up training...")
    
    # Loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
    )
    
    # Learning rate scheduler
    scheduler = None
    if config['training']['use_scheduler']:
        if config['training']['scheduler_type'] == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=config['training']['scheduler_factor'],
                patience=config['training']['scheduler_patience'],
            )
            print(f"  Using ReduceLROnPlateau scheduler")
        elif config['training']['scheduler_type'] == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config['training']['num_epochs'],
            )
    
    print(f"  Loss: BCEWithLogitsLoss")
    print(f"  Optimizer: Adam (lr={config['training']['learning_rate']})")
    print(f"  Scheduler: {config['training']['scheduler_type'] if scheduler else 'None'}")
    print(f"  Early stopping patience: {config['training']['early_stopping_patience']}")
    
    # 6. Train model
    print("\n[6/7] Training model...")
    
    history = train_model(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=config['training']['num_epochs'],
        device=device,
        checkpoint_dir=paths['checkpoints'],
        early_stopping_patience=config['training']['early_stopping_patience'],
        scheduler=scheduler,
    )
    
    # 7. Generate diagnostics
    print("\n[7/7] Generating diagnostic report...")
    
    # Load best model
    load_checkpoint(model, paths['checkpoints'] / 'best_model.pt', device)
    
    # Get validation predictions
    model.eval()
    val_y_true = []
    val_y_pred = []
    
    with torch.no_grad():
        for features, targets, _ in dataloaders['val']:
            features = features.to(device)
            logits = model(features)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            val_y_true.extend(targets.numpy().flatten())
            val_y_pred.extend(probs.flatten())
    
    val_y_true = np.array(val_y_true)
    val_y_pred = np.array(val_y_pred)
    
    # Generate report
    generate_layer3_report(
        history=history,
        val_y_true=val_y_true,
        val_y_pred_proba=val_y_pred,
        output_dir=paths['figures'],
        prefix='layer3',
    )
    
    # Final summary
    print("\n" + "="*70)
    print("LAYER 3 COMPLETE ✓")
    print("="*70)
    print("\nModel Performance:")
    print(f"  Best val AUC: {max(history['val_auc']):.4f}")
    print(f"  Final val AUC: {history['val_auc'][-1]:.4f}")
    print(f"  Final val accuracy: {history['val_acc'][-1]:.4f}")
    
    print("\nPrediction Quality:")
    print(f"  Mean probability: {val_y_pred.mean():.3f}")
    print(f"  Std probability: {val_y_pred.std():.3f}")
    print(f"  Range: [{val_y_pred.min():.3f}, {val_y_pred.max():.3f}]")
    
    # Success criteria
    print("\nSuccess Criteria Check:")
    success = True
    
    if max(history['val_auc']) > 0.5:
        print("  ✓ Model learns signal (AUC > 0.5)")
    else:
        print("  ✗ Model not learning (AUC ≤ 0.5)")
        success = False
    
    if val_y_pred.std() > 0.01:
        print("  ✓ Predictions have variance")
    else:
        print("  ✗ Predictions are degenerate")
        success = False
    
    overfit_ratio = history['train_auc'][-1] / history['val_auc'][-1]
    if overfit_ratio < 1.2:
        print(f"  ✓ No severe overfitting (train/val ratio: {overfit_ratio:.2f})")
    else:
        print(f"  ⚠️  Possible overfitting (train/val ratio: {overfit_ratio:.2f})")
    
    print("\nNext steps:")
    if success:
        print("  1. Review diagnostic plots in:", paths['figures'])
        print("  2. Check calibration plot - are predictions well-calibrated?")
        print("  3. If everything looks good, proceed to Layer 4 (out-of-sample inference)")
    else:
        print("  1. Review training curves")
        print("  2. Consider:")
        print("     - Increasing model capacity (more channels)")
        print("     - Increasing sequence length")
        print("     - Adding more features")
        print("     - Reducing dropout")
        print("     - Training for more epochs")
        print("  3. Re-run this script after adjustments")
    
    print("\nModel saved to:", paths['checkpoints'])
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

