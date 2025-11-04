"""Training utilities for TCN model."""

from typing import Dict, Optional, Tuple
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
import json


class EarlyStopping:
    """Early stopping to stop training when validation metric stops improving."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = 'max'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' to maximize metric, 'min' to minimize
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation metric
            
        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def compute_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute binary classification metrics.
    
    Args:
        y_true: True labels (0 or 1)
        y_pred_proba: Predicted probabilities
        threshold: Classification threshold
        
    Returns:
        Dictionary of metrics
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.0,
    }
    
    return metrics


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        
    Returns:
        Tuple of (average loss, metrics dict)
    """
    model.train()
    
    total_loss = 0.0
    all_targets = []
    all_preds = []
    
    for features, targets, _ in dataloader:
        # Move to device
        features = features.to(device)
        targets = targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        
        # Convert to probabilities
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.extend(probs.flatten())
        all_targets.extend(targets.cpu().numpy().flatten())
    
    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(np.array(all_targets), np.array(all_preds))
    
    return avg_loss, metrics


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """
    Validate model.
    
    Args:
        model: Model to validate
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to validate on
        
    Returns:
        Tuple of (average loss, metrics dict)
    """
    model.eval()
    
    total_loss = 0.0
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        for features, targets, _ in dataloader:
            # Move to device
            features = features.to(device)
            targets = targets.to(device)
            
            # Forward pass
            logits = model(features)
            loss = criterion(logits, targets)
            
            # Track metrics
            total_loss += loss.item()
            
            # Convert to probabilities
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs.flatten())
            all_targets.extend(targets.cpu().numpy().flatten())
    
    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(np.array(all_targets), np.array(all_preds))
    
    return avg_loss, metrics


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    checkpoint_dir: Path,
    early_stopping_patience: int = 5,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> Dict[str, list]:
    """
    Full training loop with validation and checkpointing.
    
    Args:
        model: Model to train
        train_loader: Training dataloader
        val_loader: Validation dataloader
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of epochs to train
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
        early_stopping_patience: Patience for early stopping
        scheduler: Optional learning rate scheduler
        
    Returns:
        Dictionary with training history
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=early_stopping_patience, mode='max')
    
    # Track history
    history = {
        'train_loss': [],
        'train_acc': [],
        'train_auc': [],
        'val_loss': [],
        'val_acc': [],
        'val_auc': [],
        'lr': [],
    }
    
    best_val_auc = 0.0
    
    print("\n" + "="*70)
    print("TRAINING START")
    print("="*70)
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device
        )
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_metrics['accuracy'])
        history['train_auc'].append(train_metrics['auc'])
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_auc'].append(val_metrics['auc'])
        
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
        
        # Print progress
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f}, AUC: {train_metrics['auc']:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}, AUC: {val_metrics['auc']:.4f}")
        print(f"  LR: {current_lr:.6f}")
        
        # Save best model
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_metrics['auc'],
                'val_loss': val_loss,
            }, checkpoint_dir / 'best_model.pt')
            print(f"  ✓ Saved new best model (AUC: {best_val_auc:.4f})")
        
        # Learning rate scheduling
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['auc'])
            else:
                scheduler.step()
        
        # Early stopping check
        if early_stopping(val_metrics['auc']):
            print(f"\n  Early stopping triggered after {epoch+1} epochs")
            break
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print(f"Best validation AUC: {best_val_auc:.4f}")
    print("="*70 + "\n")
    
    # Save final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
    }, checkpoint_dir / 'final_model.pt')
    
    # Save history as JSON
    with open(checkpoint_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    return history


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> Dict:
    """
    Load model from checkpoint.
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded checkpoint from {checkpoint_path}")
    if 'val_auc' in checkpoint:
        print(f"  Validation AUC: {checkpoint['val_auc']:.4f}")
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    
    return checkpoint

