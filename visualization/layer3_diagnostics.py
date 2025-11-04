"""Layer 3 diagnostic plots: Model Training.

This module validates:
- Train loss goes down
- Val loss doesn't immediately diverge
- Predictions aren't degenerate
- Model is learning signal
"""

from typing import Dict, List, Optional
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_training_curves(
    history: Dict[str, List[float]],
    figsize: tuple = (14, 10),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot training and validation curves.
    
    Args:
        history: Training history dictionary
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curves
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], label='Train', linewidth=2)
    ax.plot(epochs, history['val_loss'], label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax = axes[0, 1]
    ax.plot(epochs, history['train_acc'], label='Train', linewidth=2)
    ax.plot(epochs, history['val_acc'], label='Val', linewidth=2)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.3, label='Random')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # AUC curves
    ax = axes[1, 0]
    ax.plot(epochs, history['train_auc'], label='Train', linewidth=2)
    ax.plot(epochs, history['val_auc'], label='Val', linewidth=2)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.3, label='Random')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('AUC')
    ax.set_title('AUC Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Learning rate
    ax = axes[1, 1]
    ax.plot(epochs, history['lr'], linewidth=2, color='green')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    
    return fig


def plot_calibration(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10,
    figsize: tuple = (8, 6),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot calibration curve to check if predicted probabilities match reality.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        n_bins: Number of bins for calibration
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Bin predictions
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred_proba, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    # Calculate empirical frequency in each bin
    bin_means = []
    bin_freqs = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_means.append(y_pred_proba[mask].mean())
            bin_freqs.append(y_true[mask].mean())
            bin_counts.append(mask.sum())
        else:
            bin_means.append((bins[i] + bins[i+1]) / 2)
            bin_freqs.append(np.nan)
            bin_counts.append(0)
    
    # Plot
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2, alpha=0.5)
    
    # Plot bins with size proportional to count
    sizes = np.array(bin_counts) / np.sum(bin_counts) * 1000
    scatter = ax.scatter(bin_means, bin_freqs, s=sizes, alpha=0.6, edgecolors='black')
    
    # Connect with line
    valid_mask = ~np.isnan(bin_freqs)
    if valid_mask.sum() > 0:
        ax.plot(np.array(bin_means)[valid_mask], np.array(bin_freqs)[valid_mask], 
                'b-', linewidth=2, alpha=0.7, label='Model calibration')
    
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Empirical Frequency')
    ax.set_title('Calibration Plot\n(bubble size = sample count)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Saved calibration plot to {save_path}")
    
    return fig


def plot_prediction_distribution(
    y_pred_proba: np.ndarray,
    figsize: tuple = (10, 5),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot distribution of predicted probabilities.
    
    Check if model is predicting a variety of values or stuck at 0.5.
    
    Args:
        y_pred_proba: Predicted probabilities
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram
    ax1.hist(y_pred_proba, bins=50, alpha=0.7, edgecolor='black')
    ax1.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Neutral')
    ax1.set_xlabel('Predicted Probability')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of Predictions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # CDF
    sorted_probs = np.sort(y_pred_proba)
    cumulative = np.arange(1, len(sorted_probs) + 1) / len(sorted_probs)
    ax2.plot(sorted_probs, cumulative, linewidth=2)
    ax2.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Neutral')
    ax2.set_xlabel('Predicted Probability')
    ax2.set_ylabel('Cumulative Fraction')
    ax2.set_title('Cumulative Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add statistics
    stats_text = f"Mean: {y_pred_proba.mean():.3f}\nStd: {y_pred_proba.std():.3f}\nMin: {y_pred_proba.min():.3f}\nMax: {y_pred_proba.max():.3f}"
    ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Saved prediction distribution to {save_path}")
    
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
    figsize: tuple = (6, 5),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        threshold: Classification threshold
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Figure object
    """
    from sklearn.metrics import confusion_matrix
    
    y_pred = (y_pred_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Down', 'Up'],
                yticklabels=['Down', 'Up'])
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix (threshold={threshold:.2f})')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    
    return fig


def generate_layer3_report(
    history: Dict[str, List[float]],
    val_y_true: np.ndarray,
    val_y_pred_proba: np.ndarray,
    output_dir: Path,
    prefix: str = "layer3",
) -> Dict[str, Path]:
    """
    Generate complete Layer 3 diagnostic report.
    
    Args:
        history: Training history
        val_y_true: Validation true labels
        val_y_pred_proba: Validation predicted probabilities
        output_dir: Directory to save plots
        prefix: Prefix for filenames
        
    Returns:
        Dictionary mapping plot names to file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("LAYER 3 DIAGNOSTIC REPORT: Model Training")
    print("="*60)
    
    # Summary statistics
    print(f"\nTraining Summary:")
    print(f"  Epochs trained: {len(history['train_loss'])}")
    print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"  Final val loss: {history['val_loss'][-1]:.4f}")
    print(f"  Best val AUC: {max(history['val_auc']):.4f}")
    print(f"  Final val AUC: {history['val_auc'][-1]:.4f}")
    
    # Prediction statistics
    print(f"\nValidation Predictions:")
    print(f"  Mean probability: {val_y_pred_proba.mean():.3f}")
    print(f"  Std probability: {val_y_pred_proba.std():.3f}")
    print(f"  Range: [{val_y_pred_proba.min():.3f}, {val_y_pred_proba.max():.3f}]")
    
    # Check for degeneracy
    if val_y_pred_proba.std() < 0.01:
        print("\n  ⚠️  WARNING: Predictions have very low variance (model may be stuck)")
    
    # Generate plots
    print("\nGenerating diagnostic plots...")
    
    saved_paths = {}
    
    # 1. Training curves
    path = output_dir / f"{prefix}_training_curves.png"
    plot_training_curves(history, save_path=path)
    saved_paths['training_curves'] = path
    plt.close()
    
    # 2. Calibration plot
    path = output_dir / f"{prefix}_calibration.png"
    plot_calibration(val_y_true, val_y_pred_proba, save_path=path)
    saved_paths['calibration'] = path
    plt.close()
    
    # 3. Prediction distribution
    path = output_dir / f"{prefix}_prediction_distribution.png"
    plot_prediction_distribution(val_y_pred_proba, save_path=path)
    saved_paths['prediction_dist'] = path
    plt.close()
    
    # 4. Confusion matrix
    path = output_dir / f"{prefix}_confusion_matrix.png"
    plot_confusion_matrix(val_y_true, val_y_pred_proba, save_path=path)
    saved_paths['confusion'] = path
    plt.close()
    
    print("\n" + "="*60)
    print("Layer 3 diagnostics complete!")
    print(f"Plots saved to: {output_dir}")
    print("="*60 + "\n")
    
    return saved_paths

