"""Layer 1 diagnostic plots: Data & Feature Inspection.

This module validates:
- Features aren't flat or all super-correlated
- Label isn't 90% one class
- No insane values or major data quality issues
"""

from typing import Optional, List, Dict
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_feature_timeseries(
    features_df: pd.DataFrame,
    features_to_plot: Optional[List[str]] = None,
    figsize: tuple = (14, 10),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot time series of selected features to check for drift.
    
    Args:
        features_df: DataFrame with features
        features_to_plot: List of feature names to plot (default: all)
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Figure object
    """
    if features_to_plot is None:
        features_to_plot = features_df.columns.tolist()
    
    n_features = len(features_to_plot)
    n_cols = 3
    n_rows = int(np.ceil(n_features / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for i, feature in enumerate(features_to_plot):
        ax = axes[i]
        features_df[feature].plot(ax=ax, linewidth=0.5, alpha=0.7)
        ax.set_title(f'{feature}', fontsize=10)
        ax.set_xlabel('')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Saved feature timeseries to {save_path}")
    
    return fig


def plot_feature_distributions(
    features_df: pd.DataFrame,
    features_to_plot: Optional[List[str]] = None,
    figsize: tuple = (14, 10),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot histograms to check for insane values or extreme skew.
    
    Args:
        features_df: DataFrame with features
        features_to_plot: List of feature names to plot (default: all)
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Figure object
    """
    if features_to_plot is None:
        features_to_plot = features_df.columns.tolist()
    
    n_features = len(features_to_plot)
    n_cols = 3
    n_rows = int(np.ceil(n_features / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for i, feature in enumerate(features_to_plot):
        ax = axes[i]
        
        # Remove infinite and NaN values
        data = features_df[feature].replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(data) > 0:
            ax.hist(data, bins=50, alpha=0.7, edgecolor='black', linewidth=0.5)
            ax.set_title(f'{feature}\nμ={data.mean():.3f}, σ={data.std():.3f}', fontsize=9)
            ax.set_xlabel('')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
    
    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Saved feature distributions to {save_path}")
    
    return fig


def plot_correlation_matrix(
    features_df: pd.DataFrame,
    target: Optional[pd.Series] = None,
    figsize: tuple = (12, 10),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot correlation matrix to check for redundant features.
    
    Args:
        features_df: DataFrame with features
        target: Optional target series to include
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Figure object
    """
    # Combine features and target if provided
    if target is not None:
        df_combined = features_df.copy()
        df_combined['target'] = target
    else:
        df_combined = features_df
    
    # Calculate correlation
    corr = df_combined.corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(
        corr,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={'shrink': 0.8},
        ax=ax,
    )
    
    ax.set_title('Feature Correlation Matrix', fontsize=14, pad=20)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Saved correlation matrix to {save_path}")
    
    return fig


def plot_label_balance(
    target: pd.Series,
    split_by_date: bool = False,
    window: Optional[str] = '30D',
    figsize: tuple = (12, 6),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot label distribution to check for class imbalance.
    
    Args:
        target: Target series
        split_by_date: Whether to show balance over time
        window: Rolling window for time-based plot (e.g., '30D')
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Figure object
    """
    if split_by_date:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
        ax2 = None
    
    # Overall distribution
    counts = target.value_counts().sort_index()
    percentages = (counts / counts.sum() * 100).round(2)
    
    bars = ax1.bar(counts.index, counts.values, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Label')
    ax1.set_ylabel('Count')
    ax1.set_title('Overall Label Distribution')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels on bars
    for bar, pct in zip(bars, percentages.values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Time-based balance
    if split_by_date and ax2 is not None:
        # Rolling balance
        rolling_mean = target.rolling(window=window).mean()
        
        ax2.plot(rolling_mean.index, rolling_mean.values, linewidth=1.5, alpha=0.8)
        ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% baseline')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Fraction of Up Labels')
        ax2.set_title(f'Label Balance Over Time (rolling {window})')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Saved label balance to {save_path}")
    
    return fig


def plot_missing_data(
    df: pd.DataFrame,
    figsize: tuple = (12, 6),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Visualize missing data patterns.
    
    Args:
        df: DataFrame to check
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Figure object
    """
    # Calculate missing data statistics
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df) * 100).sort_values(ascending=False)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Bar chart of missing percentages
    if missing_pct.sum() > 0:
        missing_pct_nonzero = missing_pct[missing_pct > 0]
        if len(missing_pct_nonzero) > 0:
            missing_pct_nonzero.plot(kind='barh', ax=ax1, alpha=0.7, edgecolor='black')
            ax1.set_xlabel('Missing %')
            ax1.set_title('Missing Data by Feature')
            ax1.grid(True, alpha=0.3, axis='x')
        else:
            ax1.text(0.5, 0.5, 'No missing data', ha='center', va='center', transform=ax1.transAxes)
    else:
        ax1.text(0.5, 0.5, 'No missing data', ha='center', va='center', transform=ax1.transAxes)
    
    # Heatmap of missing data over time (sample if too large)
    if len(df) > 1000:
        # Sample every nth row to keep visualization manageable
        sample_stride = len(df) // 1000
        df_sample = df.iloc[::sample_stride]
    else:
        df_sample = df
    
    # Create binary mask for missing data
    missing_mask = df_sample.isnull().T
    
    if missing_mask.sum().sum() > 0:
        sns.heatmap(missing_mask, cmap='YlOrRd', cbar=True, ax=ax2, yticklabels=True)
        ax2.set_title('Missing Data Pattern Over Time')
        ax2.set_xlabel('Time (sampled)')
    else:
        ax2.text(0.5, 0.5, 'No missing data', ha='center', va='center', transform=ax2.transAxes)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Saved missing data plot to {save_path}")
    
    return fig


def generate_layer1_report(
    features_df: pd.DataFrame,
    target: pd.Series,
    output_dir: Path,
    prefix: str = "layer1",
) -> Dict[str, Path]:
    """
    Generate complete Layer 1 diagnostic report.
    
    Args:
        features_df: DataFrame with features
        target: Target series
        output_dir: Directory to save plots
        prefix: Prefix for filenames
        
    Returns:
        Dictionary mapping plot names to file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("LAYER 1 DIAGNOSTIC REPORT: Data & Feature Inspection")
    print("="*60)
    
    # Summary statistics
    print(f"\nData shape: {features_df.shape}")
    print(f"Date range: {features_df.index[0]} to {features_df.index[-1]}")
    print(f"Total samples: {len(features_df)}")
    
    # Feature statistics
    print("\nFeature Summary:")
    print(features_df.describe().T[['mean', 'std', 'min', 'max']])
    
    # Label balance
    print("\nLabel Balance:")
    target_clean = target.dropna()
    counts = target_clean.value_counts().sort_index()
    for label, count in counts.items():
        pct = count / len(target_clean) * 100
        print(f"  Class {label}: {count} ({pct:.2f}%)")
    
    # Missing data
    missing = features_df.isnull().sum()
    if missing.sum() > 0:
        print("\nMissing Data:")
        for col, count in missing[missing > 0].items():
            pct = count / len(features_df) * 100
            print(f"  {col}: {count} ({pct:.2f}%)")
    else:
        print("\nNo missing data detected ✓")
    
    # Generate plots
    print("\nGenerating diagnostic plots...")
    
    saved_paths = {}
    
    # 1. Feature timeseries
    path = output_dir / f"{prefix}_feature_timeseries.png"
    plot_feature_timeseries(features_df, save_path=path)
    saved_paths['timeseries'] = path
    plt.close()
    
    # 2. Feature distributions
    path = output_dir / f"{prefix}_feature_distributions.png"
    plot_feature_distributions(features_df, save_path=path)
    saved_paths['distributions'] = path
    plt.close()
    
    # 3. Correlation matrix
    path = output_dir / f"{prefix}_correlation_matrix.png"
    plot_correlation_matrix(features_df, target, save_path=path)
    saved_paths['correlation'] = path
    plt.close()
    
    # 4. Label balance
    path = output_dir / f"{prefix}_label_balance.png"
    plot_label_balance(target, split_by_date=True, save_path=path)
    saved_paths['label_balance'] = path
    plt.close()
    
    # 5. Missing data
    path = output_dir / f"{prefix}_missing_data.png"
    combined_df = features_df.copy()
    combined_df['target'] = target
    plot_missing_data(combined_df, save_path=path)
    saved_paths['missing_data'] = path
    plt.close()
    
    print("\n" + "="*60)
    print("Layer 1 diagnostics complete!")
    print(f"Plots saved to: {output_dir}")
    print("="*60 + "\n")
    
    return saved_paths

