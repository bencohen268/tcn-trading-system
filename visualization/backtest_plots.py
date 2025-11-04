"""Backtest visualization utilities."""

from typing import Optional
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_equity_curve(
    equity: pd.Series,
    figsize: tuple = (12, 6),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot equity curve.
    
    Args:
        equity: Equity series
        figsize: Figure size
        save_path: Optional path to save
        
    Returns:
        Figure object
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Equity curve
    ax1.plot(equity.index, equity.values, linewidth=1.5, alpha=0.8)
    ax1.set_ylabel('Equity ($)')
    ax1.set_title('Equity Curve')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=equity.iloc[0], color='gray', linestyle='--', alpha=0.5, label='Initial')
    ax1.legend()
    
    # Drawdown
    cummax = equity.expanding().max()
    drawdown = (equity - cummax) / cummax
    
    ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
    ax2.plot(drawdown.index, drawdown.values, linewidth=1, color='darkred', alpha=0.7)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Drawdown')
    ax2.set_title('Drawdown')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([drawdown.min() * 1.1, 0.05])
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Saved equity curve to {save_path}")
    
    return fig


def plot_positions_vs_price(
    prices: pd.Series,
    positions: pd.Series,
    figsize: tuple = (12, 8),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot positions overlaid with price.
    
    Args:
        prices: Price series
        positions: Position series
        figsize: Figure size
        save_path: Optional path to save
        
    Returns:
        Figure object
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True, 
                                    gridspec_kw={'height_ratios': [2, 1]})
    
    # Price
    ax1.plot(prices.index, prices.values, linewidth=1, alpha=0.8, color='black')
    ax1.set_ylabel('Price')
    ax1.set_title('Price and Position Over Time')
    ax1.grid(True, alpha=0.3)
    
    # Positions
    # Color code by direction
    long_mask = positions > 0
    short_mask = positions < 0
    flat_mask = positions == 0
    
    ax2.fill_between(positions.index, 0, positions.values, 
                     where=positions > 0, alpha=0.3, color='green', label='Long')
    ax2.fill_between(positions.index, 0, positions.values,
                     where=positions < 0, alpha=0.3, color='red', label='Short')
    ax2.plot(positions.index, positions.values, linewidth=0.8, alpha=0.7, color='black')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Position')
    ax2.set_title('Position Size')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Saved positions vs price to {save_path}")
    
    return fig


def plot_returns_distribution(
    returns: pd.Series,
    figsize: tuple = (12, 5),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot distribution of returns.
    
    Args:
        returns: Returns series
        figsize: Figure size
        save_path: Optional path to save
        
    Returns:
        Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Remove zero returns (non-trading days)
    returns_clean = returns[returns != 0]
    
    # Histogram
    ax1.hist(returns_clean, bins=50, alpha=0.7, edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Return')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Return Distribution')
    ax1.grid(True, alpha=0.3)
    
    # Q-Q plot (check normality)
    from scipy import stats
    stats.probplot(returns_clean, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot (Normality Check)')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics
    stats_text = (f"Mean: {returns_clean.mean():.4f}\n"
                 f"Std: {returns_clean.std():.4f}\n"
                 f"Skew: {returns_clean.skew():.2f}\n"
                 f"Kurt: {returns_clean.kurtosis():.2f}")
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Saved returns distribution to {save_path}")
    
    return fig


def plot_monthly_returns(
    returns: pd.Series,
    figsize: tuple = (12, 6),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot monthly returns heatmap.
    
    Args:
        returns: Returns series
        figsize: Figure size
        save_path: Optional path to save
        
    Returns:
        Figure object
    """
    # Calculate cumulative returns
    equity = (1 + returns).cumprod()
    
    # Resample to monthly
    monthly_equity = equity.resample('M').last()
    monthly_returns = monthly_equity.pct_change()
    
    # Pivot for heatmap
    monthly_returns_pct = monthly_returns * 100  # Convert to percentage
    
    # Create year x month matrix
    monthly_returns_pct.index = pd.to_datetime(monthly_returns_pct.index)
    pivot = monthly_returns_pct.groupby([monthly_returns_pct.index.year, 
                                         monthly_returns_pct.index.month]).first().unstack()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Heatmap
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                cbar_kws={'label': 'Return (%)'}, ax=ax, linewidths=0.5)
    
    ax.set_xlabel('Month')
    ax.set_ylabel('Year')
    ax.set_title('Monthly Returns Heatmap (%)')
    
    # Set month names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_xticklabels(month_names)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Saved monthly returns to {save_path}")
    
    return fig


def plot_turnover_analysis(
    positions: pd.Series,
    figsize: tuple = (12, 5),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot turnover analysis.
    
    Args:
        positions: Position series
        figsize: Figure size
        save_path: Optional path to save
        
    Returns:
        Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Calculate position changes
    position_changes = np.abs(np.diff(positions.values, prepend=0))
    
    # Turnover over time
    ax1.plot(positions.index, position_changes, linewidth=0.5, alpha=0.7)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Position Change')
    ax1.set_title('Turnover Over Time')
    ax1.grid(True, alpha=0.3)
    
    # Distribution of position changes
    ax2.hist(position_changes[position_changes > 0], bins=50, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Position Change')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Turnover')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics
    stats_text = (f"Mean: {position_changes.mean():.4f}\n"
                 f"Median: {np.median(position_changes):.4f}\n"
                 f"Max: {position_changes.max():.4f}")
    ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', alpha=0.1))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Saved turnover analysis to {save_path}")
    
    return fig

