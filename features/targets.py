"""Target label creation utilities."""

from typing import Optional
import numpy as np
import pandas as pd


def create_binary_target(df: pd.DataFrame, threshold: float = 0.0) -> pd.Series:
    """
    Create binary next-bar direction target.
    
    Target is 1 if next bar's close is higher than current close,
    0 otherwise.
    
    Args:
        df: OHLCV DataFrame
        threshold: Optional return threshold (default 0.0 for pure direction)
        
    Returns:
        Series with binary labels (0 or 1)
        
    Example:
        If threshold=0, label is 1 if next_close > close
        If threshold=0.001, label is 1 if (next_close - close)/close > 0.001
    """
    # Calculate next-bar return
    next_return = df['close'].shift(-1) / df['close'] - 1
    
    # Binary label based on threshold
    target = (next_return > threshold).astype(int)
    
    # Last bar has no target (future unknown)
    target.iloc[-1] = np.nan
    
    return target


def create_threshold_target(
    df: pd.DataFrame,
    threshold_up: float = 0.001,
    threshold_down: float = -0.001,
) -> pd.Series:
    """
    Create ternary target with neutral zone.
    
    Args:
        df: OHLCV DataFrame
        threshold_up: Return threshold for "up" class
        threshold_down: Return threshold for "down" class
        
    Returns:
        Series with labels: 1 (up), 0 (neutral), -1 (down)
    """
    # Calculate next-bar return
    next_return = df['close'].shift(-1) / df['close'] - 1
    
    # Ternary classification
    target = pd.Series(0, index=df.index)  # Default to neutral
    target[next_return > threshold_up] = 1  # Up
    target[next_return < threshold_down] = -1  # Down
    
    # Last bar has no target
    target.iloc[-1] = np.nan
    
    return target


def create_regression_target(df: pd.DataFrame, clip: Optional[float] = None) -> pd.Series:
    """
    Create regression target (next-bar return).
    
    Args:
        df: OHLCV DataFrame
        clip: Optional value to clip outliers (e.g., 0.05 for ±5%)
        
    Returns:
        Series with next-bar returns
    """
    # Calculate next-bar return
    next_return = df['close'].shift(-1) / df['close'] - 1
    
    # Optionally clip outliers
    if clip is not None:
        next_return = next_return.clip(-clip, clip)
    
    # Last bar has no target
    next_return.iloc[-1] = np.nan
    
    return next_return


def get_label_balance(target: pd.Series) -> dict:
    """
    Get label distribution statistics.
    
    Args:
        target: Target series
        
    Returns:
        Dictionary with class counts and percentages
    """
    counts = target.value_counts().sort_index()
    percentages = (counts / counts.sum() * 100).round(2)
    
    result = {
        'counts': counts.to_dict(),
        'percentages': percentages.to_dict(),
        'total': len(target.dropna()),
    }
    
    return result

