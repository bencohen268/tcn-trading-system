"""
Alternative labeling strategies beyond simple next-bar direction.

These labels are often more predictable and robust:
- Multi-day returns (less noise)
- Return quantiles (focus on tails)
- Regime classification (trend/reversal/neutral)
- Volatility prediction (often easier than direction)
"""

from typing import Optional, Dict
import numpy as np
import pandas as pd


def create_multiday_return_label(
    df: pd.DataFrame,
    horizon: int = 5,
    threshold: float = 0.01,
) -> pd.Series:
    """
    Create label based on multi-day forward return.
    
    Less noisy than next-bar prediction.
    
    Args:
        df: OHLCV DataFrame
        horizon: Number of days to look forward
        threshold: Return threshold for classification
        
    Returns:
        Binary series: 1 if forward return > threshold, 0 otherwise
    """
    # Forward return over horizon
    forward_return = df['close'].shift(-horizon) / df['close'] - 1
    
    # Binary label
    label = (forward_return > threshold).astype(int)
    
    # Last `horizon` bars have no label
    label.iloc[-horizon:] = np.nan
    
    return label


def create_quantile_labels(
    df: pd.DataFrame,
    horizon: int = 5,
    n_quantiles: int = 3,
) -> pd.Series:
    """
    Create labels based on return quantiles.
    
    Focuses on predicting extreme moves (tails) which may be more predictable.
    
    Args:
        df: OHLCV DataFrame
        horizon: Number of days to look forward
        n_quantiles: Number of quantile bins (3 = terciles, 5 = quintiles)
        
    Returns:
        Series with quantile labels (0 to n_quantiles-1)
    """
    # Forward return
    forward_return = df['close'].shift(-horizon) / df['close'] - 1
    
    # Quantile labels
    label = pd.qcut(
        forward_return.rank(method='first'),
        q=n_quantiles,
        labels=False,
        duplicates='drop'
    )
    
    # Last `horizon` bars have no label
    label.iloc[-horizon:] = np.nan
    
    return label


def create_tail_event_labels(
    df: pd.DataFrame,
    horizon: int = 5,
    threshold_pct: float = 0.02,
) -> pd.Series:
    """
    Create labels for tail events (large moves).
    
    Ternary classification: large up, large down, or neutral.
    
    Args:
        df: OHLCV DataFrame
        horizon: Number of days to look forward
        threshold_pct: Threshold for "large" move (e.g., 2%)
        
    Returns:
        Series with labels: 1 (large up), 0 (neutral), -1 (large down)
    """
    # Forward return
    forward_return = df['close'].shift(-horizon) / df['close'] - 1
    
    # Ternary classification
    label = pd.Series(0, index=df.index)  # Default: neutral
    label[forward_return > threshold_pct] = 1  # Large up
    label[forward_return < -threshold_pct] = -1  # Large down
    
    # Last `horizon` bars have no label
    label.iloc[-horizon:] = np.nan
    
    return label


def create_regime_labels(
    df: pd.DataFrame,
    window: int = 20,
) -> pd.Series:
    """
    Create regime labels (trend up, trend down, sideways).
    
    Based on price position relative to recent range and momentum.
    
    Args:
        df: OHLCV DataFrame
        window: Window for regime detection
        
    Returns:
        Series with labels: 1 (uptrend), 0 (sideways), -1 (downtrend)
    """
    returns = np.log(df['close'] / df['close'].shift(1))
    
    # Momentum
    momentum = returns.rolling(window).mean()
    volatility = returns.rolling(window).std()
    trend_strength = momentum / (volatility + 1e-6)
    
    # Price position in recent range
    rolling_high = df['close'].rolling(window).max()
    rolling_low = df['close'].rolling(window).min()
    range_position = (df['close'] - rolling_low) / (rolling_high - rolling_low + 1e-6)
    
    # Regime classification
    label = pd.Series(0, index=df.index)  # Default: sideways
    
    # Uptrend: positive momentum AND near top of range
    uptrend_mask = (trend_strength > 0.5) & (range_position > 0.6)
    label[uptrend_mask] = 1
    
    # Downtrend: negative momentum AND near bottom of range
    downtrend_mask = (trend_strength < -0.5) & (range_position < 0.4)
    label[downtrend_mask] = -1
    
    return label


def create_volatility_labels(
    df: pd.DataFrame,
    horizon: int = 5,
    threshold_ratio: float = 1.2,
) -> pd.Series:
    """
    Create labels for volatility prediction.
    
    Often more predictable than direction.
    
    Args:
        df: OHLCV DataFrame
        horizon: Number of days to look forward
        threshold_ratio: Ratio for "high vol" classification
        
    Returns:
        Binary series: 1 if future vol > threshold × current vol, 0 otherwise
    """
    returns = np.log(df['close'] / df['close'].shift(1))
    
    # Current volatility (realized)
    current_vol = returns.rolling(20).std()
    
    # Future volatility (looking forward)
    future_vol = returns.rolling(horizon).std().shift(-horizon)
    
    # Binary label: will vol increase?
    label = (future_vol > threshold_ratio * current_vol).astype(int)
    
    # Last `horizon` bars have no label
    label.iloc[-horizon:] = np.nan
    
    return label


def create_reversal_labels(
    df: pd.DataFrame,
    lookback: int = 5,
    lookahead: int = 5,
    threshold: float = 0.015,
) -> pd.Series:
    """
    Create labels for mean reversion / reversal events.
    
    Predicts if price will reverse after a move.
    
    Args:
        df: OHLCV DataFrame
        lookback: Window for detecting prior move
        lookahead: Window for detecting reversal
        threshold: Threshold for "significant" move
        
    Returns:
        Series with labels: 1 (reversal up), 0 (continuation), -1 (reversal down)
    """
    # Recent move
    recent_return = df['close'] / df['close'].shift(lookback) - 1
    
    # Future move
    future_return = df['close'].shift(-lookahead) / df['close'] - 1
    
    # Reversal detection
    label = pd.Series(0, index=df.index)  # Default: continuation
    
    # Reversal up: was down significantly, will go up
    reversal_up = (recent_return < -threshold) & (future_return > threshold)
    label[reversal_up] = 1
    
    # Reversal down: was up significantly, will go down
    reversal_down = (recent_return > threshold) & (future_return < -threshold)
    label[reversal_down] = -1
    
    # Last `lookahead` bars have no label
    label.iloc[-lookahead:] = np.nan
    
    return label


def create_breakout_labels(
    df: pd.DataFrame,
    window: int = 20,
    horizon: int = 5,
    threshold: float = 0.01,
) -> pd.Series:
    """
    Create labels for breakout events.
    
    Predicts if price will break out of recent range.
    
    Args:
        df: OHLCV DataFrame
        window: Window for defining range
        horizon: Lookahead period
        threshold: Threshold for "significant" breakout
        
    Returns:
        Series with labels: 1 (breakout up), 0 (no breakout), -1 (breakout down)
    """
    # Recent range
    rolling_high = df['close'].rolling(window).max()
    rolling_low = df['close'].rolling(window).min()
    
    # Future price
    future_close = df['close'].shift(-horizon)
    
    # Breakout detection
    label = pd.Series(0, index=df.index)  # Default: no breakout
    
    # Breakout up: future price > recent high by threshold
    breakout_up = (future_close > rolling_high * (1 + threshold))
    label[breakout_up] = 1
    
    # Breakout down: future price < recent low by threshold
    breakout_down = (future_close < rolling_low * (1 - threshold))
    label[breakout_down] = -1
    
    # Last `horizon` bars have no label
    label.iloc[-horizon:] = np.nan
    
    return label


def compare_label_strategies(
    df: pd.DataFrame,
    horizon: int = 5,
) -> pd.DataFrame:
    """
    Compare different labeling strategies.
    
    Shows label balance and correlation between strategies.
    
    Args:
        df: OHLCV DataFrame
        horizon: Lookahead period
        
    Returns:
        DataFrame with all labels for comparison
    """
    labels_df = pd.DataFrame(index=df.index)
    
    # Basic next-bar
    labels_df['next_bar'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    # Multi-day
    labels_df['multiday'] = create_multiday_return_label(df, horizon=horizon)
    
    # Quantiles
    labels_df['quantile'] = create_quantile_labels(df, horizon=horizon, n_quantiles=3)
    
    # Tail events
    labels_df['tail_event'] = create_tail_event_labels(df, horizon=horizon)
    
    # Regimes
    labels_df['regime'] = create_regime_labels(df)
    
    # Volatility
    labels_df['volatility'] = create_volatility_labels(df, horizon=horizon)
    
    # Reversal
    labels_df['reversal'] = create_reversal_labels(df)
    
    # Breakout
    labels_df['breakout'] = create_breakout_labels(df, horizon=horizon)
    
    return labels_df


def analyze_label_quality(label: pd.Series, name: str = "Label") -> Dict:
    """
    Analyze label quality and predictability.
    
    Args:
        label: Label series
        name: Name for reporting
        
    Returns:
        Dictionary with label statistics
    """
    label_clean = label.dropna()
    
    # Class balance
    value_counts = label_clean.value_counts()
    balance = value_counts / len(label_clean)
    
    # Autocorrelation (persistence)
    autocorr_1 = label_clean.autocorr(lag=1) if len(label_clean) > 1 else 0
    
    # Entropy (predictability measure)
    entropy = -(balance * np.log2(balance + 1e-6)).sum()
    
    return {
        'name': name,
        'n_samples': len(label_clean),
        'n_classes': len(value_counts),
        'balance': balance.to_dict(),
        'autocorrelation': autocorr_1,
        'entropy': entropy,
        'most_common_pct': balance.max(),
    }

