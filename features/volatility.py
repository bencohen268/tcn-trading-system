"""Volatility estimation utilities."""

from typing import Optional
import numpy as np
import pandas as pd


def estimate_volatility(
    df: pd.DataFrame,
    method: str = "std",
    window: int = 20,
    **kwargs,
) -> pd.Series:
    """
    Estimate rolling volatility.
    
    Args:
        df: OHLCV DataFrame
        method: Estimation method ("std", "ewma", "parkinson", "garman_klass")
        window: Rolling window size
        **kwargs: Additional parameters for specific methods
        
    Returns:
        Series with volatility estimates
    """
    if method == "std":
        return _volatility_std(df, window)
    elif method == "ewma":
        return _volatility_ewma(df, window, kwargs.get('alpha', 0.94))
    elif method == "parkinson":
        return _volatility_parkinson(df, window)
    elif method == "garman_klass":
        return _volatility_garman_klass(df, window)
    else:
        raise ValueError(f"Unknown volatility method: {method}")


def _volatility_std(df: pd.DataFrame, window: int) -> pd.Series:
    """Simple rolling standard deviation of returns."""
    returns = np.log(df['close'] / df['close'].shift(1))
    vol = returns.rolling(window=window).std()
    return vol


def _volatility_ewma(df: pd.DataFrame, window: int, alpha: float = 0.94) -> pd.Series:
    """Exponentially weighted moving average volatility."""
    returns = np.log(df['close'] / df['close'].shift(1))
    vol = returns.ewm(span=window, alpha=1-alpha).std()
    return vol


def _volatility_parkinson(df: pd.DataFrame, window: int) -> pd.Series:
    """
    Parkinson volatility estimator using high-low range.
    
    More efficient than close-to-close for assets that trade continuously.
    Formula: vol = sqrt((1/(4*log(2))) * mean((log(H/L))^2))
    """
    hl_ratio = np.log(df['high'] / df['low'])
    hl_squared = hl_ratio ** 2
    
    factor = 1 / (4 * np.log(2))
    vol = np.sqrt(factor * hl_squared.rolling(window=window).mean())
    
    return vol


def _volatility_garman_klass(df: pd.DataFrame, window: int) -> pd.Series:
    """
    Garman-Klass volatility estimator.
    
    Uses open, high, low, close for more efficient estimation.
    Formula: vol = sqrt((0.5 * (log(H/L))^2 - (2*log(2)-1) * (log(C/O))^2))
    """
    hl = np.log(df['high'] / df['low']) ** 2
    co = np.log(df['close'] / df['open']) ** 2
    
    vol_squared = (0.5 * hl - (2 * np.log(2) - 1) * co).rolling(window=window).mean()
    vol = np.sqrt(vol_squared)
    
    return vol


def estimate_vol_forecast(
    df: pd.DataFrame,
    method: str = "std",
    window: int = 20,
    annualize: bool = False,
    periods_per_year: int = 252,
) -> pd.Series:
    """
    Estimate volatility forecast for risk scaling.
    
    Args:
        df: OHLCV DataFrame
        method: Estimation method
        window: Rolling window
        annualize: Whether to annualize the volatility
        periods_per_year: Number of periods per year (252 for daily, ~78*252 for 5min)
        
    Returns:
        Series with volatility forecasts
    """
    vol = estimate_volatility(df, method=method, window=window)
    
    if annualize:
        vol = vol * np.sqrt(periods_per_year)
    
    return vol

