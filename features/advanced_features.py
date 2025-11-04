"""
Advanced feature engineering with regime detection, volatility forecasting,
and cross-asset signals.

This module adds ~20 new features beyond the basic 12.
"""

from typing import Optional, Dict
import numpy as np
import pandas as pd


class AdvancedFeatureEngineer:
    """
    Advanced feature engineering for improved signal detection.
    
    Adds features for:
    - Regime detection (trend strength, volatility regime)
    - Volatility forecasting (GARCH-style features)
    - Microstructure proxies
    - Higher-order interactions
    """
    
    def __init__(
        self,
        regime_window: int = 20,
        vol_windows: list = [5, 10, 20, 60],
        momentum_windows: list = [5, 10, 20],
    ):
        """
        Initialize advanced feature engineer.
        
        Args:
            regime_window: Window for regime detection
            vol_windows: Multiple windows for volatility features
            momentum_windows: Windows for momentum features
        """
        self.regime_window = regime_window
        self.vol_windows = vol_windows
        self.momentum_windows = momentum_windows
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all advanced features.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            DataFrame with additional features
        """
        features = pd.DataFrame(index=df.index)
        
        # === REGIME FEATURES ===
        features = self._add_regime_features(df, features)
        
        # === VOLATILITY FEATURES ===
        features = self._add_volatility_features(df, features)
        
        # === MOMENTUM FEATURES ===
        features = self._add_momentum_features(df, features)
        
        # === MICROSTRUCTURE PROXIES ===
        features = self._add_microstructure_features(df, features)
        
        # === INTERACTION FEATURES ===
        features = self._add_interaction_features(df, features)
        
        # Fill NaNs
        features = features.fillna(0.0)
        
        return features
    
    def _add_regime_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add regime detection features."""
        returns = np.log(df['close'] / df['close'].shift(1))
        
        # 1. Trend strength (abs momentum / volatility)
        momentum = returns.rolling(self.regime_window).mean()
        volatility = returns.rolling(self.regime_window).std()
        features['trend_strength'] = np.abs(momentum) / (volatility + 1e-6)
        
        # 2. Directional trend (signed momentum / vol)
        features['trend_direction'] = momentum / (volatility + 1e-6)
        
        # 3. Volatility regime (current vol / long-term vol)
        vol_short = returns.rolling(5).std()
        vol_long = returns.rolling(60).std()
        features['vol_regime'] = vol_short / (vol_long + 1e-6)
        
        # 4. Autocorrelation (mean reversion indicator)
        features['autocorr_1'] = returns.rolling(self.regime_window).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, raw=False
        )
        
        # 5. Higher highs / lower lows (trend confirmation)
        high_rolling_max = df['high'].rolling(self.regime_window).max()
        low_rolling_min = df['low'].rolling(self.regime_window).min()
        features['new_high'] = (df['high'] >= high_rolling_max.shift(1)).astype(float)
        features['new_low'] = (df['low'] <= low_rolling_min.shift(1)).astype(float)
        
        return features
    
    def _add_volatility_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add volatility forecasting features."""
        returns = np.log(df['close'] / df['close'].shift(1))
        
        # Multiple volatility windows
        for window in self.vol_windows:
            # 1. Simple volatility
            features[f'vol_{window}'] = returns.rolling(window).std()
            
            # 2. Parkinson volatility (high-low)
            hl_ratio = np.log(df['high'] / df['low'])
            features[f'vol_parkinson_{window}'] = np.sqrt(
                (1 / (4 * np.log(2))) * (hl_ratio ** 2).rolling(window).mean()
            )
        
        # 3. GARCH-style: squared returns (variance proxy)
        features['squared_return'] = returns ** 2
        features['squared_return_ma'] = features['squared_return'].rolling(20).mean()
        
        # 4. Volatility of volatility (uncertainty)
        vol_20 = returns.rolling(20).std()
        features['vol_of_vol'] = vol_20.rolling(20).std()
        
        # 5. Realized range volatility
        features['range_vol'] = (df['high'] - df['low']) / df['close']
        features['range_vol_ma'] = features['range_vol'].rolling(20).mean()
        
        return features
    
    def _add_momentum_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add momentum and mean reversion features."""
        returns = np.log(df['close'] / df['close'].shift(1))
        
        for window in self.momentum_windows:
            # 1. Simple momentum (cumulative return)
            features[f'momentum_{window}'] = returns.rolling(window).sum()
            
            # 2. Momentum acceleration (change in momentum)
            mom = returns.rolling(window).sum()
            features[f'momentum_accel_{window}'] = mom.diff()
            
            # 3. Mean reversion score (distance from mean / vol)
            rolling_mean = df['close'].rolling(window).mean()
            rolling_std = df['close'].rolling(window).std()
            features[f'mean_reversion_{window}'] = (
                (df['close'] - rolling_mean) / (rolling_std + 1e-6)
            )
        
        # 4. RSI-style momentum
        delta = returns
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / (loss + 1e-6)
        features['rsi'] = 1 - (1 / (1 + rs))
        
        return features
    
    def _add_microstructure_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add microstructure proxy features."""
        
        # 1. Amihud illiquidity (price impact of volume)
        returns = np.abs(np.log(df['close'] / df['close'].shift(1)))
        volume_dollars = df['volume'] * df['close']
        features['illiquidity'] = returns / (volume_dollars + 1e-6)
        features['illiquidity_ma'] = features['illiquidity'].rolling(20).mean()
        
        # 2. Volume-weighted price movement
        price_change = df['close'].diff()
        features['volume_weighted_change'] = price_change * df['volume']
        features['volume_weighted_change_ma'] = features['volume_weighted_change'].rolling(10).mean()
        
        # 3. Intraday gap (open vs previous close)
        features['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        
        # 4. Gap follow-through (does gap fill or extend?)
        features['gap_followthrough'] = (
            (df['close'] - df['open']) / df['open']
        ) * np.sign(features['gap'])
        
        # 5. Volume surge indicator
        vol_ma = df['volume'].rolling(20).mean()
        vol_std = df['volume'].rolling(20).std()
        features['volume_surge'] = (df['volume'] - vol_ma) / (vol_std + 1e-6)
        
        return features
    
    def _add_interaction_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add feature interactions."""
        returns = np.log(df['close'] / df['close'].shift(1))
        
        # 1. Momentum × Volatility
        mom_10 = returns.rolling(10).sum()
        vol_10 = returns.rolling(10).std()
        features['mom_vol_interaction'] = mom_10 * vol_10
        
        # 2. Volume × Price change
        price_change = returns
        vol_norm = (df['volume'] - df['volume'].rolling(20).mean()) / (
            df['volume'].rolling(20).std() + 1e-6
        )
        features['volume_price_interaction'] = vol_norm * price_change
        
        # 3. Trend × Mean reversion
        if 'trend_strength' in features and 'mean_reversion_10' in features:
            features['trend_meanrev_interaction'] = (
                features['trend_strength'] * features['mean_reversion_10']
            )
        
        return features


def add_cross_asset_features(
    primary_df: pd.DataFrame,
    vix_df: Optional[pd.DataFrame] = None,
    spy_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Add cross-asset features (VIX, SPY, etc.).
    
    Args:
        primary_df: Primary asset DataFrame
        vix_df: VIX DataFrame (if available)
        spy_df: SPY DataFrame (for non-SPY assets)
        
    Returns:
        DataFrame with cross-asset features
    """
    features = pd.DataFrame(index=primary_df.index)
    
    if vix_df is not None:
        # Align indices
        vix_aligned = vix_df['close'].reindex(primary_df.index, method='ffill')
        
        # VIX level
        features['vix_level'] = vix_aligned
        
        # VIX change
        features['vix_change'] = vix_aligned.pct_change()
        
        # VIX regime (high vs low vol environment)
        vix_ma = vix_aligned.rolling(60).mean()
        features['vix_regime'] = vix_aligned / (vix_ma + 1e-6)
    
    if spy_df is not None:
        # Market beta proxy
        spy_returns = np.log(spy_df['close'] / spy_df['close'].shift(1))
        asset_returns = np.log(primary_df['close'] / primary_df['close'].shift(1))
        
        # Rolling correlation with market
        features['market_correlation'] = spy_returns.rolling(60).corr(asset_returns)
        
        # Relative strength
        spy_mom = spy_returns.rolling(20).sum()
        asset_mom = asset_returns.rolling(20).sum()
        features['relative_strength'] = asset_mom - spy_mom
    
    features = features.fillna(0.0)
    return features


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar effect features.
    
    Args:
        df: DataFrame with datetime index
        
    Returns:
        DataFrame with calendar features
    """
    features = pd.DataFrame(index=df.index)
    
    # Day of week effects
    features['monday'] = (df.index.dayofweek == 0).astype(float)
    features['friday'] = (df.index.dayofweek == 4).astype(float)
    
    # End of month effect
    features['end_of_month'] = (df.index.day >= 26).astype(float)
    
    # Quarter end effect
    features['quarter_end'] = df.index.to_series().apply(
        lambda x: 1.0 if x.month in [3, 6, 9, 12] and x.day >= 26 else 0.0
    ).values
    
    # Holiday proximity (approximate - need calendar library for exact)
    # Simplified: flag first/last day of month as "near holiday"
    features['near_holiday'] = (
        (df.index.day <= 3) | (df.index.day >= 26)
    ).astype(float)
    
    return features

