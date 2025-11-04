"""Feature engineering for TCN trading system.

This module creates the 12 core features per bar:
1. bar_return: log or pct return
2. rolling_volatility: std of returns
3. hl_range: (high - low) / close
4. intrabar_position: (close - low) / (high - low)
5. sma_fast: fast simple moving average
6. sma_slow: slow simple moving average
7. sma_ratio: fast/slow ratio
8. log_volume: log(volume)
9. volume_ratio: volume vs rolling avg
10. time_of_day: hour encoding (for intraday)
11. day_of_week: weekday encoding
12. prev_return: lag-1 return
"""

from typing import Dict, Optional, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeatureEngineer:
    """
    Feature engineering pipeline for OHLCV data.
    
    Produces a fixed set of 12 features per bar, normalized and ready
    for TCN input.
    """
    
    def __init__(
        self,
        vol_window: int = 20,
        sma_fast_period: int = 5,
        sma_slow_period: int = 20,
        volume_window: int = 20,
        normalization: str = "zscore",
        rolling_norm: bool = False,
    ):
        """
        Initialize feature engineer.
        
        Args:
            vol_window: Rolling window for volatility
            sma_fast_period: Fast SMA period
            sma_slow_period: Slow SMA period
            volume_window: Rolling window for volume average
            normalization: Normalization method ("zscore", "minmax", "robust")
            rolling_norm: Use rolling normalization (for live trading)
        """
        self.vol_window = vol_window
        self.sma_fast_period = sma_fast_period
        self.sma_slow_period = sma_slow_period
        self.volume_window = volume_window
        self.normalization = normalization
        self.rolling_norm = rolling_norm
        
        # Fitted scalers (for zscore normalization)
        self.scalers: Dict[str, StandardScaler] = {}
        self.is_fitted = False
        
    def fit(self, df: pd.DataFrame) -> 'FeatureEngineer':
        """
        Fit normalization parameters on training data.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            self
        """
        # Create features without normalization
        features_df = self._create_features(df)
        
        # Fit scalers
        if self.normalization == "zscore":
            for col in features_df.columns:
                scaler = StandardScaler()
                scaler.fit(features_df[[col]].values)
                self.scalers[col] = scaler
        
        self.is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform OHLCV data into features.
        
        Args:
            df: OHLCV DataFrame with columns: open, high, low, close, volume
            
        Returns:
            DataFrame with 12 feature columns
        """
        # Create raw features
        features_df = self._create_features(df)
        
        # Normalize
        if self.normalization == "zscore":
            if self.rolling_norm:
                # Rolling normalization (for live trading)
                features_df = self._rolling_normalize(features_df)
            else:
                # Static normalization using fitted scalers
                if not self.is_fitted:
                    raise ValueError("Must call fit() before transform() when using static normalization")
                features_df = self._static_normalize(features_df)
        
        return features_df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)
    
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all 12 raw features.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            DataFrame with 12 feature columns (not normalized)
        """
        features = pd.DataFrame(index=df.index)
        
        # 1. Bar return (log return)
        features['bar_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # 2. Rolling volatility (std of returns)
        returns = features['bar_return']
        features['rolling_volatility'] = returns.rolling(window=self.vol_window).std()
        
        # 3. High-low range (normalized by close)
        features['hl_range'] = (df['high'] - df['low']) / df['close']
        
        # 4. Intrabar position (where close is within the bar)
        hl_diff = df['high'] - df['low']
        features['intrabar_position'] = np.where(
            hl_diff > 0,
            (df['close'] - df['low']) / hl_diff,
            0.5  # Default to midpoint if no range
        )
        
        # 5. Fast SMA (normalized by close)
        sma_fast = df['close'].rolling(window=self.sma_fast_period).mean()
        features['sma_fast'] = sma_fast / df['close'] - 1  # Percentage deviation
        
        # 6. Slow SMA (normalized by close)
        sma_slow = df['close'].rolling(window=self.sma_slow_period).mean()
        features['sma_slow'] = sma_slow / df['close'] - 1
        
        # 7. SMA ratio (fast/slow)
        features['sma_ratio'] = sma_fast / sma_slow - 1
        
        # 8. Log volume
        features['log_volume'] = np.log(df['volume'] + 1)  # +1 to avoid log(0)
        
        # 9. Volume ratio (vs rolling average)
        vol_ma = df['volume'].rolling(window=self.volume_window).mean()
        features['volume_ratio'] = df['volume'] / (vol_ma + 1e-6) - 1
        
        # 10. Time of day (hour) - normalized to [-1, 1]
        if hasattr(df.index, 'hour'):
            # For intraday data
            features['time_of_day'] = (df.index.hour - 12) / 12  # Center at noon
        else:
            # For daily data, set to 0
            features['time_of_day'] = 0.0
        
        # 11. Day of week - normalized to [-1, 1]
        if hasattr(df.index, 'dayofweek'):
            features['day_of_week'] = (df.index.dayofweek - 2) / 2  # Center at Wednesday
        else:
            features['day_of_week'] = 0.0
        
        # 12. Previous return (lag-1)
        features['prev_return'] = features['bar_return'].shift(1)
        
        # Fill NaN values with 0 (from initial rolling windows)
        features = features.fillna(0.0)
        
        return features
    
    def _static_normalize(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Apply static normalization using fitted scalers."""
        normalized = features_df.copy()
        
        for col in features_df.columns:
            if col in self.scalers:
                normalized[col] = self.scalers[col].transform(features_df[[col]].values).flatten()
        
        return normalized
    
    def _rolling_normalize(self, features_df: pd.DataFrame, window: int = 252) -> pd.DataFrame:
        """Apply rolling normalization (for live trading)."""
        normalized = features_df.copy()
        
        for col in features_df.columns:
            rolling_mean = features_df[col].rolling(window=window, min_periods=20).mean()
            rolling_std = features_df[col].rolling(window=window, min_periods=20).std()
            
            normalized[col] = (features_df[col] - rolling_mean) / (rolling_std + 1e-6)
        
        # Fill initial NaNs with 0
        normalized = normalized.fillna(0.0)
        
        return normalized
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names in order."""
        return [
            'bar_return',
            'rolling_volatility',
            'hl_range',
            'intrabar_position',
            'sma_fast',
            'sma_slow',
            'sma_ratio',
            'log_volume',
            'volume_ratio',
            'time_of_day',
            'day_of_week',
            'prev_return',
        ]

