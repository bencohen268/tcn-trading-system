"""
Risk map: Convert model probability to position size.

This is where trading logic lives. The risk map:
1. Applies dead band (neutral zone)
2. Scales by volatility (adversity adjustment)
3. Caps exposure (risk limit)

Goal: Turn a noisy ML classifier into a sparse, risk-aware trading signal.
"""

from typing import Union
import numpy as np
import pandas as pd


def risk_map(
    p: Union[float, np.ndarray],
    vol: Union[float, np.ndarray],
    deadband: float = 0.02,
    base_leverage: float = 1.0,
    vol_target: float = 0.01,
    vol_scale_min: float = 0.2,
    vol_scale_max: float = 1.5,
    max_abs_position: float = 1.0,
) -> Union[float, np.ndarray]:
    """
    Map model probability and volatility to position size.
    
    Args:
        p: Model probability (0..1)
        vol: Volatility estimate
        deadband: Minimum edge to trade (|p - 0.5| must exceed this)
        base_leverage: Base leverage multiplier
        vol_target: Target volatility for scaling
        vol_scale_min: Minimum volatility scaling factor
        vol_scale_max: Maximum volatility scaling factor
        max_abs_position: Maximum absolute position size
        
    Returns:
        Position size in range [-max_abs_position, max_abs_position]
        
    Example:
        >>> risk_map(0.6, 0.01)  # 60% probability, 1% vol
        0.4  # Some positive position
        
        >>> risk_map(0.51, 0.01)  # Just barely above 50%
        0.0  # Within dead band, no position
    """
    # Convert to numpy for vectorized operations
    is_scalar = np.isscalar(p)
    p = np.atleast_1d(p)
    vol = np.atleast_1d(vol)
    
    # IMPORTANT: Invert probabilities! 
    # Model outputs were found to be inverted (high prob → down moves)
    # So we use (1 - p) to correct this
    p = 1.0 - p
    
    # 1. Calculate edge (distance from neutral)
    edge = p - 0.5
    
    # 2. Apply dead band
    position = np.zeros_like(edge)
    active_mask = np.abs(edge) >= deadband
    
    # 3. Scale edge beyond dead band to [0, 1]
    # Map from [deadband, 0.5] → [0, 1]
    sign = np.sign(edge[active_mask])
    mag = (np.abs(edge[active_mask]) - deadband) / (0.5 - deadband)
    mag = np.clip(mag, 0.0, 1.0)
    
    # 4. Volatility scaling (adversity adjustment)
    # When vol is high, scale down; when vol is low, scale up
    vol_scale = vol_target / (vol[active_mask] + 1e-8)
    vol_scale = np.clip(vol_scale, vol_scale_min, vol_scale_max)
    
    # 5. Combine: sign × magnitude × vol_scale × base_leverage
    position[active_mask] = sign * mag * vol_scale * base_leverage
    
    # 6. Cap exposure
    position = np.clip(position, -max_abs_position, max_abs_position)
    
    # Return scalar if input was scalar
    if is_scalar:
        return float(position[0])
    
    return position


class RiskMapper:
    """
    Stateful risk mapper with configurable parameters.
    
    Useful for backtesting and production where you want to
    maintain consistent risk parameters.
    """
    
    def __init__(
        self,
        deadband: float = 0.02,
        base_leverage: float = 1.0,
        vol_target: float = 0.01,
        vol_scale_min: float = 0.2,
        vol_scale_max: float = 1.5,
        max_abs_position: float = 1.0,
    ):
        """
        Initialize risk mapper.
        
        Args:
            deadband: Minimum edge to trade
            base_leverage: Base leverage multiplier
            vol_target: Target volatility
            vol_scale_min: Min vol scaling factor
            vol_scale_max: Max vol scaling factor
            max_abs_position: Max absolute position
        """
        self.deadband = deadband
        self.base_leverage = base_leverage
        self.vol_target = vol_target
        self.vol_scale_min = vol_scale_min
        self.vol_scale_max = vol_scale_max
        self.max_abs_position = max_abs_position
    
    def __call__(
        self,
        p: Union[float, np.ndarray],
        vol: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """
        Map probability and volatility to position.
        
        Args:
            p: Model probability
            vol: Volatility estimate
            
        Returns:
            Position size
        """
        return risk_map(
            p=p,
            vol=vol,
            deadband=self.deadband,
            base_leverage=self.base_leverage,
            vol_target=self.vol_target,
            vol_scale_min=self.vol_scale_min,
            vol_scale_max=self.vol_scale_max,
            max_abs_position=self.max_abs_position,
        )
    
    def map_series(
        self,
        probabilities: pd.Series,
        volatility: pd.Series,
    ) -> pd.Series:
        """
        Map time series of probabilities to positions.
        
        Args:
            probabilities: Series of model probabilities
            volatility: Series of volatility estimates
            
        Returns:
            Series of position sizes
        """
        positions = risk_map(
            p=probabilities.values,
            vol=volatility.values,
            deadband=self.deadband,
            base_leverage=self.base_leverage,
            vol_target=self.vol_target,
            vol_scale_min=self.vol_scale_min,
            vol_scale_max=self.vol_scale_max,
            max_abs_position=self.max_abs_position,
        )
        
        return pd.Series(positions, index=probabilities.index)
    
    def get_params(self) -> dict:
        """Get risk map parameters."""
        return {
            'deadband': self.deadband,
            'base_leverage': self.base_leverage,
            'vol_target': self.vol_target,
            'vol_scale_min': self.vol_scale_min,
            'vol_scale_max': self.vol_scale_max,
            'max_abs_position': self.max_abs_position,
        }
    
    @classmethod
    def from_config(cls, config: dict) -> 'RiskMapper':
        """Create RiskMapper from configuration dictionary."""
        risk_config = config['risk_map']
        
        return cls(
            deadband=risk_config['deadband'],
            base_leverage=risk_config['base_leverage'],
            vol_target=risk_config['vol_target'],
            vol_scale_min=risk_config['vol_scale_min'],
            vol_scale_max=risk_config['vol_scale_max'],
            max_abs_position=risk_config['max_abs_position'],
        )


def analyze_risk_map(
    probabilities: np.ndarray,
    volatilities: np.ndarray,
    deadband: float = 0.02,
    vol_target: float = 0.01,
) -> dict:
    """
    Analyze risk map behavior on a set of probabilities.
    
    Args:
        probabilities: Array of model probabilities
        volatilities: Array of volatility estimates
        deadband: Dead band parameter
        vol_target: Target volatility
        
    Returns:
        Dictionary with analysis results
    """
    positions = risk_map(
        p=probabilities,
        vol=volatilities,
        deadband=deadband,
        vol_target=vol_target,
    )
    
    # Calculate statistics
    pct_flat = (positions == 0).sum() / len(positions) * 100
    pct_long = (positions > 0).sum() / len(positions) * 100
    pct_short = (positions < 0).sum() / len(positions) * 100
    
    avg_long_size = positions[positions > 0].mean() if (positions > 0).any() else 0
    avg_short_size = positions[positions < 0].mean() if (positions < 0).any() else 0
    
    return {
        'pct_flat': pct_flat,
        'pct_long': pct_long,
        'pct_short': pct_short,
        'avg_position': positions.mean(),
        'avg_abs_position': np.abs(positions).mean(),
        'avg_long_size': avg_long_size,
        'avg_short_size': avg_short_size,
        'max_position': positions.max(),
        'min_position': positions.min(),
    }

