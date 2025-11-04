"""Feature engineering modules."""

from .engineering import FeatureEngineer
from .targets import create_binary_target, create_threshold_target
from .volatility import estimate_volatility
from .advanced_features import AdvancedFeatureEngineer, add_calendar_features
from .alternative_labels import (
    create_multiday_return_label,
    create_regime_labels,
    create_volatility_labels,
)

__all__ = [
    'FeatureEngineer',
    'create_binary_target',
    'create_threshold_target',
    'estimate_volatility',
    'AdvancedFeatureEngineer',
    'add_calendar_features',
    'create_multiday_return_label',
    'create_regime_labels',
    'create_volatility_labels',
]

