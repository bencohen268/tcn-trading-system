"""Visualization and diagnostic modules."""

from .layer1_diagnostics import (
    plot_feature_timeseries,
    plot_feature_distributions,
    plot_correlation_matrix,
    plot_label_balance,
    plot_missing_data,
    generate_layer1_report,
)

from .layer3_diagnostics import (
    plot_training_curves,
    plot_calibration,
    plot_prediction_distribution,
    plot_confusion_matrix,
    generate_layer3_report,
)

from .backtest_plots import (
    plot_equity_curve,
    plot_positions_vs_price,
    plot_returns_distribution,
    plot_monthly_returns,
    plot_turnover_analysis,
)

__all__ = [
    # Layer 1
    'plot_feature_timeseries',
    'plot_feature_distributions',
    'plot_correlation_matrix',
    'plot_label_balance',
    'plot_missing_data',
    'generate_layer1_report',
    # Layer 3
    'plot_training_curves',
    'plot_calibration',
    'plot_prediction_distribution',
    'plot_confusion_matrix',
    'generate_layer3_report',
    # Backtest
    'plot_equity_curve',
    'plot_positions_vs_price',
    'plot_returns_distribution',
    'plot_monthly_returns',
    'plot_turnover_analysis',
]
