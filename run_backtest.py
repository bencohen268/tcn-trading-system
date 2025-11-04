#!/usr/bin/env python3
"""
Layers 4, 5, 6: Inference, Risk Map, and Backtest

This script:
1. Loads trained model
2. Runs inference on test data (Layer 4)
3. Applies risk map to convert probabilities to positions (Layer 5)
4. Runs backtest with costs (Layer 6)
5. Generates comprehensive performance report

Success criteria:
- Positive Sharpe ratio out-of-sample
- Reasonable drawdown (<20-30%)
- Model predictions correlate with returns
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# Project imports
from utils import load_config, get_paths
from models import (
    load_processed_data_for_training,
    create_tcn_from_config,
)
from models.train import load_checkpoint
from strategy import RiskMapper, Backtester
from visualization.backtest_plots import (
    plot_equity_curve,
    plot_positions_vs_price,
    plot_returns_distribution,
    plot_monthly_returns,
    plot_turnover_analysis,
)


def run_inference(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
) -> tuple:
    """
    Run model inference on dataset.
    
    Args:
        model: Trained model
        dataset: Dataset to predict on
        device: Device to run on
        
    Returns:
        Tuple of (timestamps, probabilities, targets)
    """
    model.eval()
    
    timestamps = []
    probabilities = []
    targets = []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            features, target, timestamp = dataset[i]
            
            # Add batch dimension
            features = features.unsqueeze(0).to(device)
            
            # Predict
            logit = model(features)
            prob = torch.sigmoid(logit).item()
            
            timestamps.append(timestamp)
            probabilities.append(prob)
            targets.append(target.item())
    
    return timestamps, np.array(probabilities), np.array(targets)


def analyze_predictions(
    timestamps: list,
    probabilities: np.ndarray,
    targets: np.ndarray,
    returns: pd.Series,
) -> dict:
    """
    Analyze prediction quality.
    
    Args:
        timestamps: List of timestamps
        probabilities: Predicted probabilities
        targets: True targets
        returns: Actual returns
        
    Returns:
        Dictionary of analysis metrics
    """
    from sklearn.metrics import roc_auc_score, accuracy_score
    
    # Basic metrics
    auc = roc_auc_score(targets, probabilities)
    accuracy = accuracy_score(targets, (probabilities > 0.5).astype(int))
    
    # Correlation with returns
    # Align probabilities with returns
    probs_series = pd.Series(probabilities, index=timestamps)
    aligned_probs, aligned_returns = probs_series.align(returns, join='inner')
    
    correlation = aligned_probs.corr(aligned_returns)
    
    # Directional accuracy
    pred_direction = (aligned_probs > 0.5).astype(int)
    true_direction = (aligned_returns > 0).astype(int)
    directional_accuracy = (pred_direction == true_direction).mean()
    
    return {
        'auc': auc,
        'accuracy': accuracy,
        'correlation_with_returns': correlation,
        'directional_accuracy': directional_accuracy,
        'mean_probability': probabilities.mean(),
        'std_probability': probabilities.std(),
    }


def main():
    """Run Layers 4, 5, 6: Inference, Risk Map, and Backtest."""
    
    print("\n" + "="*70)
    print("LAYERS 4, 5, 6: INFERENCE → RISK MAP → BACKTEST")
    print("="*70 + "\n")
    
    # 1. Load configuration
    print("[1/7] Loading configuration...")
    config = load_config()
    paths = get_paths()
    
    symbol = config['data']['symbol']
    frequency = config['data']['frequency']
    device = torch.device(config['experiment']['device'])
    
    print(f"  Symbol: {symbol}")
    print(f"  Frequency: {frequency}")
    
    # 2. Load datasets
    print("\n[2/7] Loading test dataset...")
    
    _, _, test_dataset = load_processed_data_for_training(
        processed_data_dir=paths['processed_data'],
        symbol=symbol,
        frequency=frequency,
        seq_len=config['data']['seq_len'],
    )
    
    # Also load raw data for backtesting
    test_data = pd.read_parquet(
        paths['processed_data'] / f"{symbol}_{frequency}_test.parquet"
    )
    
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Date range: {test_data.index[0]} to {test_data.index[-1]}")
    
    # 3. Load trained model
    print("\n[3/7] Loading trained model...")
    
    model = create_tcn_from_config(config)
    model = model.to(device)
    
    checkpoint_path = paths['checkpoints'] / 'best_model.pt'
    load_checkpoint(model, checkpoint_path, device)
    
    # 4. Run inference (Layer 4)
    print("\n[4/7] Running inference on test data...")
    
    timestamps, probabilities, targets = run_inference(model, test_dataset, device)
    
    print(f"  Generated {len(probabilities)} predictions")
    print(f"  Mean probability: {probabilities.mean():.3f}")
    print(f"  Std probability: {probabilities.std():.3f}")
    
    # Calculate actual returns for analysis
    returns = test_data['raw_close'].pct_change()
    
    # Analyze prediction quality
    pred_analysis = analyze_predictions(timestamps, probabilities, targets, returns)
    
    print(f"\n  Prediction Quality:")
    print(f"    AUC: {pred_analysis['auc']:.4f}")
    print(f"    Accuracy: {pred_analysis['accuracy']:.4f}")
    print(f"    Correlation with returns: {pred_analysis['correlation_with_returns']:.4f}")
    print(f"    Directional accuracy: {pred_analysis['directional_accuracy']:.4f}")
    
    # 5. Apply risk map (Layer 5)
    print("\n[5/7] Applying risk map...")
    
    risk_mapper = RiskMapper.from_config(config)
    
    # Create series for risk mapping
    prob_series = pd.Series(probabilities, index=timestamps)
    vol_series = test_data['volatility'].loc[timestamps]
    
    # Map to positions
    positions = risk_mapper.map_series(prob_series, vol_series)
    
    print(f"  Risk map parameters:")
    for key, value in risk_mapper.get_params().items():
        print(f"    {key}: {value}")
    
    # Analyze positions
    pct_flat = (positions == 0).sum() / len(positions) * 100
    pct_long = (positions > 0).sum() / len(positions) * 100
    pct_short = (positions < 0).sum() / len(positions) * 100
    
    print(f"\n  Position Distribution:")
    print(f"    Flat: {pct_flat:.1f}%")
    print(f"    Long: {pct_long:.1f}%")
    print(f"    Short: {pct_short:.1f}%")
    print(f"    Avg abs position: {np.abs(positions).mean():.3f}")
    
    # 6. Run backtest (Layer 6)
    print("\n[6/7] Running backtest...")
    
    backtester = Backtester.from_config(config)
    
    # Get prices for backtest
    prices = test_data['raw_close'].loc[timestamps]
    
    # Run backtest
    result = backtester.run(
        prices=prices,
        target_positions=positions,
        volatility=vol_series,
    )
    
    # Print summary
    print(result.summary())
    
    # 7. Generate visualizations
    print("\n[7/7] Generating visualizations...")
    
    output_dir = paths['figures']
    
    # Equity curve
    plot_equity_curve(
        result.equity_curve,
        save_path=output_dir / 'backtest_equity_curve.png'
    )
    plt.close()
    
    # Positions vs price
    plot_positions_vs_price(
        prices,
        result.positions,
        save_path=output_dir / 'backtest_positions_vs_price.png'
    )
    plt.close()
    
    # Returns distribution
    plot_returns_distribution(
        result.returns,
        save_path=output_dir / 'backtest_returns_distribution.png'
    )
    plt.close()
    
    # Monthly returns
    if len(result.returns) > 30:  # Only if we have enough data
        plot_monthly_returns(
            result.returns,
            save_path=output_dir / 'backtest_monthly_returns.png'
        )
        plt.close()
    
    # Turnover
    plot_turnover_analysis(
        result.positions,
        save_path=output_dir / 'backtest_turnover.png'
    )
    plt.close()
    
    print(f"  All plots saved to: {output_dir}")
    
    # Save results to CSV
    results_file = paths['backtests'] / f"{symbol}_{frequency}_backtest_results.csv"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    results_df = pd.DataFrame({
        'timestamp': timestamps,
        'probability': probabilities,
        'position': positions.values,
        'equity': result.equity_curve.values,
        'price': prices.values,
    })
    results_df.to_csv(results_file, index=False)
    print(f"  Results saved to: {results_file}")
    
    # Final assessment
    print("\n" + "="*70)
    print("BACKTEST COMPLETE ✓")
    print("="*70)
    
    print("\nSuccess Criteria Check:")
    success = True
    
    if result.sharpe_ratio > 0:
        print(f"  ✓ Positive Sharpe ratio ({result.sharpe_ratio:.2f})")
    else:
        print(f"  ✗ Negative Sharpe ratio ({result.sharpe_ratio:.2f})")
        success = False
    
    if result.max_drawdown > -0.30:
        print(f"  ✓ Acceptable drawdown ({result.max_drawdown:.2%})")
    else:
        print(f"  ⚠️  Large drawdown ({result.max_drawdown:.2%})")
    
    if result.total_return > 0:
        print(f"  ✓ Positive total return ({result.total_return:.2%})")
    else:
        print(f"  ✗ Negative total return ({result.total_return:.2%})")
        success = False
    
    print("\nNext steps:")
    if success:
        print("  1. Review all diagnostic plots")
        print("  2. Analyze monthly performance breakdown")
        print("  3. Consider walk-forward testing (Layer 7)")
        print("  4. If robust, consider paper trading")
    else:
        print("  1. Review where performance breaks down")
        print("  2. Check if model predictions have edge")
        print("  3. Consider adjusting:")
        print("     - Risk map parameters (deadband, vol_target)")
        print("     - Model architecture")
        print("     - Feature engineering")
        print("  4. Iterate and re-test")
    
    print("\nAll results saved to:")
    print(f"  Figures: {paths['figures']}")
    print(f"  Data: {paths['backtests']}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

