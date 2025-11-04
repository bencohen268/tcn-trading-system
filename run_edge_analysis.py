#!/usr/bin/env python3
"""
Edge Analysis: Comprehensive diagnostics to detect and measure trading edge.

This script analyzes the ML trading system to determine if it has genuine edge:
  • Hit rate by prediction confidence
  • Return distribution by signal strength
  • Rolling performance metrics
  • Sharpe ratio evolution over time
  • Transaction cost sensitivity
  • Calibration analysis

Usage:
    python run_edge_analysis.py

Author: TCN Trading System
Date: 2025-11-04
"""

import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Project imports
from utils import load_config, get_paths
from data import load_processed_data
from strategy import Backtester


def analyze_hit_rate_by_confidence(predictions, actuals, n_bins=10):
    """Analyze hit rate across prediction confidence bins."""
    df = pd.DataFrame({
        'pred': predictions,
        'actual': actuals
    }).dropna()
    
    # Create confidence bins
    df['confidence'] = np.abs(df['pred'] - 0.5)
    df['bin'] = pd.qcut(df['confidence'], q=n_bins, labels=False, duplicates='drop')
    
    # Calculate hit rate per bin
    results = []
    for bin_id in sorted(df['bin'].unique()):
        bin_data = df[df['bin'] == bin_id]
        
        # Hit rate (prediction matches actual direction)
        predicted_direction = (bin_data['pred'] > 0.5).astype(int)
        hit_rate = (predicted_direction == bin_data['actual']).mean()
        
        # Average confidence in this bin
        avg_confidence = bin_data['confidence'].mean()
        avg_pred = bin_data['pred'].mean()
        n_samples = len(bin_data)
        
        results.append({
            'bin': bin_id,
            'confidence': avg_confidence,
            'avg_prediction': avg_pred,
            'hit_rate': hit_rate,
            'n_samples': n_samples,
        })
    
    return pd.DataFrame(results)


def analyze_returns_by_signal(predictions, returns, n_bins=5):
    """Analyze actual returns grouped by prediction strength."""
    df = pd.DataFrame({
        'pred': predictions,
        'ret': returns
    }).dropna()
    
    # Create signal strength bins
    df['signal'] = df['pred'] - 0.5  # -0.5 to +0.5
    df['bin'] = pd.qcut(df['signal'], q=n_bins, labels=False, duplicates='drop')
    
    # Analyze returns per bin
    results = []
    for bin_id in sorted(df['bin'].unique()):
        bin_data = df[df['bin'] == bin_id]
        
        results.append({
            'bin': bin_id,
            'avg_signal': bin_data['signal'].mean(),
            'avg_return': bin_data['ret'].mean(),
            'std_return': bin_data['ret'].std(),
            'sharpe': bin_data['ret'].mean() / (bin_data['ret'].std() + 1e-8) * np.sqrt(252),
            'n_samples': len(bin_data),
        })
    
    return pd.DataFrame(results)


def rolling_performance(equity_curve, window_days=60):
    """Calculate rolling performance metrics."""
    df = equity_curve.copy()
    df['returns'] = df['equity'].pct_change()
    
    # Rolling metrics
    df['rolling_sharpe'] = df['returns'].rolling(window_days).apply(
        lambda x: x.mean() / (x.std() + 1e-8) * np.sqrt(252)
    )
    df['rolling_returns'] = df['returns'].rolling(window_days).sum()
    df['rolling_volatility'] = df['returns'].rolling(window_days).std() * np.sqrt(252)
    
    # Cumulative max and drawdown
    df['cum_max'] = df['equity'].cummax()
    df['drawdown'] = (df['equity'] - df['cum_max']) / df['cum_max']
    
    return df


def transaction_cost_sensitivity(backtest_results, base_cost_bps):
    """Analyze how edge degrades with transaction costs."""
    costs_to_test = np.linspace(0, base_cost_bps * 3, 20)
    
    # Get base data
    positions = backtest_results['position']
    returns = backtest_results['returns']
    
    results = []
    for cost_bps in costs_to_test:
        # Recalculate P&L with different costs
        position_changes = positions.diff().abs()
        cost_drag = position_changes * (cost_bps / 10000)
        
        # Net returns
        net_returns = returns - cost_drag
        
        # Metrics
        total_return = net_returns.sum()
        sharpe = net_returns.mean() / (net_returns.std() + 1e-8) * np.sqrt(252)
        
        results.append({
            'cost_bps': cost_bps,
            'total_return': total_return,
            'sharpe': sharpe,
        })
    
    return pd.DataFrame(results)


def plot_edge_analysis(hit_rate_df, returns_df, rolling_df, cost_df, save_dir):
    """Create comprehensive edge analysis plots."""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Hit Rate by Confidence
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(hit_rate_df['confidence'], hit_rate_df['hit_rate'], alpha=0.7, color='steelblue')
    ax1.axhline(y=0.5, color='red', linestyle='--', label='Random (50%)')
    ax1.set_xlabel('Prediction Confidence')
    ax1.set_ylabel('Hit Rate')
    ax1.set_title('Hit Rate by Confidence\n(Higher confidence = better predictions?)')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. Sample count per confidence bin
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(hit_rate_df['confidence'], hit_rate_df['n_samples'], alpha=0.7, color='coral')
    ax2.set_xlabel('Prediction Confidence')
    ax2.set_ylabel('Number of Samples')
    ax2.set_title('Sample Distribution\n(Are high-confidence predictions rare?)')
    ax2.grid(alpha=0.3)
    
    # 3. Returns by Signal Strength
    ax3 = fig.add_subplot(gs[0, 2])
    colors = ['red' if x < 0 else 'green' for x in returns_df['avg_signal']]
    ax3.bar(returns_df['avg_signal'], returns_df['avg_return'], alpha=0.7, color=colors)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('Signal Strength (Bearish ← → Bullish)')
    ax3.set_ylabel('Average Return')
    ax3.set_title('Returns by Signal\n(Do bearish signals → negative returns?)')
    ax3.grid(alpha=0.3)
    
    # 4. Sharpe by Signal Strength
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.bar(returns_df['avg_signal'], returns_df['sharpe'], alpha=0.7, color='purple')
    ax4.axhline(y=0, color='red', linestyle='--')
    ax4.set_xlabel('Signal Strength')
    ax4.set_ylabel('Sharpe Ratio')
    ax4.set_title('Risk-Adjusted Returns by Signal\n(Is edge consistent?)')
    ax4.grid(alpha=0.3)
    
    # 5. Rolling Sharpe Ratio
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(rolling_df.index, rolling_df['rolling_sharpe'], linewidth=2, color='darkblue')
    ax5.axhline(y=0, color='red', linestyle='--', label='Zero')
    ax5.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Sharpe=1')
    ax5.set_xlabel('Date')
    ax5.set_ylabel('Rolling Sharpe (60d)')
    ax5.set_title('Sharpe Ratio Over Time\n(Is performance consistent?)')
    ax5.legend()
    ax5.grid(alpha=0.3)
    
    # 6. Drawdown
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.fill_between(rolling_df.index, 0, rolling_df['drawdown'] * 100, 
                      alpha=0.7, color='red', label='Drawdown')
    ax6.set_xlabel('Date')
    ax6.set_ylabel('Drawdown (%)')
    ax6.set_title('Drawdown Over Time\n(Risk management quality)')
    ax6.legend()
    ax6.grid(alpha=0.3)
    
    # 7. Transaction Cost Sensitivity - Total Return
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.plot(cost_df['cost_bps'], cost_df['total_return'] * 100, linewidth=2, color='darkgreen')
    ax7.axhline(y=0, color='red', linestyle='--')
    ax7.set_xlabel('Transaction Cost (bps per side)')
    ax7.set_ylabel('Total Return (%)')
    ax7.set_title('Return vs Transaction Costs\n(Edge robustness)')
    ax7.grid(alpha=0.3)
    
    # 8. Transaction Cost Sensitivity - Sharpe
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.plot(cost_df['cost_bps'], cost_df['sharpe'], linewidth=2, color='purple')
    ax8.axhline(y=0, color='red', linestyle='--')
    ax8.set_xlabel('Transaction Cost (bps per side)')
    ax8.set_ylabel('Sharpe Ratio')
    ax8.set_title('Sharpe vs Transaction Costs\n(When does edge disappear?)')
    ax8.grid(alpha=0.3)
    
    # 9. Equity Curve with confidence bands
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.plot(rolling_df.index, rolling_df['equity'], linewidth=2, color='darkblue', label='Equity')
    ax9.plot(rolling_df.index, rolling_df['cum_max'], linewidth=1, linestyle='--', 
             color='green', alpha=0.5, label='All-time high')
    ax9.set_xlabel('Date')
    ax9.set_ylabel('Equity')
    ax9.set_title('Equity Curve\n(Overall performance)')
    ax9.legend()
    ax9.grid(alpha=0.3)
    
    # Overall title
    fig.suptitle('EDGE ANALYSIS: Does the System Have Predictive Power?', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Save
    save_path = save_dir / 'edge_summary.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def print_edge_verdict(hit_rate_df, returns_df, rolling_df, cost_df):
    """Print a clear verdict on whether the system has edge."""
    print("\n" + "="*80)
    print("🎯 EDGE ANALYSIS VERDICT")
    print("="*80)
    
    # 1. Overall hit rate
    overall_hit_rate = hit_rate_df['hit_rate'].mean()
    print(f"\n1. Overall Hit Rate: {overall_hit_rate:.2%}")
    if overall_hit_rate > 0.52:
        print("   ✅ Above random (good sign)")
    elif overall_hit_rate > 0.50:
        print("   ⚠️  Slightly above random (marginal)")
    else:
        print("   ❌ At or below random (no edge detected)")
    
    # 2. Confidence calibration
    high_conf_data = hit_rate_df[hit_rate_df['confidence'] > hit_rate_df['confidence'].median()]
    high_conf_hit_rate = high_conf_data['hit_rate'].mean()
    print(f"\n2. High-Confidence Hit Rate: {high_conf_hit_rate:.2%}")
    if high_conf_hit_rate > overall_hit_rate + 0.03:
        print("   ✅ High confidence predictions are better (well-calibrated)")
    elif high_conf_hit_rate > overall_hit_rate:
        print("   ⚠️  Slightly better (weak calibration)")
    else:
        print("   ❌ No better than average (poor calibration)")
    
    # 3. Returns by signal
    strong_long = returns_df[returns_df['avg_signal'] > 0]['avg_return'].mean()
    strong_short = returns_df[returns_df['avg_signal'] < 0]['avg_return'].mean()
    print(f"\n3. Returns by Signal Direction:")
    print(f"   Bullish signals: {strong_long:.4f} avg return")
    print(f"   Bearish signals: {strong_short:.4f} avg return")
    if strong_long > 0 and strong_short < 0:
        print("   ✅ Signals align with returns (good)")
    elif strong_long > 0:
        print("   ⚠️  Longs work, shorts don't")
    elif strong_short < 0:
        print("   ⚠️  Shorts work, longs don't")
    else:
        print("   ❌ Signals don't align with returns")
    
    # 4. Sharpe ratio
    final_sharpe = rolling_df['rolling_sharpe'].iloc[-1]
    median_sharpe = rolling_df['rolling_sharpe'].median()
    print(f"\n4. Sharpe Ratio:")
    print(f"   Current (60d): {final_sharpe:.2f}")
    print(f"   Median (60d): {median_sharpe:.2f}")
    if median_sharpe > 0.5:
        print("   ✅ Decent risk-adjusted returns")
    elif median_sharpe > 0:
        print("   ⚠️  Positive but weak")
    else:
        print("   ❌ Negative or zero Sharpe")
    
    # 5. Transaction cost robustness
    zero_cost_return = cost_df.iloc[0]['total_return']
    double_cost_return = cost_df[cost_df['cost_bps'] <= cost_df['cost_bps'].max() / 2]['total_return'].iloc[-1]
    print(f"\n5. Transaction Cost Sensitivity:")
    print(f"   Zero costs: {zero_cost_return:.2%} return")
    print(f"   2x costs: {double_cost_return:.2%} return")
    if double_cost_return > 0.05:
        print("   ✅ Edge survives realistic costs")
    elif double_cost_return > 0:
        print("   ⚠️  Edge barely survives costs")
    else:
        print("   ❌ Edge disappears with realistic costs")
    
    # Final verdict
    print("\n" + "="*80)
    print("FINAL VERDICT:")
    print("="*80)
    
    score = 0
    if overall_hit_rate > 0.52: score += 2
    elif overall_hit_rate > 0.50: score += 1
    
    if high_conf_hit_rate > overall_hit_rate + 0.03: score += 2
    elif high_conf_hit_rate > overall_hit_rate: score += 1
    
    if strong_long > 0 and strong_short < 0: score += 2
    elif strong_long > 0 or strong_short < 0: score += 1
    
    if median_sharpe > 0.5: score += 2
    elif median_sharpe > 0: score += 1
    
    if double_cost_return > 0.05: score += 2
    elif double_cost_return > 0: score += 1
    
    print(f"\nScore: {score}/10")
    
    if score >= 8:
        print("\n🎉 STRONG EDGE DETECTED")
        print("   The system shows consistent, robust predictive power.")
        print("   Next steps: Validate with walk-forward testing and live paper trading.")
    elif score >= 5:
        print("\n⚠️  MARGINAL EDGE DETECTED")
        print("   The system shows some predictive power but it's weak.")
        print("   Next steps: Try different label types, more features, or different assets.")
    else:
        print("\n❌ NO EDGE DETECTED")
        print("   The system does not demonstrate reliable predictive power.")
        print("   Next steps:")
        print("   • Try volatility prediction instead of direction")
        print("   • Use intraday data (more signal)")
        print("   • Try different assets (less efficient markets)")
        print("   • Accept that daily SPY direction is just too hard")
    
    print("\n" + "="*80 + "\n")


def main():
    """Run edge analysis."""
    print("\n" + "="*80)
    print("🔍 EDGE ANALYSIS")
    print("="*80)
    print("\nAnalyzing trading system for genuine predictive power...\n")
    
    # Load config
    config = load_config()
    paths = get_paths()
    
    # Create output directory
    output_dir = Path(paths['figures']) / 'edge_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load backtest results
    backtest_path = Path(paths['backtests']) / f"{config['data']['symbol']}_{config['data']['frequency']}_backtest_results.csv"
    
    if not backtest_path.exists():
        print(f"❌ Backtest results not found: {backtest_path}")
        print("   Run 'python run_backtest.py' first")
        return 1
    
    print(f"Loading backtest results from {backtest_path}...")
    backtest_df = pd.read_csv(backtest_path, index_col=0, parse_dates=True)
    
    # Calculate returns from price if not present
    if 'returns' not in backtest_df.columns:
        backtest_df['returns'] = backtest_df['price'].pct_change()
    
    # Extract key data
    predictions = backtest_df['probability'].values
    actuals = (backtest_df['returns'] > 0).astype(int).values
    returns = backtest_df['returns'].values
    
    # 1. Hit rate analysis
    print("\n[1/5] Analyzing hit rate by confidence...")
    hit_rate_df = analyze_hit_rate_by_confidence(predictions, actuals, n_bins=10)
    
    # 2. Returns by signal
    print("[2/5] Analyzing returns by signal strength...")
    returns_df = analyze_returns_by_signal(predictions, returns, n_bins=5)
    
    # 3. Rolling performance
    print("[3/5] Calculating rolling performance metrics...")
    rolling_df = rolling_performance(backtest_df, window_days=60)
    
    # 4. Transaction cost sensitivity
    print("[4/5] Analyzing transaction cost sensitivity...")
    base_cost = config['backtest']['commission_bps']
    cost_df = transaction_cost_sensitivity(backtest_df, base_cost)
    
    # 5. Create visualizations
    print("[5/5] Generating edge analysis plots...")
    plot_edge_analysis(hit_rate_df, returns_df, rolling_df, cost_df, output_dir)
    
    # Print verdict
    print_edge_verdict(hit_rate_df, returns_df, rolling_df, cost_df)
    
    # Save detailed results
    results_path = output_dir / 'edge_metrics.csv'
    summary = pd.DataFrame({
        'metric': [
            'overall_hit_rate',
            'high_conf_hit_rate',
            'final_sharpe',
            'median_sharpe',
            'max_drawdown',
        ],
        'value': [
            hit_rate_df['hit_rate'].mean(),
            hit_rate_df[hit_rate_df['confidence'] > hit_rate_df['confidence'].median()]['hit_rate'].mean(),
            rolling_df['rolling_sharpe'].iloc[-1],
            rolling_df['rolling_sharpe'].median(),
            rolling_df['drawdown'].min(),
        ]
    })
    summary.to_csv(results_path, index=False)
    print(f"Detailed metrics saved to {results_path}")
    
    print("\n✅ Edge analysis complete!")
    print(f"📊 Results saved to {output_dir}/")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

