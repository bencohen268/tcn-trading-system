#!/usr/bin/env python3
"""
Run benchmark strategies for comparison.

Tests if ML model actually adds value over simple strategies.
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib.pyplot as plt

# Project imports
from utils import load_config, get_paths
from strategy.benchmarks import run_all_benchmarks, compare_to_benchmarks


def main():
    """Run all benchmark strategies."""
    
    print("\n" + "="*70)
    print("BENCHMARK STRATEGIES")
    print("="*70 + "\n")
    
    # Load configuration
    config = load_config()
    paths = get_paths()
    
    symbol = config['data']['symbol']
    frequency = config['data']['frequency']
    
    # Load test data
    print("[1/3] Loading test data...")
    test_file = paths['processed_data'] / f"{symbol}_{frequency}_test.parquet"
    
    if not test_file.exists():
        print(f"Error: {test_file} not found")
        print("Run run_layer1.py first to generate processed data")
        return
    
    test_df = pd.read_parquet(test_file)
    prices = test_df['raw_close']
    
    print(f"  Test period: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"  Total bars: {len(prices)}")
    
    # Run all benchmarks
    print("\n[2/3] Running benchmark strategies...")
    
    benchmarks_df = run_all_benchmarks(prices)
    
    print("\n" + "="*70)
    print("BENCHMARK RESULTS")
    print("="*70)
    print("\n" + benchmarks_df.to_string(index=False))
    print("\n" + "="*70)
    
    # Load ML results if available
    print("\n[3/3] Comparing to ML strategy...")
    
    backtest_file = paths['backtests'] / f"{symbol}_{frequency}_backtest_results.csv"
    
    if backtest_file.exists():
        # Read ML backtest results
        ml_results = pd.read_csv(backtest_file)
        
        # Calculate ML metrics
        equity_values = ml_results['equity'].values
        total_return = (equity_values[-1] - equity_values[0]) / equity_values[0]
        
        returns = pd.Series(equity_values).pct_change()
        sharpe = returns.mean() / returns.std() * (252 ** 0.5) if returns.std() > 0 else 0
        
        cummax = pd.Series(equity_values).expanding().max()
        drawdown = (pd.Series(equity_values) - cummax) / cummax
        max_dd = drawdown.min()
        
        ml_result = {
            'sharpe_ratio': sharpe,
            'total_return': total_return,
            'max_drawdown': max_dd,
        }
        
        # Compare
        comparison = compare_to_benchmarks(ml_result, prices)
        print(comparison)
        
    else:
        print(f"\nML backtest results not found at {backtest_file}")
        print("Run run_backtest.py first to generate ML results")
    
    # Save results
    output_file = paths['results'] / 'benchmarks' / f"{symbol}_{frequency}_benchmarks.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    benchmarks_df.to_csv(output_file, index=False)
    print(f"\nBenchmark results saved to: {output_file}")
    
    print("\n" + "="*70)
    print("INSIGHTS:")
    print("="*70)
    print("\n1. If ML Sharpe < Buy & Hold Sharpe:")
    print("   → Model has no edge, just use buy & hold")
    print("\n2. If ML Sharpe < Best benchmark Sharpe:")
    print("   → Model adds no value, use simpler strategy")
    print("\n3. If ML Sharpe > All benchmarks:")
    print("   → Model might have edge, but verify:")
    print("     - Walk-forward test shows consistency?")
    print("     - No data leakage?")
    print("     - Makes intuitive sense?")
    print("\n4. Typical benchmark Sharpe ranges:")
    print("   - Buy & Hold SPY: 0.3 - 0.8")
    print("   - MA Crossover: -0.2 - 0.5")
    print("   - Momentum: -0.5 - 0.8 (high variance)")
    print("   - Mean Reversion: -0.3 - 0.4")
    print("\n5. To beat benchmarks:")
    print("   - Need Sharpe > 0.5 minimum")
    print("   - Need consistency across time periods")
    print("   - Need statistical significance")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

