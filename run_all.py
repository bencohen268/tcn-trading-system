#!/usr/bin/env python3
"""
Run the complete TCN trading system pipeline with all improvements.

This script runs all layers sequentially:
  1. Data loading & feature engineering (with advanced features)
  2. Dataset creation & validation
  3. Model training
  4. Out-of-sample inference
  5. Risk mapping
  6. Backtesting
  7. Benchmark comparison
  8. Edge analysis

Usage:
    python run_all.py

Author: TCN Trading System
Date: 2025-11-04
"""

import sys
import time
import subprocess
from pathlib import Path

def run_script(script_name: str, description: str) -> bool:
    """Run a Python script and handle errors."""
    print(f"\n{'='*80}")
    print(f"🚀 {description}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False,
            text=True
        )
        
        elapsed = time.time() - start_time
        print(f"\n✅ {description} completed in {elapsed:.1f}s")
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n❌ {description} failed after {elapsed:.1f}s")
        print(f"   Error: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  {description} interrupted by user")
        return False


def main():
    """Run the complete pipeline."""
    print("\n" + "="*80)
    print("🏆 TCN TRADING SYSTEM - COMPLETE PIPELINE WITH IMPROVEMENTS")
    print("="*80)
    print("\n📋 This will run:")
    print("   Layer 1: Data & Features (with 42 advanced features)")
    print("   Layer 2: Dataset Creation & Validation")
    print("   Layer 3: Model Training (larger TCN for more features)")
    print("   Layer 4-6: Inference, Risk Mapping & Backtesting")
    print("   Layer 7: Benchmark Comparison")
    print("   Layer 8: Edge Analysis")
    print("\n⚡ Using improvements:")
    print("   ✓ 42 features (12 basic + 30 advanced)")
    print("   ✓ Multi-day return labels (5-day horizon)")
    print("   ✓ Larger TCN architecture")
    print("   ✓ Benchmark strategies")
    print("   ✓ Comprehensive edge diagnostics")
    print("\n" + "="*80)
    
    start_total = time.time()
    
    # Layer 1: Data & Features
    if not run_script("run_layer1.py", "LAYER 1: Data & Feature Engineering"):
        print("\n❌ Pipeline failed at Layer 1")
        return 1
    
    # Layer 2: Dataset Creation
    if not run_script("run_layer2.py", "LAYER 2: Dataset Creation & Validation"):
        print("\n❌ Pipeline failed at Layer 2")
        return 1
    
    # Layer 3: Model Training
    if not run_script("run_layer3.py", "LAYER 3: Model Training"):
        print("\n❌ Pipeline failed at Layer 3")
        return 1
    
    # Layers 4-6: Backtest
    if not run_script("run_backtest.py", "LAYERS 4-6: Inference, Risk Mapping & Backtesting"):
        print("\n❌ Pipeline failed at Backtest")
        return 1
    
    # Layer 7: Benchmarks
    print("\n" + "="*80)
    print("📊 RUNNING BENCHMARK COMPARISON")
    print("="*80)
    print("\nComparing ML strategy against:")
    print("  • Buy & Hold")
    print("  • Moving Average Crossover")
    print("  • Momentum Strategy")
    print("  • Mean Reversion")
    print()
    
    if not run_script("run_benchmarks.py", "LAYER 7: Benchmark Comparison"):
        print("\n⚠️  Benchmark comparison failed (non-critical)")
        print("   Continuing to edge analysis...")
    
    # Layer 8: Edge Analysis
    print("\n" + "="*80)
    print("🔍 RUNNING EDGE ANALYSIS")
    print("="*80)
    print("\nGenerating comprehensive edge diagnostics:")
    print("  • Rolling performance metrics")
    print("  • Hit rate by prediction confidence")
    print("  • Return distribution by signal")
    print("  • Sharpe ratio evolution")
    print("  • Transaction cost impact")
    print()
    
    if not run_script("run_edge_analysis.py", "LAYER 8: Edge Analysis"):
        print("\n⚠️  Edge analysis failed (non-critical)")
    
    # Summary
    elapsed_total = time.time() - start_total
    
    print("\n" + "="*80)
    print("🎉 PIPELINE COMPLETE")
    print("="*80)
    print(f"\n⏱️  Total time: {elapsed_total/60:.1f} minutes")
    print("\n📂 Results saved to:")
    print("   • results/figures/layer1/ - Data diagnostics")
    print("   • results/figures/layer3/ - Training curves")
    print("   • results/figures/backtest/ - Performance plots")
    print("   • results/figures/benchmarks/ - Strategy comparison")
    print("   • results/figures/edge_analysis/ - Edge diagnostics")
    print("\n📊 Key files to review:")
    print("   1. results/figures/benchmarks/comparison.png")
    print("      → Shows if ML beats simple strategies")
    print()
    print("   2. results/figures/edge_analysis/edge_summary.png")
    print("      → Shows where edge comes from (if any)")
    print()
    print("   3. results/figures/backtest/equity_curve.png")
    print("      → Your system's P&L")
    print()
    print("   4. results/figures/layer3/training_curves.png")
    print("      → Model learning (check for overfitting)")
    print()
    print("\n" + "="*80)
    print("🎯 NEXT STEPS:")
    print("="*80)
    print("\n1. Review the benchmark comparison:")
    print("   → Does ML beat Buy & Hold?")
    print("   → What's the Sharpe ratio?")
    print()
    print("2. Check the edge analysis:")
    print("   → Is hit rate > 50%?")
    print("   → Are high-confidence predictions better?")
    print("   → Is performance consistent over time?")
    print()
    print("3. Look at training curves:")
    print("   → Is the model overfitting?")
    print("   → Is validation AUC > 0.52?")
    print()
    print("4. If no edge found:")
    print("   → Try different label types (edit config/params.yaml)")
    print("   → Try volatility prediction instead of returns")
    print("   → Consider intraday data or different assets")
    print()
    print("5. If edge IS found:")
    print("   → Validate with walk-forward testing")
    print("   → Check for data leakage")
    print("   → Test on other time periods")
    print()
    print("="*80 + "\n")
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
