#!/usr/bin/env python3
"""
Master runner: Execute all layers in sequence.

This script runs the complete pipeline:
1. Layer 1: Data & Feature Inspection
2. Layer 2: Windowing & Dataset Builder
3. Layer 3: Model Training
4. Layers 4-6: Inference, Risk Map, Backtest

Use this for a complete end-to-end run.
"""

import sys
import subprocess
from pathlib import Path


def run_layer(script_name: str, layer_name: str):
    """Run a layer script."""
    print("\n" + "="*80)
    print(f"RUNNING {layer_name}")
    print("="*80 + "\n")
    
    result = subprocess.run([sys.executable, script_name], cwd=Path.cwd())
    
    if result.returncode != 0:
        print(f"\n❌ {layer_name} failed with exit code {result.returncode}")
        print("Fix errors before proceeding to next layer.")
        sys.exit(result.returncode)
    
    print(f"\n✓ {layer_name} completed successfully\n")


def main():
    """Run all layers in sequence."""
    
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                                                                      ║
    ║               TCN TRADING SYSTEM - FULL PIPELINE                     ║
    ║                                                                      ║
    ║  This will run all layers sequentially:                             ║
    ║    Layer 1: Data & Feature Inspection                               ║
    ║    Layer 2: Windowing & Dataset Builder                             ║
    ║    Layer 3: Model Training                                          ║
    ║    Layers 4-6: Inference, Risk Map, Backtest                        ║
    ║                                                                      ║
    ║  Time estimate: 10-30 minutes (depending on data size)              ║
    ║                                                                      ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    response = input("Continue? [y/N]: ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    # Run each layer
    layers = [
        ("run_layer1.py", "Layer 1: Data & Feature Inspection"),
        ("run_layer2.py", "Layer 2: Windowing & Dataset Builder"),
        ("run_layer3.py", "Layer 3: Model Training"),
        ("run_backtest.py", "Layers 4-6: Inference, Risk Map, Backtest"),
    ]
    
    for script, name in layers:
        run_layer(script, name)
    
    # Final summary
    print("\n" + "="*80)
    print("🎉 ALL LAYERS COMPLETE!")
    print("="*80)
    print("\nResults saved in:")
    print("  📊 Figures: results/figures/")
    print("  📈 Models: models/checkpoints/")
    print("  💾 Data: data/processed/")
    print("  📋 Backtests: results/backtests/")
    print("\nNext steps:")
    print("  1. Review all diagnostic plots")
    print("  2. Analyze backtest performance")
    print("  3. Consider walk-forward testing (Layer 7)")
    print("  4. Iterate on features/model/risk-map as needed")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

