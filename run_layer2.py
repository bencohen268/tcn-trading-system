#!/usr/bin/env python3
"""
Layer 2: Windowing & Dataset Builder

This script creates PyTorch datasets with sliding windows and verifies:
- No future leakage
- Correct sequence construction
- Proper temporal ordering

Success criteria:
- Windows are strictly causal (no lookahead)
- Sample counts match expectations
- No data overlap between train/val/test
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# Project imports
from utils import load_config, get_paths
from models import TCNDataset, load_processed_data_for_training


def visualize_sample(dataset: TCNDataset, idx: int = 0):
    """Visualize a single sample for inspection."""
    features, target, timestamp = dataset[idx]
    info = dataset.get_sample_info(idx)
    
    print(f"\nSample {idx} details:")
    print(f"  Dataset index: {info['dataset_idx']}")
    print(f"  Data index: {info['data_idx']}")
    print(f"  Window: {info['window_start']} to {info['window_end']}")
    print(f"  Target value: {info['target_value']}")
    print(f"\n  Features shape: {features.shape}")
    print(f"  Target shape: {target.shape}")
    print(f"\n  First 3 timesteps of features:")
    print(features[:3].numpy())


def check_temporal_splits(
    train_dataset: TCNDataset,
    val_dataset: TCNDataset,
    test_dataset: TCNDataset,
):
    """Verify that splits don't overlap in time."""
    print("\n" + "="*70)
    print("TEMPORAL SPLIT VERIFICATION")
    print("="*70)
    
    # Get last timestamp from each dataset
    train_end = train_dataset.timestamps[train_dataset.valid_indices[-1]]
    val_start = val_dataset.timestamps[val_dataset.valid_indices[0]]
    val_end = val_dataset.timestamps[val_dataset.valid_indices[-1]]
    test_start = test_dataset.timestamps[test_dataset.valid_indices[0]]
    
    print(f"\nTrain ends:   {train_end}")
    print(f"Val starts:   {val_start}")
    print(f"Val ends:     {val_end}")
    print(f"Test starts:  {test_start}")
    
    # Check no overlap
    train_val_gap = (val_start - train_end).days if hasattr((val_start - train_end), 'days') else 0
    val_test_gap = (test_start - val_end).days if hasattr((test_start - val_end), 'days') else 0
    
    print(f"\nTrain → Val gap:  {train_val_gap} periods")
    print(f"Val → Test gap:   {val_test_gap} periods")
    
    if train_end < val_start and val_end < test_start:
        print("\n✓ NO TEMPORAL OVERLAP - Splits are properly separated")
    else:
        print("\n✗ TEMPORAL OVERLAP DETECTED - Check date ranges in config")
    
    print("="*70)


def main():
    """Run Layer 2: Windowing & Dataset Builder."""
    
    print("\n" + "="*70)
    print("LAYER 2: WINDOWING & DATASET BUILDER")
    print("="*70 + "\n")
    
    # 1. Load configuration
    print("[1/4] Loading configuration...")
    config = load_config()
    paths = get_paths()
    
    symbol = config['data']['symbol']
    frequency = config['data']['frequency']
    seq_len = config['data']['seq_len']
    
    print(f"  Symbol: {symbol}")
    print(f"  Frequency: {frequency}")
    print(f"  Sequence length: {seq_len}")
    
    # 2. Load processed data and create datasets
    print("\n[2/4] Creating datasets from processed data...")
    
    train_dataset, val_dataset, test_dataset = load_processed_data_for_training(
        processed_data_dir=paths['processed_data'],
        symbol=symbol,
        frequency=frequency,
        seq_len=seq_len,
    )
    
    # 3. Verify no leakage
    print("\n[3/4] Running leakage checks...")
    
    print("\n--- Train Dataset ---")
    train_dataset.verify_no_leakage(n_samples=5)
    
    print("\n--- Validation Dataset ---")
    val_dataset.verify_no_leakage(n_samples=5)
    
    print("\n--- Test Dataset ---")
    test_dataset.verify_no_leakage(n_samples=5)
    
    # 4. Check temporal splits
    check_temporal_splits(train_dataset, val_dataset, test_dataset)
    
    # 5. Visualize samples
    print("\n[4/4] Sample visualization...")
    
    print("\n--- First training sample ---")
    visualize_sample(train_dataset, idx=0)
    
    print("\n--- Random training sample ---")
    random_idx = np.random.randint(0, len(train_dataset))
    visualize_sample(train_dataset, idx=random_idx)
    
    print("\n--- First validation sample ---")
    visualize_sample(val_dataset, idx=0)
    
    # Summary statistics
    print("\n" + "="*70)
    print("DATASET SUMMARY")
    print("="*70)
    
    print(f"\nTrain:")
    print(f"  Total bars: {len(train_dataset.timestamps)}")
    print(f"  Valid samples: {len(train_dataset)}")
    print(f"  Window size: {seq_len}")
    print(f"  Features: {train_dataset.n_features}")
    print(f"  Date range: {train_dataset.timestamps[0]} to {train_dataset.timestamps[-1]}")
    
    print(f"\nValidation:")
    print(f"  Total bars: {len(val_dataset.timestamps)}")
    print(f"  Valid samples: {len(val_dataset)}")
    print(f"  Date range: {val_dataset.timestamps[0]} to {val_dataset.timestamps[-1]}")
    
    print(f"\nTest:")
    print(f"  Total bars: {len(test_dataset.timestamps)}")
    print(f"  Valid samples: {len(test_dataset)}")
    print(f"  Date range: {test_dataset.timestamps[0]} to {test_dataset.timestamps[-1]}")
    
    # Label balance in samples
    print(f"\nLabel balance:")
    for name, dataset in [('Train', train_dataset), ('Val', val_dataset), ('Test', test_dataset)]:
        targets = []
        for i in range(len(dataset)):
            _, target, _ = dataset[i]
            targets.append(target.item())
        targets = np.array(targets)
        pct_up = (targets == 1).sum() / len(targets) * 100
        print(f"  {name}: {pct_up:.2f}% up labels")
    
    print("\n" + "="*70)
    print("LAYER 2 COMPLETE ✓")
    print("="*70)
    print("\nNext steps:")
    print("  1. All leakage checks should pass")
    print("  2. Sample counts should match expectations")
    print("  3. Temporal splits should not overlap")
    print("  4. If everything looks good, proceed to Layer 3 (model training)")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

