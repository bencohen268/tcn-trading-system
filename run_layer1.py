#!/usr/bin/env python3
"""
Layer 1: Data & Feature Inspection

This script validates that we're feeding the model something sane.

Success criteria:
- Features aren't flat or all super-correlated
- Label isn't 90% one class in an era
- No missing data or insane values

If bad: Adjust scalers, change feature definitions, add time-of-day, maybe resample.
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# Project imports
from utils import load_config, get_paths, set_seed
from data import load_raw_data, save_processed_data, split_by_date
from features import FeatureEngineer, create_binary_target, estimate_volatility
from visualization import generate_layer1_report


def main():
    """Run Layer 1: Data & Feature Inspection."""
    
    print("\n" + "="*70)
    print("LAYER 1: DATA & FEATURE INSPECTION")
    print("="*70 + "\n")
    
    # 1. Load configuration
    print("[1/8] Loading configuration...")
    config = load_config()
    paths = get_paths(create_if_missing=True)
    set_seed(config['experiment']['seed'])
    
    symbol = config['data']['symbol']
    frequency = config['data']['frequency']
    
    print(f"  Symbol: {symbol}")
    print(f"  Frequency: {frequency}")
    print(f"  Train: {config['data']['train_start']} to {config['data']['train_end']}")
    print(f"  Val:   {config['data']['val_start']} to {config['data']['val_end']}")
    print(f"  Test:  {config['data']['test_start']} to {config['data']['test_end']}")
    
    # 2. Load raw data
    print("\n[2/8] Loading raw data...")
    
    start_date = config['data']['train_start']
    end_date = config['data']['test_end']
    
    # Map frequency to Yahoo Finance interval codes
    interval_map = {
        '1m': '1m',
        '5m': '5m',
        '15m': '15m',
        '1h': '1h',
        'daily': '1d',
    }
    interval = interval_map.get(frequency, '1d')
    
    raw_df = load_raw_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        source='yahoo',
        cache_dir=paths['raw_data'],
    )
    
    print(f"  Raw data shape: {raw_df.shape}")
    print(f"  Date range: {raw_df.index[0]} to {raw_df.index[-1]}")
    
    # Quick sanity checks
    missing = raw_df.isnull().sum().sum()
    negative_prices = (raw_df[['open', 'high', 'low', 'close']] <= 0).sum().sum()
    invalid_hl = (raw_df['high'] < raw_df['low']).sum()
    
    print(f"  Missing values: {missing}")
    print(f"  Negative prices: {negative_prices}")
    print(f"  Invalid high/low: {invalid_hl}")
    
    if negative_prices > 0 or invalid_hl > 0:
        print("  ⚠️  WARNING: Data quality issues detected!")
    
    # 3. Split data
    print("\n[3/8] Splitting data...")
    
    splits = split_by_date(
        raw_df,
        train_start=config['data']['train_start'],
        train_end=config['data']['train_end'],
        val_start=config['data']['val_start'],
        val_end=config['data']['val_end'],
        test_start=config['data']['test_start'],
        test_end=config['data']['test_end'],
    )
    
    # 4. Create features
    print("\n[4/8] Engineering features...")
    
    feature_engineer = FeatureEngineer(
        vol_window=config['features']['vol_window'],
        sma_fast_period=config['features']['sma_fast_period'],
        sma_slow_period=config['features']['sma_slow_period'],
        volume_window=config['features']['volume_window'],
        normalization=config['features']['normalization'],
        rolling_norm=config['features']['rolling_norm'],
    )
    
    # Fit on training data only
    feature_engineer.fit(splits['train'])
    
    # Transform all splits
    features = {}
    for split_name, split_df in splits.items():
        features[split_name] = feature_engineer.transform(split_df)
        print(f"  {split_name}: {features[split_name].shape}")
    
    # 5. Create labels
    print("\n[5/8] Creating labels...")
    
    labels = {}
    for split_name, split_df in splits.items():
        labels[split_name] = create_binary_target(
            split_df,
            threshold=config['data']['label_threshold']
        )
        
        # Summary
        label_clean = labels[split_name].dropna()
        pct_up = (label_clean == 1).sum() / len(label_clean) * 100
        print(f"  {split_name}: {pct_up:.2f}% up labels ({len(label_clean)} samples)")
    
    # 6. Estimate volatility
    print("\n[6/8] Estimating volatility...")
    
    volatility = {}
    for split_name, split_df in splits.items():
        volatility[split_name] = estimate_volatility(
            split_df,
            method=config['risk_map']['vol_method'],
            window=config['risk_map']['vol_window'],
        )
        vol_mean = volatility[split_name].mean()
        vol_std = volatility[split_name].std()
        print(f"  {split_name}: mean={vol_mean:.4f}, std={vol_std:.4f}")
    
    # 7. Generate diagnostic reports
    print("\n[7/8] Generating diagnostic reports...")
    
    for split_name in ['train', 'val', 'test']:
        print(f"\n  Generating {split_name} diagnostics...")
        generate_layer1_report(
            features_df=features[split_name],
            target=labels[split_name],
            output_dir=paths['figures'],
            prefix=f'layer1_{split_name}',
        )
    
    # Cross-split comparison
    print("\n  Cross-split feature comparison:")
    comparison_df = pd.DataFrame({
        'train_mean': features['train'].mean(),
        'val_mean': features['val'].mean(),
        'test_mean': features['test'].mean(),
        'train_std': features['train'].std(),
        'val_std': features['val'].std(),
        'test_std': features['test'].std(),
    })
    print(comparison_df)
    
    # 8. Save processed data
    print("\n[8/8] Saving processed data...")
    
    for split_name in ['train', 'val', 'test']:
        # Combine everything into one DataFrame
        processed_df = features[split_name].copy()
        processed_df['target'] = labels[split_name]
        processed_df['volatility'] = volatility[split_name]
        
        # Also keep original OHLCV for reference
        for col in ['open', 'high', 'low', 'close', 'volume']:
            processed_df[f'raw_{col}'] = splits[split_name][col]
        
        # Save to parquet
        save_processed_data(
            df=processed_df,
            name=f"{symbol}_{frequency}_{split_name}",
            output_dir=paths['processed_data'],
            format='parquet',
        )
    
    # Final summary
    print("\n" + "="*70)
    print("LAYER 1 COMPLETE ✓")
    print("="*70)
    print("\nNext steps:")
    print("  1. Review diagnostic plots in:", paths['figures'])
    print("  2. Check for red flags:")
    print("     - Highly correlated features (>0.95)")
    print("     - Extreme label imbalance (>70% one class)")
    print("     - Missing data patterns")
    print("     - Non-stationary features (strong drift)")
    print("  3. If everything looks good, proceed to Layer 2 (windowing)")
    print("\nProcessed data saved to:", paths['processed_data'])
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

