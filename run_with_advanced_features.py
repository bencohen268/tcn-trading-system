#!/usr/bin/env python3
"""
Run TCN system with advanced features.

This script uses the enhanced feature set (~30 features instead of 12)
and tests alternative labeling strategies.
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
from features import FeatureEngineer, estimate_volatility
from features.advanced_features import AdvancedFeatureEngineer, add_calendar_features
from features.alternative_labels import (
    create_multiday_return_label,
    create_regime_labels,
    create_volatility_labels,
    analyze_label_quality,
)
from visualization import generate_layer1_report


def main():
    """Run with advanced features and alternative labels."""
    
    print("\n" + "="*70)
    print("TCN SYSTEM WITH ADVANCED FEATURES")
    print("="*70 + "\n")
    
    # 1. Load configuration
    print("[1/7] Loading configuration...")
    config = load_config()
    paths = get_paths()
    set_seed(config['experiment']['seed'])
    
    symbol = config['data']['symbol']
    frequency = config['data']['frequency']
    
    print(f"  Symbol: {symbol}")
    print(f"  Frequency: {frequency}")
    print(f"  Using ADVANCED feature set (~30 features)")
    
    # 2. Load raw data
    print("\n[2/7] Loading raw data...")
    
    start_date = config['data']['train_start']
    end_date = config['data']['test_end']
    interval_map = {'1m': '1m', '5m': '5m', '15m': '15m', '1h': '1h', 'daily': '1d'}
    interval = interval_map.get(frequency, '1d')
    
    raw_df = load_raw_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        source='yahoo',
        cache_dir=paths['raw_data'],
    )
    
    print(f"  Loaded {len(raw_df)} bars")
    
    # 3. Split data
    print("\n[3/7] Splitting data...")
    
    splits = split_by_date(
        raw_df,
        train_start=config['data']['train_start'],
        train_end=config['data']['train_end'],
        val_start=config['data']['val_start'],
        val_end=config['data']['val_end'],
        test_start=config['data']['test_start'],
        test_end=config['data']['test_end'],
    )
    
    # 4. Create BASIC features (original 12)
    print("\n[4/7] Creating basic features...")
    
    basic_engineer = FeatureEngineer(
        vol_window=config['features']['vol_window'],
        sma_fast_period=config['features']['sma_fast_period'],
        sma_slow_period=config['features']['sma_slow_period'],
        volume_window=config['features']['volume_window'],
        normalization=config['features']['normalization'],
        rolling_norm=False,
    )
    
    # Fit on train
    basic_engineer.fit(splits['train'])
    
    # Transform all splits
    basic_features = {}
    for split_name, split_df in splits.items():
        basic_features[split_name] = basic_engineer.transform(split_df)
        print(f"  {split_name}: {basic_features[split_name].shape}")
    
    # 5. Create ADVANCED features
    print("\n[5/7] Creating advanced features...")
    
    advanced_engineer = AdvancedFeatureEngineer(
        regime_window=20,
        vol_windows=[5, 10, 20, 60],
        momentum_windows=[5, 10, 20],
    )
    
    advanced_features = {}
    for split_name, split_df in splits.items():
        # Create advanced features
        adv_feats = advanced_engineer.create_features(split_df)
        
        # Add calendar features
        cal_feats = add_calendar_features(split_df)
        
        # Combine: basic + advanced + calendar
        combined = pd.concat([
            basic_features[split_name],
            adv_feats,
            cal_feats,
        ], axis=1)
        
        advanced_features[split_name] = combined
        print(f"  {split_name}: {combined.shape} ({combined.shape[1]} total features)")
    
    # 6. Create alternative labels
    print("\n[6/7] Creating alternative labels...")
    
    label_strategies = {}
    
    # A. Next-bar direction (baseline)
    from features import create_binary_target
    label_strategies['next_bar'] = {}
    for split_name, split_df in splits.items():
        label_strategies['next_bar'][split_name] = create_binary_target(split_df)
    
    # B. Multi-day return (5-day)
    label_strategies['multiday_5'] = {}
    for split_name, split_df in splits.items():
        label_strategies['multiday_5'][split_name] = create_multiday_return_label(
            split_df, horizon=5, threshold=0.005
        )
    
    # C. Regime classification
    label_strategies['regime'] = {}
    for split_name, split_df in splits.items():
        label_strategies['regime'][split_name] = create_regime_labels(split_df)
    
    # D. Volatility prediction
    label_strategies['volatility'] = {}
    for split_name, split_df in splits.items():
        label_strategies['volatility'][split_name] = create_volatility_labels(
            split_df, horizon=5
        )
    
    # Analyze label quality
    print("\n  Label Quality Analysis:")
    for label_name, labels_dict in label_strategies.items():
        train_label = labels_dict['train']
        quality = analyze_label_quality(train_label, name=label_name)
        print(f"\n  {label_name}:")
        print(f"    Samples: {quality['n_samples']}")
        print(f"    Classes: {quality['n_classes']}")
        print(f"    Balance: {quality['balance']}")
        print(f"    Autocorr: {quality['autocorrelation']:.3f}")
    
    # 7. Save all versions
    print("\n[7/7] Saving processed data...")
    
    for split_name in ['train', 'val', 'test']:
        # Save basic version (12 features)
        basic_df = basic_features[split_name].copy()
        basic_df['target'] = label_strategies['next_bar'][split_name]
        basic_df['volatility'] = estimate_volatility(splits[split_name], method='std', window=20)
        
        # Add raw OHLCV
        for col in ['open', 'high', 'low', 'close', 'volume']:
            basic_df[f'raw_{col}'] = splits[split_name][col]
        
        save_processed_data(
            df=basic_df,
            name=f"{symbol}_{frequency}_{split_name}_basic",
            output_dir=paths['processed_data'],
            format='parquet',
        )
        
        # Save advanced version (~30 features)
        advanced_df = advanced_features[split_name].copy()
        advanced_df['target'] = label_strategies['next_bar'][split_name]
        advanced_df['target_multiday'] = label_strategies['multiday_5'][split_name]
        advanced_df['target_regime'] = label_strategies['regime'][split_name]
        advanced_df['target_volatility'] = label_strategies['volatility'][split_name]
        advanced_df['volatility'] = estimate_volatility(splits[split_name], method='std', window=20)
        
        # Add raw OHLCV
        for col in ['open', 'high', 'low', 'close', 'volume']:
            advanced_df[f'raw_{col}'] = splits[split_name][col]
        
        save_processed_data(
            df=advanced_df,
            name=f"{symbol}_{frequency}_{split_name}_advanced",
            output_dir=paths['processed_data'],
            format='parquet',
        )
    
    # Generate diagnostics
    print("\nGenerating diagnostics for advanced features...")
    generate_layer1_report(
        features_df=advanced_features['train'],
        target=label_strategies['next_bar']['train'],
        output_dir=paths['figures'],
        prefix='layer1_advanced_train',
    )
    
    print("\n" + "="*70)
    print("ADVANCED FEATURES COMPLETE ✓")
    print("="*70)
    print("\nSaved Data:")
    print(f"  Basic version: {symbol}_{frequency}_*_basic.parquet (12 features)")
    print(f"  Advanced version: {symbol}_{frequency}_*_advanced.parquet (~30 features)")
    print("\nAvailable Labels:")
    print("  - target: Next-bar direction (baseline)")
    print("  - target_multiday: 5-day return")
    print("  - target_regime: Trend classification")
    print("  - target_volatility: Vol prediction")
    print("\nNext Steps:")
    print("  1. Update config to use advanced features")
    print("  2. Try different label strategies")
    print("  3. Re-train model with:")
    print("     - More features (better signal?)")
    print("     - Better labels (more predictable?)")
    print("  4. Compare performance")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

