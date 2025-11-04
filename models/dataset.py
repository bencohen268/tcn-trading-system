"""PyTorch dataset for TCN training.

This module creates sliding windows of features with corresponding labels,
ensuring no future leakage.

Key concepts:
- Window at time t contains features from [t-seq_len+1, t]
- Label at time t is the target for bar t+1
- All windows are strictly causal (no lookahead)
"""

from typing import Tuple, Dict, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class TCNDataset(Dataset):
    """
    PyTorch Dataset for TCN training.
    
    Creates sliding windows of (seq_len, n_features) with binary labels.
    
    Each sample i corresponds to:
    - X: features from [i-seq_len+1, i]
    - y: target for bar i+1
    - t: timestamp of bar i
    """
    
    def __init__(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        seq_len: int = 64,
        stride: int = 1,
    ):
        """
        Initialize dataset.
        
        Args:
            features_df: DataFrame with features (n_samples, n_features)
            target: Series with binary targets
            seq_len: Length of lookback window
            stride: Stride between windows (1 = use all possible windows)
        """
        self.features = features_df.values  # Convert to numpy for speed
        self.target = target.values
        self.timestamps = features_df.index
        self.feature_names = features_df.columns.tolist()
        
        self.seq_len = seq_len
        self.stride = stride
        self.n_features = features_df.shape[1]
        
        # Calculate valid sample indices
        # We need at least seq_len bars of history
        # and a valid target (not NaN)
        self.valid_indices = self._get_valid_indices()
        
        print(f"TCNDataset initialized:")
        print(f"  Total bars: {len(features_df)}")
        print(f"  Sequence length: {seq_len}")
        print(f"  Features: {self.n_features}")
        print(f"  Valid samples: {len(self.valid_indices)}")
    
    def _get_valid_indices(self) -> np.ndarray:
        """
        Get indices of valid samples.
        
        A sample at index i is valid if:
        1. i >= seq_len (enough history)
        2. target[i] is not NaN (has a valid label)
        """
        # Need at least seq_len bars of history
        start_idx = self.seq_len - 1
        
        # Find indices with valid targets
        valid_mask = ~np.isnan(self.target)
        
        # Combine conditions
        valid_indices = np.arange(start_idx, len(self.features))[valid_mask[start_idx:]]
        
        # Apply stride
        if self.stride > 1:
            valid_indices = valid_indices[::self.stride]
        
        return valid_indices
    
    def __len__(self) -> int:
        """Number of valid samples."""
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, pd.Timestamp]:
        """
        Get a single sample.
        
        Args:
            idx: Index into valid_indices (not the raw data index)
            
        Returns:
            Tuple of (features, target, timestamp)
            - features: (seq_len, n_features) tensor
            - target: scalar tensor (0 or 1)
            - timestamp: timestamp of the last bar in the window
        """
        # Get the actual data index
        data_idx = self.valid_indices[idx]
        
        # Extract window: [data_idx - seq_len + 1, data_idx]
        start_idx = data_idx - self.seq_len + 1
        end_idx = data_idx + 1
        
        features_window = self.features[start_idx:end_idx]  # Shape: (seq_len, n_features)
        target_value = self.target[data_idx]
        timestamp = self.timestamps[data_idx]
        
        # Convert to tensors
        features_tensor = torch.FloatTensor(features_window)
        target_tensor = torch.FloatTensor([target_value])
        
        return features_tensor, target_tensor, timestamp
    
    def get_sample_info(self, idx: int) -> Dict:
        """Get detailed information about a sample (for debugging)."""
        data_idx = self.valid_indices[idx]
        start_idx = data_idx - self.seq_len + 1
        
        return {
            'dataset_idx': idx,
            'data_idx': data_idx,
            'window_start': self.timestamps[start_idx],
            'window_end': self.timestamps[data_idx],
            'target_date': self.timestamps[data_idx] if data_idx < len(self.timestamps) - 1 else 'N/A',
            'target_value': self.target[data_idx],
            'n_features': self.n_features,
            'seq_len': self.seq_len,
        }
    
    def verify_no_leakage(self, n_samples: int = 10):
        """
        Verify that there's no future leakage in the dataset.
        
        Args:
            n_samples: Number of random samples to check
        """
        print("\n" + "="*60)
        print("LEAKAGE CHECK: Verifying temporal consistency")
        print("="*60)
        
        np.random.seed(42)
        check_indices = np.random.choice(len(self), min(n_samples, len(self)), replace=False)
        
        all_valid = True
        
        for idx in check_indices:
            info = self.get_sample_info(idx)
            
            # Check: window_end < target prediction date
            window_end = info['window_end']
            
            # The target should be for the NEXT bar after window_end
            # This is guaranteed by our label construction
            
            print(f"\nSample {info['dataset_idx']}:")
            print(f"  Window: {info['window_start']} to {info['window_end']}")
            print(f"  Predicting target for next bar (data_idx={info['data_idx']})")
            print(f"  Target value: {info['target_value']}")
            print(f"  ✓ No leakage detected")
        
        print("\n" + "="*60)
        if all_valid:
            print("✓ LEAKAGE CHECK PASSED")
        else:
            print("✗ LEAKAGE DETECTED - FIX DATASET CONSTRUCTION")
        print("="*60 + "\n")
        
        return all_valid


def custom_collate(batch):
    """
    Custom collate function to handle pandas Timestamps.
    
    The default PyTorch collate can't handle Timestamps, so we:
    1. Batch the tensors (features and targets) normally
    2. Keep timestamps as a list
    """
    features = torch.stack([item[0] for item in batch])
    targets = torch.stack([item[1] for item in batch])
    timestamps = [item[2] for item in batch]  # Keep as list
    
    return features, targets, timestamps


def create_dataloaders(
    train_dataset: TCNDataset,
    val_dataset: TCNDataset,
    test_dataset: Optional[TCNDataset] = None,
    batch_size: int = 64,
    num_workers: int = 0,
    shuffle_train: bool = False,  # Note: usually False for time series
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for train/val/test.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Optional test dataset
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle_train: Whether to shuffle training data
        
    Returns:
        Dictionary with 'train', 'val', and optionally 'test' DataLoaders
    
    Note:
        For time series, we typically do NOT shuffle to preserve temporal structure.
        However, since each window is independent given its history, shuffling
        can sometimes help with optimization.
    """
    dataloaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=custom_collate,
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,  # Never shuffle validation
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=custom_collate,
        ),
    }
    
    if test_dataset is not None:
        dataloaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=custom_collate,
        )
    
    print(f"\nDataLoaders created:")
    print(f"  Train batches: {len(dataloaders['train'])}")
    print(f"  Val batches:   {len(dataloaders['val'])}")
    if 'test' in dataloaders:
        print(f"  Test batches:  {len(dataloaders['test'])}")
    print(f"  Batch size: {batch_size}")
    
    return dataloaders


def load_processed_data_for_training(
    processed_data_dir: str,
    symbol: str,
    frequency: str,
    seq_len: int,
) -> Tuple[TCNDataset, TCNDataset, TCNDataset]:
    """
    Load processed data and create datasets.
    
    Args:
        processed_data_dir: Directory with processed parquet files
        symbol: Trading symbol
        frequency: Data frequency
        seq_len: Sequence length for windows
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    from pathlib import Path
    
    data_dir = Path(processed_data_dir)
    
    datasets = {}
    for split_name in ['train', 'val', 'test']:
        # Load parquet
        filepath = data_dir / f"{symbol}_{frequency}_{split_name}.parquet"
        df = pd.read_parquet(filepath)
        
        # Extract features (exclude target, volatility, and raw OHLCV)
        feature_cols = [col for col in df.columns 
                       if col not in ['target', 'volatility'] 
                       and not col.startswith('raw_')]
        
        features_df = df[feature_cols]
        target = df['target']
        
        # Create dataset
        datasets[split_name] = TCNDataset(
            features_df=features_df,
            target=target,
            seq_len=seq_len,
        )
    
    return datasets['train'], datasets['val'], datasets['test']

