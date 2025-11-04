"""Data splitting utilities."""

from typing import Tuple, Dict
import pandas as pd
import numpy as np


def split_by_date(
    df: pd.DataFrame,
    train_start: str,
    train_end: str,
    val_start: str,
    val_end: str,
    test_start: str,
    test_end: str,
) -> Dict[str, pd.DataFrame]:
    """
    Split data by date ranges.
    
    Args:
        df: DataFrame with datetime index
        train_start, train_end: Training date range
        val_start, val_end: Validation date range
        test_start, test_end: Test date range
        
    Returns:
        Dictionary with 'train', 'val', 'test' DataFrames
    """
    splits = {
        'train': df.loc[train_start:train_end].iloc[:-1],  # Exclude last row to avoid overlap
        'val': df.loc[val_start:val_end].iloc[:-1],
        'test': df.loc[test_start:test_end],
    }
    
    # Validate no overlap
    assert splits['train'].index[-1] < splits['val'].index[0], "Train and val overlap!"
    assert splits['val'].index[-1] < splits['test'].index[0], "Val and test overlap!"
    
    print(f"Train: {len(splits['train'])} bars ({splits['train'].index[0]} to {splits['train'].index[-1]})")
    print(f"Val:   {len(splits['val'])} bars ({splits['val'].index[0]} to {splits['val'].index[-1]})")
    print(f"Test:  {len(splits['test'])} bars ({splits['test'].index[0]} to {splits['test'].index[-1]})")
    
    return splits


def get_split_indices(
    df: pd.DataFrame,
    train_start: str,
    train_end: str,
    val_start: str,
    val_end: str,
    test_start: str,
    test_end: str,
) -> Dict[str, Tuple[int, int]]:
    """
    Get integer indices for each split.
    
    Args:
        df: DataFrame with datetime index
        train_start, train_end: Training date range
        val_start, val_end: Validation date range
        test_start, test_end: Test date range
        
    Returns:
        Dictionary with (start_idx, end_idx) tuples for each split
    """
    splits = split_by_date(
        df, train_start, train_end,
        val_start, val_end,
        test_start, test_end
    )
    
    indices = {}
    for split_name, split_df in splits.items():
        start_idx = df.index.get_loc(split_df.index[0])
        end_idx = df.index.get_loc(split_df.index[-1])
        indices[split_name] = (start_idx, end_idx + 1)
    
    return indices

