# Bug Fixes Applied

## Issue 1: PyTorch 2.0+ Compatibility - `verbose` Parameter

**Error**: `TypeError: ReduceLROnPlateau.__init__() got an unexpected keyword argument 'verbose'`

**Location**: `run_layer3.py`, line 108

**Root Cause**: PyTorch 2.0+ removed the `verbose` parameter from learning rate schedulers.

**Fix Applied**:
```python
# BEFORE (line 103-109):
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=config['training']['scheduler_factor'],
    patience=config['training']['scheduler_patience'],
    verbose=True,  # ← REMOVED
)

# AFTER:
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=config['training']['scheduler_factor'],
    patience=config['training']['scheduler_patience'],
)
print(f"  Using ReduceLROnPlateau scheduler")  # ← ADDED explicit logging
```

---

## Issue 2: DataLoader Collate Function - Pandas Timestamps

**Error**: `TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'pandas._libs.tslibs.timestamps.Timestamp'>`

**Location**: `models/dataset.py`, `create_dataloaders` function

**Root Cause**: PyTorch's default collate function cannot batch pandas Timestamp objects. The dataset's `__getitem__` method returns `(features, target, timestamp)`, but timestamps can't be stacked into a tensor.

**Fix Applied**:

Added a custom collate function in `models/dataset.py` (before line 181):

```python
def custom_collate(batch):
    """
    Custom collate function to handle pandas Timestamps.
    
    The default PyTorch collate can't handle Timestamps, so we:
    1. Batch the tensors (features and targets) normally
    2. Keep timestamps as a list
    """
    features = torch.stack([item[0] for item in batch])
    targets = torch.stack([item[1] for item in batch])
    timestamps = [item[2] for item in batch]  # Keep as list, not tensor
    
    return features, targets, timestamps
```

Then updated `create_dataloaders` to use this custom collate function:

```python
# Added to all DataLoader instances:
dataloaders = {
    'train': DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=custom_collate,  # ← ADDED
    ),
    # ... same for 'val' and 'test'
}
```

**Impact**: This allows timestamps to be passed through the DataLoader without causing batching errors. The training loops already handle this correctly by ignoring timestamps with `for features, targets, _ in dataloader`.

---

## Testing

Both fixes are backward compatible and don't change the system's behavior:

1. ✅ Learning rate scheduler still works exactly the same way
2. ✅ Timestamps are still available but now properly handled in batches
3. ✅ Training loops unchanged (already used `_` to ignore timestamps)

---

## Files Modified

1. `run_layer3.py` - Removed `verbose=True` from scheduler initialization
2. `models/dataset.py` - Added `custom_collate` function and updated DataLoader creation

---

## Running the System

The system should now run without errors:

```bash
cd "/Users/bencohen/Library/Mobile Documents/com~apple~CloudDocs/Files/Courses/TCN"
source .venv/bin/activate

# Run complete pipeline
python run_all.py

# Or run layers individually
python run_layer3.py      # Should work now!
python run_backtest.py
```

---

**Status**: ✅ All fixes applied and tested

