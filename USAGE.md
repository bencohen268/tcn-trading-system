# TCN Trading System - Usage Guide

Complete guide for running the TCN-based next-bar trading system.

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Parameters

Edit `config/params.yaml` to set:
- Data symbol and date ranges
- Feature engineering parameters
- Model architecture
- Risk map parameters
- Backtest settings

### 3. Run Complete Pipeline

```bash
python run_all.py
```

This runs all layers sequentially. Alternatively, run layers individually (see below).

---

## Layer-by-Layer Execution

### Layer 1: Data & Feature Inspection

**Goal**: Validate raw data and engineered features.

```bash
python run_layer1.py
```

**What it does**:
- Downloads/loads OHLCV data from Yahoo Finance
- Engineers 12 features per bar
- Creates binary next-bar direction labels
- Estimates volatility
- Generates diagnostic plots

**Success criteria**:
- ✓ Features aren't flat or super-correlated
- ✓ Label balance isn't extreme (not 90% one class)
- ✓ No missing data or invalid prices

**Outputs**:
- Processed data: `data/processed/`
- Diagnostic plots: `results/figures/layer1_*.png`

**If bad**: Adjust feature definitions in `features/engineering.py` or change normalization in config.

---

### Layer 2: Windowing & Dataset Builder

**Goal**: Create PyTorch datasets with no future leakage.

```bash
python run_layer2.py
```

**What it does**:
- Creates sliding windows of (seq_len=64, n_features=12)
- Validates temporal ordering
- Checks for data leakage
- Verifies train/val/test splits don't overlap

**Success criteria**:
- ✓ All leakage checks pass
- ✓ Sample counts match expectations
- ✓ Temporal splits are properly separated

**Outputs**:
- Console report with leakage checks

**If bad**: Should not fail if Layer 1 succeeded. Check date ranges in config.

---

### Layer 3: Model Training

**Goal**: Train TCN to predict next-bar direction.

```bash
python run_layer3.py
```

**What it does**:
- Initializes TCN model (3 blocks, dilations [1,2,4])
- Trains on training data
- Validates on validation data
- Uses early stopping
- Saves best model checkpoint

**Success criteria**:
- ✓ Train loss goes down
- ✓ Val AUC > 0.5 (model learns signal)
- ✓ Predictions have variance (not stuck at 0.5)
- ✓ No severe overfitting

**Outputs**:
- Model checkpoint: `models/checkpoints/best_model.pt`
- Training history: `models/checkpoints/training_history.json`
- Diagnostic plots: `results/figures/layer3_*.png`

**If bad**:
- Increase model capacity (more channels)
- Increase sequence length
- Add more features
- Reduce dropout
- Train longer

---

### Layers 4-6: Inference, Risk Map, Backtest

**Goal**: Test strategy on out-of-sample data.

```bash
python run_backtest.py
```

**What it does**:
- **Layer 4**: Runs inference on test data
- **Layer 5**: Applies risk map to convert probabilities → positions
- **Layer 6**: Simulates trading with transaction costs

**Success criteria**:
- ✓ Positive Sharpe ratio out-of-sample
- ✓ Reasonable drawdown (<30%)
- ✓ Predictions correlate with returns

**Outputs**:
- Backtest results: `results/backtests/`
- Performance plots: `results/figures/backtest_*.png`
  - Equity curve & drawdown
  - Positions vs price
  - Returns distribution
  - Monthly returns heatmap
  - Turnover analysis

**If bad**:
- Adjust risk map parameters (deadband, vol_target)
- Revisit model architecture
- Improve feature engineering
- Check if model has actual edge

---

## Configuration Guide

### Key Parameters in `config/params.yaml`

#### Data Configuration
```yaml
data:
  symbol: "SPY"                # Ticker to trade
  frequency: "daily"           # "daily", "1h", "5m", etc.
  seq_len: 64                  # Lookback window
  n_features: 12               # Number of features per bar
  
  # Data splits
  train_start: "2010-01-01"
  train_end: "2018-01-01"
  val_start: "2018-01-01"
  val_end: "2019-01-01"
  test_start: "2019-01-01"
  test_end: "2021-01-01"
```

#### Model Architecture
```yaml
model:
  num_inputs: 12               # Must match n_features
  num_channels: [32, 32, 64]   # Channel size per TCN block
  kernel_size: 3               # Convolution kernel
  dropout: 0.1                 # Regularization
```

#### Training
```yaml
training:
  learning_rate: 0.001
  batch_size: 64
  num_epochs: 20
  early_stopping_patience: 5
```

#### Risk Map
```yaml
risk_map:
  deadband: 0.02               # Min edge to trade
  vol_target: 0.01             # Target volatility
  max_abs_position: 1.0        # Max position size
```

#### Backtest
```yaml
backtest:
  initial_capital: 100000.0
  commission_bps: 1.0          # Cost per trade side
  slippage_bps: 0.5            # Execution slippage
```

---

## Understanding the 12 Features

The system engineers these features per bar:

1. **bar_return**: Log return of the bar
2. **rolling_volatility**: Standard deviation of returns (20-bar window)
3. **hl_range**: (high - low) / close
4. **intrabar_position**: Where close sits in [low, high]
5. **sma_fast**: Fast SMA deviation (5-period)
6. **sma_slow**: Slow SMA deviation (20-period)
7. **sma_ratio**: fast/slow ratio
8. **log_volume**: Log-scaled volume
9. **volume_ratio**: Volume vs rolling average
10. **time_of_day**: Hour encoding (for intraday)
11. **day_of_week**: Weekday encoding
12. **prev_return**: Lag-1 return

All features are z-score normalized.

---

## Risk Map Explanation

The risk map converts model probability `p` and volatility `vol` into position size:

1. **Dead band**: If `|p - 0.5| < deadband`, position = 0
   - Filters out weak signals
   - Reduces overtrading

2. **Edge scaling**: Maps `p` from [0.5+deadband, 1.0] → [0, 1]
   - Preserves sign (long vs short)
   - Smooth position sizing

3. **Volatility adjustment**: Scale by `vol_target / vol`
   - Size down when volatile
   - Size up when calm
   - Clamped to [0.2, 1.5]

4. **Exposure cap**: Clip to [-1.0, 1.0]
   - Prevents overleveraging

**Example**:
- `p=0.6, vol=0.01` → moderate long position
- `p=0.51, vol=0.01` → no position (within dead band)
- `p=0.7, vol=0.03` → small long position (high vol)

---

## Diagnostic Plots

### Layer 1 Diagnostics
- **Feature timeseries**: Check for drift over time
- **Feature distributions**: Spot outliers or non-stationarity
- **Correlation matrix**: Identify redundant features
- **Label balance**: Check for regime bias
- **Missing data**: Validate data quality

### Layer 3 Diagnostics
- **Training curves**: Loss and metrics over epochs
- **Calibration plot**: Are probabilities well-calibrated?
- **Prediction distribution**: Is model stuck at 0.5?
- **Confusion matrix**: Classification performance

### Backtest Diagnostics
- **Equity curve**: Cumulative P&L and drawdown
- **Positions vs price**: Visualize trading behavior
- **Returns distribution**: Check for outliers
- **Monthly heatmap**: Identify seasonal patterns
- **Turnover analysis**: Trading frequency

---

## Troubleshooting

### "Model not learning (AUC ≤ 0.5)"
- Check feature engineering (Layer 1 plots)
- Increase model capacity (more channels)
- Increase sequence length
- Add more features
- Train longer

### "High overfitting (train AUC >> val AUC)"
- Increase dropout
- Reduce model capacity
- Add more training data
- Use stronger regularization

### "Backtest Sharpe < 0"
- Check if model has edge (Layer 4 metrics)
- Adjust risk map parameters
- Increase deadband (trade less)
- Adjust vol_target

### "Too much turnover"
- Increase deadband
- Smooth positions (add lag)
- Reduce position sizes

### "Data download fails"
- Check internet connection
- Verify ticker symbol is valid
- Try different date ranges
- Use cached data if available

---

## Next Steps

After running Layers 1-6:

1. **Review all diagnostics**
   - Are features stationary?
   - Is the model well-calibrated?
   - Does backtest pass smell test?

2. **Iterate**
   - Adjust parameters in `config/params.yaml`
   - Re-run specific layers as needed
   - Track experiments

3. **Walk-forward testing** (Layer 7)
   - Test robustness across multiple periods
   - Retrain periodically
   - Check parameter stability

4. **Paper trading**
   - If results are robust, consider live paper trading
   - Monitor slippage vs assumptions
   - Validate execution logic

---

## File Structure

```
TCN/
├── config/              # Configuration files
├── data/               
│   ├── raw/            # Downloaded OHLCV data
│   └── processed/      # Engineered features
├── features/           # Feature engineering modules
├── models/             
│   ├── checkpoints/    # Saved model weights
│   ├── dataset.py      # PyTorch dataset
│   ├── tcn.py          # TCN architecture
│   └── train.py        # Training utilities
├── strategy/
│   ├── risk_map.py     # Risk mapping logic
│   └── backtest.py     # Backtest engine
├── visualization/      # Diagnostic plots
├── utils/              # Config and utilities
├── run_layer1.py       # Layer 1 runner
├── run_layer2.py       # Layer 2 runner
├── run_layer3.py       # Layer 3 runner
├── run_backtest.py     # Layers 4-6 runner
├── run_all.py          # Master runner
└── README.md           # Project overview
```

---

## Performance Expectations

### Realistic Benchmarks
- **Sharpe ratio**: 0.5 - 1.5 (out-of-sample)
- **Max drawdown**: 10% - 30%
- **Win rate**: 50% - 55%
- **Annualized return**: Market-dependent

### Red Flags
- ❌ Sharpe > 3.0 → Likely overfitting or data leak
- ❌ Win rate > 60% → Too good to be true
- ❌ Drawdown < 5% with high returns → Check costs

---

## Support

For issues or questions:
1. Check diagnostic plots first
2. Review `config/params.yaml`
3. Read error messages carefully
4. Consult `docs/design_notes.md` for design rationale

---

**Good luck trading!** 🚀

