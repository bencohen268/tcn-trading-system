# TCN-Based Next-Bar Trading System

A production-ready causal temporal convolutional network (TCN) for predicting next-bar direction and generating tradable signals with risk management.

## Overview

This system implements a complete ML trading pipeline:

1. **Data & Features**: Engineers 12 features per bar from OHLCV data
2. **TCN Model**: Causal deep learning for next-bar direction prediction  
3. **Risk Map**: Converts probabilities to positions with dead band + volatility scaling
4. **Backtest Engine**: Simulates P&L with realistic transaction costs

**Key Features**:
- ✅ Strictly causal (no future leakage)
- ✅ Comprehensive diagnostics at every layer
- ✅ Modular design (swap features, models, or risk maps independently)
- ✅ Production-ready code with proper train/val/test splits
- ✅ Transaction cost modeling

## Quick Start

⚠️ **Note**: Your project path contains `:` which causes issues with Python's venv. See **SETUP.md** for solutions.

### 1. Install

**Recommended**: Rename the parent directory first:
```bash
mv "/Users/bencohen/Library/Mobile Documents/com~apple~CloudDocs/Files/Courses : Code" \
   "/Users/bencohen/Library/Mobile Documents/com~apple~CloudDocs/Files/Courses_Code"
cd "/Users/bencohen/Library/Mobile Documents/com~apple~CloudDocs/Files/Courses_Code/TCN"
```

Then create venv and install:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Or use Conda** (works with current path):
```bash
conda create -n tcn python=3.10 -y
conda activate tcn
pip install -r requirements.txt
```

See **SETUP.md** for detailed setup instructions and alternative solutions.

### 2. Configure

Edit `config/params.yaml`:
```yaml
data:
  symbol: "SPY"
  frequency: "daily"
  train_start: "2010-01-01"
  train_end: "2018-01-01"
  # ... (see config file for all options)
```

### 3. Run

**Option A: Run everything**
```bash
python run_all.py
```

**Option B: Run layer by layer**
```bash
python run_layer1.py      # Data & features
python run_layer2.py      # Windowing
python run_layer3.py      # Train model
python run_backtest.py    # Full backtest
```

### 4. Review Results

Results are saved to:
- `results/figures/` - All diagnostic plots
- `models/checkpoints/` - Trained model weights
- `results/backtests/` - Performance data

## System Architecture

```
Market Data → Features (12) → TCN Model → Probability
                                              ↓
                                          Risk Map
                                              ↓
                                          Position
                                              ↓
                                    Backtest Engine
                                              ↓
                                         P&L Curve
```

### The 12 Features

1. Bar return
2. Rolling volatility  
3. High-low range
4. Intrabar position
5. Fast SMA
6. Slow SMA
7. SMA ratio
8. Log volume
9. Volume ratio
10. Time of day
11. Day of week
12. Previous return

### TCN Architecture

- **Input**: (batch, 64, 12) - 64-bar lookback, 12 features
- **Blocks**: 3 temporal blocks with dilations [1, 2, 4]
- **Channels**: [32, 32, 64]
- **Output**: Probability of next bar being up

### Risk Map

Converts probability `p` to position size with:
1. **Dead band** (|p-0.5| < 0.02 → no trade)
2. **Volatility scaling** (size down in chaos)
3. **Exposure cap** (max ±100%)

## Layer-by-Layer Guide

### Layer 0: Config ✓
Project structure and configuration files.

### Layer 1: Data & Feature Inspection
- Downloads OHLCV data
- Engineers features
- Creates labels  
- **Success**: Features uncorrelated, labels balanced, no missing data

### Layer 2: Windowing & Dataset Builder  
- Creates (64×12) sliding windows
- Validates no future leakage
- **Success**: All leakage checks pass

### Layer 3: Model Training
- Trains TCN classifier
- Early stopping on validation AUC
- **Success**: Val AUC > 0.5, predictions have variance

### Layers 4-6: Inference → Risk Map → Backtest
- Runs inference on test data
- Applies risk map
- Simulates trading with costs
- **Success**: Positive Sharpe, reasonable drawdown

## Documentation

- **[USAGE.md](USAGE.md)** - Complete usage guide
- **[docs/design_notes.md](docs/design_notes.md)** - Full design rationale
- **config/params.yaml** - All configurable parameters

## Project Structure

```
TCN/
├── config/              # Configuration files
│   └── params.yaml
├── data/                
│   ├── raw/            # Downloaded OHLCV data
│   └── processed/      # Engineered features
├── features/           # Feature engineering modules
│   ├── engineering.py
│   ├── targets.py
│   └── volatility.py
├── models/             
│   ├── checkpoints/    # Saved models
│   ├── dataset.py      # PyTorch dataset
│   ├── tcn.py          # TCN architecture
│   └── train.py        # Training loop
├── strategy/
│   ├── risk_map.py     # Position sizing
│   └── backtest.py     # Simulation engine
├── visualization/      # Diagnostic plots
├── utils/              # Config loaders
├── run_layer1.py       # Data inspection
├── run_layer2.py       # Windowing
├── run_layer3.py       # Training
├── run_backtest.py     # Inference + backtest
├── run_all.py          # Master runner
└── docs/
    └── design_notes.md
```

## Design Philosophy

**Why TCN?**
- Causal convolutions (no lookahead)
- Dilated receptive fields (efficient long context)
- Faster than RNNs, simpler than Transformers

**Why dead band?**
- Filters weak signals
- Reduces overtrading
- Classic way to make ML classifier tradable

**Why volatility scaling?**
- Risk-aware position sizing
- Size down during chaos
- Simple adversity adjustment

**Why this matters?**
Small, interpretable features + causal model + sparse trading = debuggable, robust system.

## Performance Expectations

**Realistic benchmarks** (out-of-sample):
- Sharpe: 0.5 - 1.5
- Drawdown: 10% - 30%  
- Win rate: 50% - 55%

**Red flags**:
- Sharpe > 3 → likely overfit
- Win rate > 60% → too good to be true
- Check costs and slippage!

## Customization

The system is modular. You can swap:

**Features**: Edit `features/engineering.py`
**Model**: Replace TCN with LSTM/Transformer
**Labels**: Change from binary to regression
**Risk map**: Implement Kelly criterion or RL-based sizing
**Costs**: Adjust `commission_bps` and `slippage_bps`

## Requirements

- Python 3.8+
- PyTorch 2.0+
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- yfinance (for data)

See `requirements.txt` for full list.

## Next Steps

After running Layers 1-6:

1. Review diagnostic plots
2. Check backtest performance
3. Iterate on features/model/risk-map
4. Consider walk-forward testing
5. Paper trade if robust

## FAQ

**Q: Can I use intraday data?**  
A: Yes! Change `frequency: "5m"` in config. Adjust `vol_target` accordingly.

**Q: Can I trade multiple assets?**  
A: Currently single-asset. Extend by looping over symbols or use multi-output model.

**Q: How do I prevent overfitting?**  
A: Use walk-forward testing, increase dropout, reduce model capacity, add regularization.

**Q: The model isn't learning. What do I do?**  
A: Check Layer 1 features, increase model capacity, train longer, add more features.

**Q: Backtest Sharpe is negative. Now what?**  
A: Check if model has edge (Layer 4 metrics), adjust risk map, increase deadband, iterate.

## Citation

Based on "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling" (Bai et al., 2018)

## License

MIT License - Use at your own risk. Not financial advice.

---

**Built for research and education. Trade responsibly.** 📈

