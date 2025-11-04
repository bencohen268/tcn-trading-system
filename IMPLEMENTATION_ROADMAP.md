# Implementation Roadmap - Recommendation Analysis

**Goal**: Understand what code changes are needed for each improvement, without breaking current system.

---

## Change Impact Analysis

### Legend:
- 🟢 **Low Risk** - Isolated changes, won't break existing code
- 🟡 **Medium Risk** - Multiple file changes, but modular
- 🔴 **High Risk** - Architectural changes, requires careful testing

---

## 1. Walk-Forward Analysis 🟢 LOW RISK

**What it does**: Test model stability across multiple time periods

**Files to modify**: NONE (just add new)
**Files to create**:
- `run_walk_forward.py` (new script)
- `evaluation/walk_forward.py` (new module)

**Existing code touched**: 0 files
**Risk**: Very low - completely additive

**Estimated effort**: 2-3 hours
**Complexity**: Easy

**What changes**:
```python
# New file: run_walk_forward.py
# Functionality:
# - Loop through time windows
# - Train on window N
# - Test on window N+1
# - Aggregate results
# Uses existing train/backtest infrastructure
```

**Benefits**:
- Tests robustness
- Identifies if any period has edge
- No risk to current system

---

## 2. Add More Features 🟢 LOW RISK

**What it does**: Expand from 12 to 20-30 features

**Files to modify**:
- `features/engineering.py` - Add new features
- `config/params.yaml` - Update n_features count

**Files to create**: NONE

**Existing code touched**: 1 file (just extending)
**Risk**: Very low - purely additive within existing framework

**Estimated effort**: 3-4 hours
**Complexity**: Easy to Medium

**Proposed new features**:
```python
# In features/engineering.py, add:

13. RSI (Relative Strength Index)
14. ATR (Average True Range) 
15. Bollinger Band position
16. MACD signal
17. Price momentum (5-day, 20-day)
18. Volume momentum
19. High-low range volatility
20. Intraday gap (open vs prev close)
21. Weekly/monthly trend alignment
22. Distance from 52-week high/low

# Plus regime features:
23. Trend strength (ADX-like)
24. Market regime (trending vs ranging)
25. Volatility regime (high vs low vs medium)
```

**What changes**:
- Add methods to `FeatureEngineer` class
- Update `get_feature_names()` to return new list
- Update config `n_features: 25`

**Backward compatibility**: Perfect - old features unchanged

---

## 3. Alternative Label Definitions 🟢 LOW RISK

**What it does**: Try different prediction targets

**Files to modify**:
- `features/targets.py` - Add new target functions

**Files to create**: NONE

**Existing code touched**: 1 file (just extending)
**Risk**: Very low - add new functions alongside existing

**Estimated effort**: 1-2 hours
**Complexity**: Easy

**New label functions to add**:
```python
# In features/targets.py, add:

def create_multiday_return_target(df, horizon=5, threshold=0.02):
    """Predict if price will move >2% in next 5 days"""
    
def create_volatility_target(df, window=20):
    """Predict next-day realized volatility"""
    
def create_regime_target(df, fast=20, slow=50):
    """Classify as: trend_up, trend_down, ranging"""
    
def create_tail_target(df, lower_pct=20, upper_pct=80):
    """Predict if return will be in extreme tails"""
```

**What changes**:
- Just add new functions
- Modify `run_layer1.py` to use different target via config
- Add `label_type: "multiday_return"` option to config

**Backward compatibility**: Perfect - old binary target unchanged

---

## 4. Intraday Data 🟢 LOW RISK

**What it does**: Use 5-minute or 1-hour bars instead of daily

**Files to modify**:
- `config/params.yaml` - Change `frequency: "5m"`

**Files to create**: NONE

**Existing code touched**: 0 files (just config change!)
**Risk**: Very low - system already designed for this

**Estimated effort**: 30 minutes
**Complexity**: Trivial

**What changes**:
```yaml
# In config/params.yaml:
data:
  frequency: "5m"  # or "1h", "15m", etc.
  
  # May need to adjust:
  train_start: "2024-01-01"  # Shorter window for intraday
  train_end: "2024-03-01"
  
risk_map:
  vol_target: 0.0002  # Much smaller for 5-min data
```

**Important considerations**:
- Intraday data = more samples but shorter calendar span
- Might need to adjust vol_target (5-min vol << daily vol)
- Transaction costs more impactful at high frequency

**Backward compatibility**: Perfect - just config change

---

## 5. Volatility Prediction Task 🟡 MEDIUM RISK

**What it does**: Predict next-bar volatility instead of direction

**Files to modify**:
- `features/targets.py` - Add vol target (easy)
- `models/train.py` - Change loss to MSE (easy)
- `strategy/risk_map.py` - New vol-based position logic (medium)
- `run_layer3.py` - Update metric tracking (easy)

**Files to create**:
- `strategy/vol_trading.py` - Volatility trading strategies

**Existing code touched**: 4 files
**Risk**: Medium - changes core training loop and strategy

**Estimated effort**: 6-8 hours
**Complexity**: Medium

**What changes**:
```python
# features/targets.py
def create_volatility_target(df, window=20):
    return df['close'].pct_change().rolling(window).std().shift(-1)

# models/train.py
criterion = nn.MSELoss()  # Instead of BCEWithLogitsLoss

# strategy/vol_trading.py
def vol_position_sizer(predicted_vol, realized_vol, target_vol):
    # Scale position inversely to predicted vol
    return target_vol / (predicted_vol + 1e-6)
```

**Why medium risk**:
- Changes loss function (affects all training)
- Risk map logic completely different
- Need new evaluation metrics (MSE, R², correlation)

**Mitigation**:
- Create `models/train_regression.py` as separate file
- Keep classification training unchanged
- Use config flag to switch between tasks

---

## 6. Regime Classification 🟡 MEDIUM RISK

**What it does**: Classify market state (trend_up, trend_down, ranging)

**Files to modify**:
- `features/targets.py` - Add regime labeling
- `models/train.py` - Change to multi-class (3-way)
- `strategy/risk_map.py` - Regime-conditional strategies
- `visualization/` - Update for multi-class metrics

**Files to create**:
- `strategy/regime_strategies.py` - Different strategy per regime

**Existing code touched**: 4-5 files
**Risk**: Medium - changes output structure

**Estimated effort**: 8-10 hours
**Complexity**: Medium

**What changes**:
```python
# features/targets.py
def create_regime_target(df, fast=20, slow=50):
    # Returns 0=trend_up, 1=ranging, 2=trend_down
    
# models/tcn.py
class TCNClassifier:
    def __init__(self, ..., output_dim=3):  # 3 classes
        
# strategy/regime_strategies.py
def regime_position(regime_prob, vol):
    if regime == 'trend_up':
        return momentum_strategy()
    elif regime == 'ranging':
        return mean_reversion_strategy()
    else:
        return momentum_strategy(short=True)
```

**Why medium risk**:
- Changes model output dimension
- Loss becomes CrossEntropyLoss
- Strategy becomes more complex

**Mitigation**:
- Create separate regime-specific modules
- Keep binary classifier as default
- Use config switch: `task: "regime"` vs `task: "direction"`

---

## 7. Multi-Asset Support 🔴 HIGH RISK

**What it does**: Train on multiple symbols simultaneously

**Files to modify**:
- `data/loaders.py` - Load multiple symbols
- `features/engineering.py` - Cross-asset features
- `models/dataset.py` - Handle multiple asset streams
- `models/tcn.py` - Potentially add asset embedding
- `strategy/backtest.py` - Multi-asset portfolio
- `config/params.yaml` - List of symbols

**Files to create**:
- `strategy/portfolio.py` - Portfolio construction
- `evaluation/portfolio_metrics.py` - Portfolio-level metrics

**Existing code touched**: 6+ files
**Risk**: High - architectural changes throughout

**Estimated effort**: 20-30 hours
**Complexity**: High

**What changes**:
```python
# config/params.yaml
data:
  symbols: ["SPY", "QQQ", "IWM", "TLT", "GLD"]
  
# data/loaders.py
def load_multi_asset_data(symbols, ...):
    # Load and align multiple symbols
    
# features/engineering.py
def add_cross_asset_features(dfs):
    # Correlation, relative strength, etc.
    
# models/dataset.py
class MultiAssetDataset:
    # Handle panel data (assets × time × features)
    
# strategy/portfolio.py
def construct_portfolio(asset_predictions, asset_vols):
    # Optimal weights across assets
```

**Why high risk**:
- Changes data structure fundamentally
- Affects every layer of the system
- Complex portfolio optimization logic
- Much harder to debug

**Mitigation**:
- Start with 2 assets only
- Create `multi_asset/` subdirectory
- Keep single-asset system as default
- Extensive testing before full rollout

---

## 8. Model Architecture Improvements 🟡 MEDIUM RISK

**What it does**: Enhance TCN with attention, try Transformer

**Files to modify**:
- `models/tcn.py` - Add attention layers

**Files to create**:
- `models/transformer.py` - Transformer model
- `models/ensemble.py` - Ensemble of models

**Existing code touched**: 1-2 files
**Risk**: Medium - changes model internals

**Estimated effort**: 10-15 hours
**Complexity**: Medium to High

**What changes**:
```python
# models/tcn.py - Add attention
class AttentionTCN(nn.Module):
    def __init__(self, ...):
        self.tcn = TemporalConvNet(...)
        self.attention = nn.MultiheadAttention(...)
        
# models/transformer.py - New architecture
class TimeSeriesTransformer(nn.Module):
    # Vanilla transformer for time series
    
# models/ensemble.py
class ModelEnsemble:
    models = [TCN(), Transformer(), LSTM()]
    # Average predictions
```

**Why medium risk**:
- New architectures might break existing training loop
- More hyperparameters to tune
- Harder to debug

**Mitigation**:
- Keep original TCN as `tcn_v1.py`
- Create factory pattern: `create_model(model_type="tcn")`
- Ensure all models follow same interface

---

## Recommended Implementation Order

### Phase 1: Low-Hanging Fruit (Low Risk, High Value) 🟢

**Week 1-2: Feature Engineering**
1. ✅ Add 10-15 new features to `engineering.py`
2. ✅ Test with walk-forward analysis (new script)
3. ✅ Try alternative labels (multiday, tails)

**Expected outcome**: AUC improves from 0.50 to 0.52-0.54
**Risk**: Minimal - all additive changes
**Rollback**: Just revert config to `n_features: 12`

### Phase 2: Task Exploration (Medium Risk, High Learning) 🟡

**Week 3-4: Alternative Tasks**
4. ✅ Implement volatility prediction
5. ✅ Implement regime classification
6. ✅ Test on intraday data (5-min, 1-hour)

**Expected outcome**: Find which task is most predictable
**Risk**: Moderate - creates new code paths
**Rollback**: Config flag switches back to original task

### Phase 3: Architectural Changes (High Risk, Mixed Value) 🔴

**Week 5-6: Advanced Features**
7. 🟡 Add attention to TCN
8. 🟡 Try Transformer architecture
9. 🔴 Multi-asset support (only if single-asset shows edge)

**Expected outcome**: Marginal improvements (2-5% better metrics)
**Risk**: High - potential to break system
**Rollback**: Git revert to Phase 2

---

## Implementation Guidelines

### Before Making Changes:

1. **Git Commit Current State**
   ```bash
   git add .
   git commit -m "Working baseline - no edge but stable"
   git tag v1.0-baseline
   ```

2. **Create Feature Branch**
   ```bash
   git checkout -b feature/additional-features
   ```

3. **Test Extensively**
   - Run all layers after each change
   - Verify no regressions
   - Document new parameters in config

### Safe Change Pattern:

```python
# WRONG: Modifying existing code
def create_features(self, df):
    # Change line 50
    
# RIGHT: Adding new code
def create_features(self, df):
    # Lines 1-100 unchanged
    
    if self.use_advanced_features:  # New config flag
        df = self._add_new_features(df)  # New method
    
    return df

def _add_new_features(self, df):  # New method
    # All new logic here
    return df
```

### Config-Driven Development:

```yaml
# config/params.yaml
features:
  use_advanced_features: false  # Safe default
  feature_version: "v1"  # or "v2", "v3"
  
model:
  architecture: "tcn"  # or "tcn_attention", "transformer"
  
training:
  task_type: "direction"  # or "volatility", "regime"
```

---

## Detailed Change Matrix

| Recommendation | Files Modified | Files Added | Risk | Effort | Expected Gain |
|----------------|----------------|-------------|------|--------|---------------|
| Walk-forward | 0 | 2 | 🟢 Low | 3h | High (validation) |
| More features | 1 | 0 | 🟢 Low | 4h | Medium (0.52 AUC?) |
| Alt labels | 1 | 0 | 🟢 Low | 2h | Medium (task-dependent) |
| Intraday | 0 (config) | 0 | 🟢 Low | 1h | High (more signal) |
| Vol prediction | 4 | 1 | 🟡 Med | 8h | High (easier task) |
| Regime class | 5 | 1 | 🟡 Med | 10h | Medium (interpretable) |
| Multi-asset | 6+ | 3+ | 🔴 High | 30h | High (if done right) |
| Model improve | 2 | 2 | 🟡 Med | 15h | Low (marginal) |

---

## Risk Mitigation Strategies

### 1. Feature Flags
```python
# Use config to enable/disable features
if config.get('use_experimental_features', False):
    # New code path
else:
    # Original code path (always works)
```

### 2. Separate Modules
```
models/
  tcn.py          # Original (never touch)
  tcn_v2.py       # Experimental
  transformer.py  # Experimental
  
strategy/
  risk_map.py           # Original
  vol_risk_map.py       # For vol prediction
  regime_strategies.py  # For regime task
```

### 3. Regression Testing
```python
# tests/test_backwards_compatibility.py
def test_original_pipeline_still_works():
    # Run with original config
    # Verify output matches baseline
```

### 4. Git Discipline
```bash
# Always work on branches
git checkout -b feature/new-features
# Commit frequently
git commit -m "Added RSI feature - tests pass"
# Can always revert
git checkout main  # Back to safety
```

---

## Recommended First Steps

### Safest Path to Better Results:

**Step 1: Feature Engineering (This Weekend)**
```python
# Just edit features/engineering.py
# Add 5 new features:
- RSI
- ATR  
- Bollinger position
- Momentum (5-day)
- Trend strength

# Update config: n_features: 17
# Run: python run_all.py
# Compare: Does val AUC improve?
```

**Step 2: Walk-Forward Validation (Next Week)**
```python
# Create run_walk_forward.py
# Test robustness across multiple periods
# See if any period has consistent edge
```

**Step 3: Try Intraday (If Walk-Forward Shows Promise)**
```yaml
# config/params.yaml
frequency: "1h"  # Start with 1-hour (easier than 5-min)
```

**Step 4: Alternative Tasks (If Still No Edge)**
```python
# Try volatility prediction
# Usually more predictable than direction
```

---

## What NOT to Do (Yet)

❌ **Don't touch** until you have some edge:
- Multi-asset support (complex, not proven needed)
- Advanced architectures (Transformer, etc.)
- Complex ensemble methods
- Real-time data feeds

✅ **Do first** - these have best ROI:
- More features (cheap, high value)
- Walk-forward testing (validate results)
- Alternative tasks (find what's predictable)
- Intraday data (more signal)

---

## Summary: Change Complexity

### Easy Wins (Do First):
1. 🟢 More features - 4 hours, low risk
2. 🟢 Alt labels - 2 hours, low risk
3. 🟢 Intraday - 1 hour, low risk
4. 🟢 Walk-forward - 3 hours, low risk

### Medium Changes (Do if Easy Wins Show Promise):
5. 🟡 Vol prediction - 8 hours, medium risk
6. 🟡 Regime class - 10 hours, medium risk
7. 🟡 Model improve - 15 hours, medium risk

### Hard Changes (Only if You Have Proven Edge):
8. 🔴 Multi-asset - 30+ hours, high risk

---

## Conclusion

**Good News**: System is well-architected! Most improvements are:
- Low risk (isolated changes)
- Additive (don't break existing code)
- Config-driven (easy to toggle)

**The Path Forward**:
1. Push current code to GitHub (backup!)
2. Start with feature engineering (safest, highest ROI)
3. Add walk-forward testing (validate results)
4. Only then move to bigger changes

**Your instinct is correct**: Don't risk breaking the infrastructure. The modularity you built makes iteration safe.

---

**Next Action**: Let me help you push to GitHub, then we can implement Phase 1 changes together.

