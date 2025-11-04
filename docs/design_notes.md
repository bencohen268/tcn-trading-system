# TCN Trading System Design Notes

## 1. Overall Idea

**Goal**: Predict next-bar direction (or small-threshold return) using a causal Temporal Convolutional Network (TCN) over a fixed-length feature window, then shape that prediction into a position using a risk map (dead band → volatility-aware scaling → exposure cap).

**Rationale**:
- TCNs are fast, causal, and good at short sequence patterns
- Predicting a probability is more stable than predicting raw returns
- The risk map is where trading logic lives — it makes the ML signal tradable

---

## 2. Data & Labeling

**Data frequency**: same as trading frequency (e.g. 1m, 5m, or daily)

**Core columns needed**:
- timestamp, open, high, low, close, volume
- (optional) bid/ask/microstructure if intraday

**Labeling (baseline)**:
```python
ret_1 = close.shift(-1)/close - 1
y = 1 if ret_1 > 0 else 0
```

- Predict at time t for bar t+1
- Can later generalize to: `y = 1 if ret_1 > threshold else 0`

**Why**: binary labels → BCE loss → straightforward probability

---

## 3. Feature Window: 64 × 12

**Shape per sample**: `(seq_len=64, n_features=12)`

**Interpretation**:
- 64 = lookback of last 64 bars
- 12 = engineered features per bar

**Candidate 12 features**:
1. bar return (log or pct)
2. rolling volatility (e.g. std of last N returns)
3. high–low range
4. (close - low)/(high - low) — intrabar location
5. SMA fast
6. SMA slow
7. fast/slow ratio or spread
8. rolling volume (log)
9. volume vs rolling avg volume
10. time-of-day / day-of-week encoding
11. rolling VWAP deviation (if you have VWAP)
12. previous signal / previous return

**Why 64?**
- Long enough to catch short-term regime shifts and intraday patterns
- Still small, so TCN stays fast
- Easy to expand to 96/128 later

**Why 12?**
- Small, interpretable feature set → less overfit
- You can add more, but start with 12 so shapes are consistent

---

## 4. Model: Causal TCN

**Input shape**: `(batch, seq, features)` → transpose to `(batch, features, seq)` for Conv1d

**Baseline architecture**:
- `num_inputs = 12` (features)
- `seq_len = 64`
- `num_channels = [32, 32, 64]` ← 3 TCN blocks
- `kernel_size = 3`
- `dilations = [1, 2, 4]` (implicitly from block index)
- `dropout = 0.1`
- final head: `Linear(64 → 1)` + sigmoid → probability

**Architecture**:
- **TemporalBlock**: Conv1d (causal padding) → ReLU → Dropout → Conv1d → ReLU → Dropout → Residual
- **TCN**: stack TemporalBlocks with increasing dilation, then take last timestep and map to 1 logit

**Why this combo**:
- 3 blocks is a good starter: low risk of overfitting, quick to train
- kernel=3 keeps receptive field tight but dilations let it see farther
- dropout=0.1 is enough regularization for small data

**Training hyperparams (initial)**:
- loss: BCEWithLogitsLoss or sigmoid + BCELoss
- optimizer: Adam(lr=1e-3)
- batch size: 64 or 128 (depends on frequency)
- epochs: start with 5–10 just to see learning
- val split: last 20% of time (walk-forward later)

---

## 5. Output → Risk Map / Signal Shaping

This is the "proprietary risk map" reconstruction.

**Inputs to map**:
- `p` = model probability (0..1)
- `vol` = forecast or rolling volatility for current bar/instrument

**Steps**:

1. **Centering**:
   - `edge = p - 0.5`
   - interpretation: positive → long, negative → short

2. **Neutral zone (dead band)**:
   - `if abs(edge) < deadband → position = 0`
   - hyperparam: `deadband = 0.02` (i.e. need 52%+ "conviction" to act)
   - why: reduces churn and overtrading on noise

3. **Scale edge beyond dead band**:
   - map edge from `[deadband, 0.5]` → `[0, 1]`
   - preserves sign
   - gives smooth position rather than jump

4. **Adversity / Volatility scaling**:
   - compute `vol_scale = vol_target / vol`
   - clamp to `[0.2, 1.5]`
   - multiply position by `vol_scale`
   - hyperparams:
     - `vol_target = 0.01` (example — depends on bar frequency)
     - `clamp_min = 0.2`, `clamp_max = 1.5`
   - why: when market is jumpy, shrink size

5. **Cap exposure**:
   - final: `pos = clip(pos, -1.0, 1.0)`
   - hyperparam: `max_abs_pos = 1.0` (100% notional, or 1x)

**Reference function**:
```python
def risk_map(p, vol, 
             deadband=0.02,
             base_leverage=1.0,
             vol_target=0.01):
    edge = p - 0.5
    if abs(edge) < deadband:
        return 0.0

    sign = 1 if edge > 0 else -1
    mag = (abs(edge) - deadband) / (0.5 - deadband)
    mag = max(0.0, min(1.0, mag))

    vol_scale = vol_target / (vol + 1e-6)
    vol_scale = max(0.2, min(1.5, vol_scale))

    pos = sign * mag * vol_scale * base_leverage
    pos = max(-1.0, min(1.0, pos))
    return pos
```

---

## 6. Backtest Skeleton

**State**:
- equity (start capital)
- position (current target exposure, -1..1)
- price series
- vol series (same frequency)

**Loop logic**:
1. at bar t, build window → model → p_t
2. compute vol_t
3. target_pos = risk_map(p_t, vol_t, ...)
4. cost to move from old pos → new pos (slippage + commission)
5. update position
6. P&L from holding that position into t+1

**Hyperparams**:
- `commission_bps = 0.5–2` bps per side (depends on asset)
- `slippage_bps = 1–3` bps intraday
- why: small but nonzero to avoid fantasy performance

---

## 7. Hyperparameters to Document / Explore

**Data / features**:
- `seq_len`: start 64, try 96, 128
- `n_features`: start 12, expand to 20
- feature normalization: per-feature z-score vs rolling normalization

**Model**:
- `num_channels`: [32, 32, 64] vs [32, 64, 64]
- `kernel_size`: 3 vs 5
- `dropout`: 0.1 vs 0.2
- `dilations`: powers of 2 → [1, 2, 4, 8] if you add a 4th block
- `lr`: 1e-3 → 3e-4
- `batch_size`: 64 vs 128

**Risk map**:
- `deadband`: 0.01–0.05 (controls trading frequency)
- `vol_target`: tied to your bar length (intraday vol smaller)
- `max_abs_pos`: 0.5, 1.0, 1.5 (capital at risk)
- `vol_scale clamps`: (0.2, 1.5) — safety bounds

**Execution / backtest**:
- `cost_bps`: 1–3
- `max_turnover` per bar: optional cap

---

## 8. Why This Combo is Sensible

- Binary next-bar prediction is easy to train and aligns with "probability" language
- TCN fits naturally on fixed windows (64×12) and is genuinely causal — no future leak
- Dead band is a classic way to turn a noisy ML classifier into a sparse trading signal
- Volatility-aware scaling is a simple proxy for "adversity" without needing a whole second model
- Exposure cap prevents one bad probability from overleveraging you
- Small model, small feature set → faster iteration, less overfit, easier to debug
- Everything is modular: you can swap label, features, or risk map independently

---

## 9. Data Horizons (Example)

**Daily bars**:
- 2010–2018: Train/validation
- 2018–2020: Out-of-sample backtest
- 2020–2022: Forward-walk re-train / robustness check
- 2023–2025: Live simulation or production candidate

**Intraday (5-minute)**:
- 2018–2022: Train/validation
- 2022–2023: Out-of-sample backtest
- 2023–2025: Walk-forward live simulation

---

## 10. Layer-by-Layer Build Strategy

Each layer produces something you can see and judge before moving on:

1. **Layer 0**: Config + experiment skeleton
2. **Layer 1**: Data & feature inspection (plots, correlation, label balance)
3. **Layer 2**: Windowing & dataset builder (leak checks)
4. **Layer 3**: Model training (loss curves, calibration)
5. **Layer 4**: Out-of-sample inference (prediction vs reality)
6. **Layer 5**: Risk map / signal shaping (position vs price)
7. **Layer 6**: Backtest engine (equity curve, Sharpe, drawdown)
8. **Layer 7**: Walk-forward robustness (per-era performance)
9. **Layer 8**: Trader dashboard (live view)

---

*This document captures the complete design rationale and serves as a reference for implementation.*

