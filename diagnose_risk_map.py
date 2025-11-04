#!/usr/bin/env python3
"""
Diagnose risk map issues by analyzing predictions vs actual returns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load backtest results
backtest_path = Path("results/backtests/SPY_daily_backtest_results.csv")
df = pd.read_csv(backtest_path, index_col=0, parse_dates=True)

# Calculate returns
df['returns'] = df['price'].pct_change()
df['actual_direction'] = (df['returns'] > 0).astype(int)  # 1 = up, 0 = down

# Remove NaN
df = df.dropna()

print("\n" + "="*80)
print("🔍 RISK MAP DIAGNOSIS")
print("="*80)

# 1. Check prediction distribution
print("\n📊 Prediction Distribution:")
print(f"  Mean probability: {df['probability'].mean():.4f}")
print(f"  Median probability: {df['probability'].median():.4f}")
print(f"  Std probability: {df['probability'].std():.4f}")
print(f"  Min: {df['probability'].min():.4f}")
print(f"  Max: {df['probability'].max():.4f}")

# Count by bins
print(f"\n  Probability bins:")
print(f"    < 0.1: {(df['probability'] < 0.1).sum()} ({(df['probability'] < 0.1).mean()*100:.1f}%)")
print(f"    0.1-0.4: {((df['probability'] >= 0.1) & (df['probability'] < 0.4)).sum()} ({((df['probability'] >= 0.1) & (df['probability'] < 0.4)).mean()*100:.1f}%)")
print(f"    0.4-0.6: {((df['probability'] >= 0.4) & (df['probability'] < 0.6)).sum()} ({((df['probability'] >= 0.4) & (df['probability'] < 0.6)).mean()*100:.1f}%)")
print(f"    0.6-0.9: {((df['probability'] >= 0.6) & (df['probability'] < 0.9)).sum()} ({((df['probability'] >= 0.6) & (df['probability'] < 0.9)).mean()*100:.1f}%)")
print(f"    > 0.9: {(df['probability'] > 0.9).sum()} ({(df['probability'] > 0.9).mean()*100:.1f}%)")

# 2. Check if predictions align with returns
print("\n🎯 Prediction Alignment:")
predicted_up = df['probability'] > 0.5
actual_up = df['actual_direction'] == 1

correct = predicted_up == actual_up
print(f"  Hit rate (model): {correct.mean()*100:.2f}%")
print(f"  Predicted UP: {predicted_up.sum()} times")
print(f"  Actually UP: {actual_up.sum()} times")
print(f"  Correctly predicted UP: {(predicted_up & actual_up).sum()} times")
print(f"  Correctly predicted DOWN: {(~predicted_up & ~actual_up).sum()} times")

# 3. Check position distribution
print("\n📈 Position Distribution:")
print(f"  Mean position: {df['position'].mean():.4f}")
print(f"  Long: {(df['position'] > 0).sum()} ({(df['position'] > 0).mean()*100:.1f}%)")
print(f"  Short: {(df['position'] < 0).sum()} ({(df['position'] < 0).mean()*100:.1f}%)")
print(f"  Flat: {(df['position'] == 0).sum()} ({(df['position'] == 0).mean()*100:.1f}%)")

# 4. Check alignment between positions and returns
print("\n💰 Position P&L Analysis:")
long_returns = df[df['position'] > 0]['returns']
short_returns = df[df['position'] < 0]['returns']

if len(long_returns) > 0:
    long_win_rate = (long_returns > 0).mean() * 100
    print(f"  LONG positions:")
    print(f"    Count: {len(long_returns)}")
    print(f"    Win rate: {long_win_rate:.2f}%")
    print(f"    Avg return: {long_returns.mean()*100:.4f}%")
    
if len(short_returns) > 0:
    # For shorts, we make money when price goes DOWN
    short_win_rate = (short_returns < 0).mean() * 100
    print(f"  SHORT positions:")
    print(f"    Count: {len(short_returns)}")
    print(f"    Win rate: {short_win_rate:.2f}% (price went down)")
    print(f"    Avg return: {short_returns.mean()*100:.4f}%")
    print(f"    Avg P&L: {-short_returns.mean()*100:.4f}% (inverted for shorts)")

# 5. Check if probabilities are inverted
print("\n🔄 Inversion Check:")
# If high probability → price actually goes DOWN, predictions are inverted
high_prob_mask = df['probability'] > 0.6
low_prob_mask = df['probability'] < 0.4

if high_prob_mask.sum() > 0:
    high_prob_up_rate = df[high_prob_mask]['actual_direction'].mean()
    print(f"  When prob > 0.6 → price goes UP {high_prob_up_rate*100:.1f}% of time")
    
if low_prob_mask.sum() > 0:
    low_prob_up_rate = df[low_prob_mask]['actual_direction'].mean()
    print(f"  When prob < 0.4 → price goes UP {low_prob_up_rate*100:.1f}% of time")

if high_prob_mask.sum() > 0 and low_prob_mask.sum() > 0:
    if high_prob_up_rate < low_prob_up_rate:
        print("\n  ⚠️  WARNING: Predictions appear INVERTED!")
        print("     High probabilities lead to DOWN moves")
        print("     Low probabilities lead to UP moves")
    elif high_prob_up_rate > low_prob_up_rate:
        print("\n  ✅ Predictions are correctly aligned")
    else:
        print("\n  ⚠️  Predictions have no signal (both ~50%)")

# 6. Test inverting predictions
print("\n🔧 Testing Inverted Risk Map:")
df['inverted_prob'] = 1 - df['probability']
inverted_predicted_up = df['inverted_prob'] > 0.5
inverted_correct = inverted_predicted_up == actual_up
print(f"  Hit rate with INVERTED probs: {inverted_correct.mean()*100:.2f}%")

# 7. Recommendations
print("\n" + "="*80)
print("💡 RECOMMENDATIONS")
print("="*80)

if correct.mean() < 0.45:
    print("\n❌ Model predictions are TERRIBLE (< 45% hit rate)")
    print("   → Try inverting the risk map (1 - probability)")
    print("   → Or retrain the model")
elif correct.mean() < 0.50:
    print("\n⚠️  Model predictions are below random")
    print("   → Consider inverting predictions")
elif correct.mean() > 0.52:
    print("\n✅ Model predictions are above random")
    print("   → Problem is likely in position sizing, not predictions")
else:
    print("\n⚠️  Model predictions are random (50%)")
    print("   → Model has no signal")

# Check if inverting helps
if inverted_correct.mean() > correct.mean() + 0.05:
    print(f"\n🔄 INVERTING PREDICTIONS HELPS!")
    print(f"   Current hit rate: {correct.mean()*100:.2f}%")
    print(f"   Inverted hit rate: {inverted_correct.mean()*100:.2f}%")
    print(f"   → Change risk map to use (1 - probability)")

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Probability distribution
axes[0, 0].hist(df['probability'], bins=50, alpha=0.7, edgecolor='black')
axes[0, 0].axvline(0.5, color='red', linestyle='--', label='Neutral')
axes[0, 0].set_xlabel('Probability')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_title('Probability Distribution')
axes[0, 0].legend()

# 2. Probability vs Actual Direction
prob_bins = np.linspace(0, 1, 11)
bin_centers = (prob_bins[:-1] + prob_bins[1:]) / 2
hit_rates = []
for i in range(len(prob_bins)-1):
    mask = (df['probability'] >= prob_bins[i]) & (df['probability'] < prob_bins[i+1])
    if mask.sum() > 0:
        hit_rates.append(df[mask]['actual_direction'].mean())
    else:
        hit_rates.append(np.nan)

axes[0, 1].plot(bin_centers, hit_rates, 'o-', linewidth=2, markersize=8)
axes[0, 1].axhline(0.5, color='red', linestyle='--', label='Random')
axes[0, 1].plot([0, 1], [0, 1], 'g--', alpha=0.5, label='Perfect calibration')
axes[0, 1].set_xlabel('Predicted Probability')
axes[0, 1].set_ylabel('Actual Up Rate')
axes[0, 1].set_title('Calibration: Prediction vs Reality')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# 3. Position distribution
axes[1, 0].hist(df['position'], bins=50, alpha=0.7, edgecolor='black')
axes[1, 0].axvline(0, color='red', linestyle='--')
axes[1, 0].set_xlabel('Position Size')
axes[1, 0].set_ylabel('Count')
axes[1, 0].set_title('Position Distribution')

# 4. Returns by position direction
long_rets = df[df['position'] > 0]['returns'].values
short_rets = df[df['position'] < 0]['returns'].values

axes[1, 1].hist(long_rets, bins=30, alpha=0.5, label=f'Long (n={len(long_rets)})', color='green')
axes[1, 1].hist(short_rets, bins=30, alpha=0.5, label=f'Short (n={len(short_rets)})', color='red')
axes[1, 1].axvline(0, color='black', linestyle='--')
axes[1, 1].set_xlabel('Return')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title('Return Distribution by Position Type')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('results/figures/risk_map_diagnosis.png', dpi=150, bbox_inches='tight')
print(f"\n📊 Diagnostic plots saved to: results/figures/risk_map_diagnosis.png")

print("\n" + "="*80 + "\n")

