"""
Benchmark trading strategies for comparison.

Simple strategies to test if ML adds value:
- Buy and hold
- Moving average crossover
- Momentum
- Mean reversion
"""

from typing import Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    """Results from a benchmark strategy."""
    
    name: str
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_turnover: float


def buy_and_hold_benchmark(prices: pd.Series) -> BenchmarkResult:
    """
    Buy and hold benchmark.
    
    Args:
        prices: Price series
        
    Returns:
        Benchmark result
    """
    # Always hold 100%
    positions = pd.Series(1.0, index=prices.index)
    
    # Calculate returns
    returns = prices.pct_change()
    strategy_returns = positions.shift(1) * returns
    
    # Calculate equity
    equity = (1 + strategy_returns).cumprod()
    
    # Metrics
    total_return = equity.iloc[-1] - 1
    n_years = len(prices) / 252
    annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    
    sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
    
    cummax = equity.expanding().max()
    drawdown = (equity - cummax) / cummax
    max_dd = drawdown.min()
    
    return BenchmarkResult(
        name="Buy & Hold",
        total_return=total_return,
        annualized_return=annualized_return,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        win_rate=(returns > 0).sum() / len(returns),
        avg_turnover=0.0,
    )


def ma_crossover_benchmark(
    prices: pd.Series,
    fast_period: int = 20,
    slow_period: int = 50,
) -> BenchmarkResult:
    """
    Moving average crossover strategy.
    
    Args:
        prices: Price series
        fast_period: Fast MA period
        slow_period: Slow MA period
        
    Returns:
        Benchmark result
    """
    # Moving averages
    ma_fast = prices.rolling(fast_period).mean()
    ma_slow = prices.rolling(slow_period).mean()
    
    # Signal: long when fast > slow, flat otherwise
    positions = (ma_fast > ma_slow).astype(float)
    
    # Calculate returns
    returns = prices.pct_change()
    strategy_returns = positions.shift(1) * returns
    
    # Calculate equity
    equity = (1 + strategy_returns).cumprod()
    
    # Metrics
    total_return = equity.iloc[-1] - 1
    n_years = len(prices) / 252
    annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    
    sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
    
    cummax = equity.expanding().max()
    drawdown = (equity - cummax) / cummax
    max_dd = drawdown.min()
    
    # Turnover
    position_changes = np.abs(positions.diff())
    avg_turnover = position_changes.mean()
    
    return BenchmarkResult(
        name=f"MA Crossover ({fast_period}/{slow_period})",
        total_return=total_return,
        annualized_return=annualized_return,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        win_rate=(strategy_returns[strategy_returns != 0] > 0).sum() / (strategy_returns != 0).sum(),
        avg_turnover=avg_turnover,
    )


def momentum_benchmark(
    prices: pd.Series,
    lookback: int = 20,
    threshold: float = 0.0,
) -> BenchmarkResult:
    """
    Simple momentum strategy.
    
    Long if momentum > threshold, short if < -threshold, else flat.
    
    Args:
        prices: Price series
        lookback: Lookback period for momentum
        threshold: Threshold for signal
        
    Returns:
        Benchmark result
    """
    # Momentum
    momentum = prices.pct_change(lookback)
    
    # Positions
    positions = pd.Series(0.0, index=prices.index)
    positions[momentum > threshold] = 1.0
    positions[momentum < -threshold] = -1.0
    
    # Calculate returns
    returns = prices.pct_change()
    strategy_returns = positions.shift(1) * returns
    
    # Calculate equity
    equity = (1 + strategy_returns).cumprod()
    
    # Metrics
    total_return = equity.iloc[-1] - 1
    n_years = len(prices) / 252
    annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    
    sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
    
    cummax = equity.expanding().max()
    drawdown = (equity - cummax) / cummax
    max_dd = drawdown.min()
    
    # Turnover
    position_changes = np.abs(positions.diff())
    avg_turnover = position_changes.mean()
    
    return BenchmarkResult(
        name=f"Momentum ({lookback}d)",
        total_return=total_return,
        annualized_return=annualized_return,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        win_rate=(strategy_returns[strategy_returns != 0] > 0).sum() / max((strategy_returns != 0).sum(), 1),
        avg_turnover=avg_turnover,
    )


def mean_reversion_benchmark(
    prices: pd.Series,
    lookback: int = 20,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
) -> BenchmarkResult:
    """
    Mean reversion strategy using Bollinger Bands logic.
    
    Args:
        prices: Price series
        lookback: Lookback period for mean/std
        entry_threshold: Number of std devs for entry
        exit_threshold: Number of std devs for exit
        
    Returns:
        Benchmark result
    """
    # Bollinger bands
    ma = prices.rolling(lookback).mean()
    std = prices.rolling(lookback).std()
    
    # Z-score
    zscore = (prices - ma) / (std + 1e-6)
    
    # Positions (mean reversion: short when too high, long when too low)
    positions = pd.Series(0.0, index=prices.index)
    
    # Entry signals
    positions[zscore < -entry_threshold] = 1.0  # Long when oversold
    positions[zscore > entry_threshold] = -1.0  # Short when overbought
    
    # Exit signals (move to flat)
    positions[(zscore > -exit_threshold) & (zscore < exit_threshold)] = 0.0
    
    # Forward fill positions
    positions = positions.replace(0, np.nan).fillna(method='ffill').fillna(0)
    
    # Calculate returns
    returns = prices.pct_change()
    strategy_returns = positions.shift(1) * returns
    
    # Calculate equity
    equity = (1 + strategy_returns).cumprod()
    
    # Metrics
    total_return = equity.iloc[-1] - 1
    n_years = len(prices) / 252
    annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    
    sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
    
    cummax = equity.expanding().max()
    drawdown = (equity - cummax) / cummax
    max_dd = drawdown.min()
    
    # Turnover
    position_changes = np.abs(positions.diff())
    avg_turnover = position_changes.mean()
    
    return BenchmarkResult(
        name=f"Mean Reversion (BB {lookback}d)",
        total_return=total_return,
        annualized_return=annualized_return,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        win_rate=(strategy_returns[strategy_returns != 0] > 0).sum() / max((strategy_returns != 0).sum(), 1),
        avg_turnover=avg_turnover,
    )


def run_all_benchmarks(prices: pd.Series) -> pd.DataFrame:
    """
    Run all benchmark strategies.
    
    Args:
        prices: Price series
        
    Returns:
        DataFrame with all benchmark results
    """
    benchmarks = [
        buy_and_hold_benchmark(prices),
        ma_crossover_benchmark(prices, fast_period=20, slow_period=50),
        ma_crossover_benchmark(prices, fast_period=10, slow_period=30),
        momentum_benchmark(prices, lookback=20),
        momentum_benchmark(prices, lookback=60),
        mean_reversion_benchmark(prices, lookback=20),
    ]
    
    # Convert to DataFrame
    results_df = pd.DataFrame([
        {
            'Strategy': b.name,
            'Total Return': f"{b.total_return:.2%}",
            'Ann. Return': f"{b.annualized_return:.2%}",
            'Sharpe': f"{b.sharpe_ratio:.2f}",
            'Max DD': f"{b.max_drawdown:.2%}",
            'Win Rate': f"{b.win_rate:.2%}",
            'Avg Turnover': f"{b.avg_turnover:.2%}",
        }
        for b in benchmarks
    ])
    
    return results_df


def compare_to_benchmarks(
    ml_result: dict,
    prices: pd.Series,
) -> str:
    """
    Compare ML strategy to benchmarks.
    
    Args:
        ml_result: Dictionary with ML strategy metrics
        prices: Price series for benchmarks
        
    Returns:
        Formatted comparison report
    """
    benchmarks_df = run_all_benchmarks(prices)
    
    report = [
        "\n" + "="*70,
        "BENCHMARK COMPARISON",
        "="*70,
        "\nML Strategy Performance:",
        f"  Sharpe Ratio: {ml_result.get('sharpe_ratio', 0):.2f}",
        f"  Total Return: {ml_result.get('total_return', 0):.2%}",
        f"  Max Drawdown: {ml_result.get('max_drawdown', 0):.2%}",
        "\nBenchmark Strategies:",
    ]
    
    report.append("\n" + benchmarks_df.to_string(index=False))
    
    # Analysis
    report.extend([
        "\n" + "="*70,
        "ANALYSIS:",
    ])
    
    ml_sharpe = ml_result.get('sharpe_ratio', 0)
    
    # Get numeric sharpe values from benchmarks
    bh_sharpe = float(benchmarks_df[benchmarks_df['Strategy'] == 'Buy & Hold']['Sharpe'].iloc[0])
    best_benchmark_sharpe = max([float(row['Sharpe']) for _, row in benchmarks_df.iterrows()])
    
    if ml_sharpe > best_benchmark_sharpe:
        report.append("  ✅ ML strategy outperforms all benchmarks")
    elif ml_sharpe > bh_sharpe:
        report.append("  🟡 ML strategy beats buy-and-hold but not all benchmarks")
    else:
        report.append("  ❌ ML strategy underperforms simple benchmarks")
        report.append("  → Consider using simpler strategy or improving model")
    
    report.append("="*70)
    
    return "\n".join(report)

