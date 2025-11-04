"""
Backtest engine with transaction costs.

Simulates P&L from a series of positions and prices,
accounting for:
- Position changes (turnover)
- Commission costs
- Slippage
"""

from typing import Optional, Dict
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    
    equity_curve: pd.Series
    positions: pd.Series
    returns: pd.Series
    trades: pd.DataFrame
    
    # Performance metrics
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_turnover: float
    
    # Cost analysis
    total_commissions: float
    total_slippage: float
    total_costs: float
    
    def summary(self) -> str:
        """Generate summary report."""
        lines = [
            "="*70,
            "BACKTEST SUMMARY",
            "="*70,
            f"\nReturns:",
            f"  Total Return:       {self.total_return:>8.2%}",
            f"  Annualized Return:  {self.annualized_return:>8.2%}",
            f"  Sharpe Ratio:       {self.sharpe_ratio:>8.2f}",
            f"  Sortino Ratio:      {self.sortino_ratio:>8.2f}",
            f"\nRisk:",
            f"  Max Drawdown:       {self.max_drawdown:>8.2%}",
            f"\nTrading:",
            f"  Win Rate:           {self.win_rate:>8.2%}",
            f"  Profit Factor:      {self.profit_factor:>8.2f}",
            f"  Avg Turnover:       {self.avg_turnover:>8.2%}",
            f"\nCosts:",
            f"  Total Commissions:  ${self.total_commissions:>8,.2f}",
            f"  Total Slippage:     ${self.total_slippage:>8,.2f}",
            f"  Total Costs:        ${self.total_costs:>8,.2f}",
            "="*70,
        ]
        return "\n".join(lines)


class Backtester:
    """
    Backtest engine for trading strategies.
    
    Simulates trading a single asset with:
    - Variable position sizes
    - Transaction costs
    - Slippage
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_bps: float = 1.0,
        slippage_bps: float = 0.5,
        execution_delay: int = 1,
    ):
        """
        Initialize backtester.
        
        Args:
            initial_capital: Starting capital
            commission_bps: Commission in basis points per side
            slippage_bps: Slippage in basis points
            execution_delay: Bars between signal and execution
        """
        self.initial_capital = initial_capital
        self.commission_bps = commission_bps / 10000  # Convert to fraction
        self.slippage_bps = slippage_bps / 10000
        self.execution_delay = execution_delay
    
    def run(
        self,
        prices: pd.Series,
        target_positions: pd.Series,
        volatility: Optional[pd.Series] = None,
    ) -> BacktestResult:
        """
        Run backtest.
        
        Args:
            prices: Series of prices (close prices)
            target_positions: Series of target positions (-1 to 1)
            volatility: Optional volatility series (for analysis)
            
        Returns:
            BacktestResult object
        """
        # Align indices
        common_idx = prices.index.intersection(target_positions.index)
        prices = prices.loc[common_idx]
        target_positions = target_positions.loc[common_idx]
        
        # Initialize state
        n_bars = len(prices)
        equity = np.zeros(n_bars)
        equity[0] = self.initial_capital
        
        positions = np.zeros(n_bars)
        returns = np.zeros(n_bars)
        
        commissions = np.zeros(n_bars)
        slippage = np.zeros(n_bars)
        
        # Track trades
        trade_log = []
        
        for t in range(1, n_bars):
            # Get target position from execution_delay bars ago
            signal_idx = max(0, t - self.execution_delay)
            target_pos = target_positions.iloc[signal_idx]
            
            # Current position
            current_pos = positions[t-1]
            
            # Calculate position change (turnover)
            pos_change = target_pos - current_pos
            
            # Transaction costs
            if pos_change != 0:
                # Commission on the trade size
                trade_size = abs(pos_change) * equity[t-1]
                comm = trade_size * self.commission_bps
                slip = trade_size * self.slippage_bps
                
                commissions[t] = comm
                slippage[t] = slip
                
                # Log trade
                trade_log.append({
                    'timestamp': prices.index[t],
                    'from_position': current_pos,
                    'to_position': target_pos,
                    'change': pos_change,
                    'price': prices.iloc[t],
                    'commission': comm,
                    'slippage': slip,
                })
            
            # Update position
            positions[t] = target_pos
            
            # Calculate return
            # P&L from holding position from t-1 to t
            price_return = (prices.iloc[t] - prices.iloc[t-1]) / prices.iloc[t-1]
            position_return = positions[t-1] * price_return
            
            # Costs reduce equity
            costs = commissions[t] + slippage[t]
            
            # Update equity
            equity[t] = equity[t-1] * (1 + position_return) - costs
            returns[t] = equity[t] / equity[t-1] - 1
        
        # Create result DataFrames
        equity_series = pd.Series(equity, index=prices.index)
        positions_series = pd.Series(positions, index=prices.index)
        returns_series = pd.Series(returns, index=prices.index)
        trades_df = pd.DataFrame(trade_log)
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            equity_series,
            returns_series,
            positions_series,
            commissions,
            slippage,
        )
        
        return BacktestResult(
            equity_curve=equity_series,
            positions=positions_series,
            returns=returns_series,
            trades=trades_df,
            **metrics,
        )
    
    def _calculate_metrics(
        self,
        equity: pd.Series,
        returns: pd.Series,
        positions: pd.Series,
        commissions: np.ndarray,
        slippage: np.ndarray,
    ) -> Dict:
        """Calculate performance metrics."""
        # Returns
        total_return = (equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0]
        
        # Annualized return (assume daily data)
        n_years = len(equity) / 252
        annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        
        # Sharpe ratio (annualized)
        returns_clean = returns[returns != 0]
        if len(returns_clean) > 1 and returns_clean.std() > 0:
            sharpe_ratio = returns_clean.mean() / returns_clean.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns_clean[returns_clean < 0]
        if len(downside_returns) > 1 and downside_returns.std() > 0:
            sortino_ratio = returns_clean.mean() / downside_returns.std() * np.sqrt(252)
        else:
            sortino_ratio = 0.0
        
        # Maximum drawdown
        cummax = equity.expanding().max()
        drawdown = (equity - cummax) / cummax
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_days = (returns > 0).sum()
        total_days = (returns != 0).sum()
        win_rate = winning_days / total_days if total_days > 0 else 0
        
        # Profit factor
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        # Turnover
        position_changes = np.abs(np.diff(positions.values, prepend=0))
        avg_turnover = position_changes.mean()
        
        # Costs
        total_commissions = commissions.sum()
        total_slippage = slippage.sum()
        total_costs = total_commissions + total_slippage
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_turnover': avg_turnover,
            'total_commissions': total_commissions,
            'total_slippage': total_slippage,
            'total_costs': total_costs,
        }
    
    @classmethod
    def from_config(cls, config: dict) -> 'Backtester':
        """Create Backtester from configuration dictionary."""
        backtest_config = config['backtest']
        
        return cls(
            initial_capital=backtest_config['initial_capital'],
            commission_bps=backtest_config['commission_bps'],
            slippage_bps=backtest_config['slippage_bps'],
            execution_delay=backtest_config['execution_delay'],
        )

