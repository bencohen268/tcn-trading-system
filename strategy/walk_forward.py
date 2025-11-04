"""
Walk-forward testing framework (Layer 7).

Tests strategy robustness by:
- Rolling window retraining
- Out-of-sample testing on multiple periods
- Aggregating results across time
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward testing."""
    
    train_window_days: int = 504  # ~2 years
    val_window_days: int = 126  # ~6 months
    test_window_days: int = 126  # ~6 months
    step_days: int = 63  # ~3 months (how far to step forward)
    min_train_samples: int = 1000
    retrain_every_window: bool = True


@dataclass
class WalkForwardResult:
    """Results from one walk-forward window."""
    
    window_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    
    # Model metrics
    train_auc: float
    val_auc: float
    test_auc: float
    
    # Trading metrics
    sharpe_ratio: float
    total_return: float
    max_drawdown: float
    win_rate: float
    
    # Additional info
    n_trades: int
    avg_position: float


class WalkForwardTester:
    """
    Walk-forward testing framework.
    
    Splits data into rolling windows, trains model on each window,
    tests on subsequent period, aggregates results.
    """
    
    def __init__(
        self,
        config: WalkForwardConfig,
        checkpoint_dir: Path,
    ):
        """
        Initialize walk-forward tester.
        
        Args:
            config: Walk-forward configuration
            checkpoint_dir: Directory to save/load checkpoints
        """
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: List[WalkForwardResult] = []
    
    def create_windows(
        self,
        df: pd.DataFrame,
    ) -> List[Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]]:
        """
        Create rolling windows for walk-forward analysis.
        
        Args:
            df: Full DataFrame with datetime index
            
        Returns:
            List of window specifications
        """
        windows = []
        
        # Get date range
        start_date = df.index[0]
        end_date = df.index[-1]
        
        current_train_start = start_date
        
        window_id = 0
        while True:
            # Define window boundaries
            train_start = current_train_start
            train_end_idx = df.index.get_loc(train_start) + self.config.train_window_days
            
            if train_end_idx >= len(df):
                break
            
            train_end = df.index[train_end_idx]
            
            val_end_idx = train_end_idx + self.config.val_window_days
            if val_end_idx >= len(df):
                break
            
            val_start = df.index[train_end_idx + 1]
            val_end = df.index[val_end_idx]
            
            test_end_idx = val_end_idx + self.config.test_window_days
            if test_end_idx >= len(df):
                break
            
            test_start = df.index[val_end_idx + 1]
            test_end = df.index[test_end_idx]
            
            # Store window
            windows.append({
                'id': window_id,
                'train': (train_start, train_end),
                'val': (val_start, val_end),
                'test': (test_start, test_end),
            })
            
            # Step forward
            current_train_start = df.index[
                df.index.get_loc(current_train_start) + self.config.step_days
            ]
            window_id += 1
        
        print(f"Created {len(windows)} walk-forward windows")
        for i, window in enumerate(windows[:3]):  # Show first 3
            print(f"  Window {i}:")
            print(f"    Train: {window['train'][0].date()} to {window['train'][1].date()}")
            print(f"    Val:   {window['val'][0].date()} to {window['val'][1].date()}")
            print(f"    Test:  {window['test'][0].date()} to {window['test'][1].date()}")
        
        if len(windows) > 3:
            print(f"  ... {len(windows) - 3} more windows")
        
        return windows
    
    def run_window(
        self,
        window: Dict,
        df: pd.DataFrame,
        model_factory,  # Function that creates a new model
        train_fn,  # Function that trains model
        evaluate_fn,  # Function that evaluates model
        backtest_fn,  # Function that runs backtest
    ) -> WalkForwardResult:
        """
        Run single walk-forward window.
        
        Args:
            window: Window specification
            df: Full DataFrame
            model_factory: Function to create model
            train_fn: Training function
            evaluate_fn: Evaluation function
            backtest_fn: Backtest function
            
        Returns:
            WalkForwardResult for this window
        """
        window_id = window['id']
        print(f"\n{'='*70}")
        print(f"WALK-FORWARD WINDOW {window_id}")
        print(f"{'='*70}")
        
        # Extract window data
        train_df = df.loc[window['train'][0]:window['train'][1]]
        val_df = df.loc[window['val'][0]:window['val'][1]]
        test_df = df.loc[window['test'][0]:window['test'][1]]
        
        print(f"Train: {len(train_df)} samples")
        print(f"Val:   {len(val_df)} samples")
        print(f"Test:  {len(test_df)} samples")
        
        # Create model
        model = model_factory()
        
        # Train (or load if not retraining)
        checkpoint_path = self.checkpoint_dir / f"window_{window_id}_model.pt"
        
        if self.config.retrain_every_window or not checkpoint_path.exists():
            print("Training model...")
            train_history = train_fn(model, train_df, val_df)
            
            # Save checkpoint
            torch.save({
                'model_state_dict': model.state_dict(),
                'window_id': window_id,
                'history': train_history,
            }, checkpoint_path)
        else:
            print(f"Loading existing model from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            train_history = checkpoint.get('history', {})
        
        # Evaluate on test set
        print("Evaluating on test set...")
        test_metrics = evaluate_fn(model, test_df)
        
        # Run backtest
        print("Running backtest...")
        backtest_result = backtest_fn(model, test_df)
        
        # Create result
        result = WalkForwardResult(
            window_id=window_id,
            train_start=window['train'][0],
            train_end=window['train'][1],
            test_start=window['test'][0],
            test_end=window['test'][1],
            train_auc=train_history.get('train_auc', [0])[-1] if train_history else 0,
            val_auc=train_history.get('val_auc', [0])[-1] if train_history else 0,
            test_auc=test_metrics.get('auc', 0),
            sharpe_ratio=backtest_result.sharpe_ratio,
            total_return=backtest_result.total_return,
            max_drawdown=backtest_result.max_drawdown,
            win_rate=backtest_result.win_rate,
            n_trades=len(backtest_result.trades),
            avg_position=np.abs(backtest_result.positions).mean(),
        )
        
        self.results.append(result)
        
        print(f"\nWindow {window_id} Results:")
        print(f"  Val AUC: {result.val_auc:.4f}")
        print(f"  Test AUC: {result.test_auc:.4f}")
        print(f"  Sharpe: {result.sharpe_ratio:.2f}")
        print(f"  Return: {result.total_return:.2%}")
        
        return result
    
    def aggregate_results(self) -> Dict:
        """
        Aggregate results across all windows.
        
        Returns:
            Dictionary with aggregated metrics
        """
        if not self.results:
            return {}
        
        # Convert to DataFrame for easier analysis
        results_df = pd.DataFrame([
            {
                'window_id': r.window_id,
                'test_start': r.test_start,
                'test_end': r.test_end,
                'val_auc': r.val_auc,
                'test_auc': r.test_auc,
                'sharpe': r.sharpe_ratio,
                'return': r.total_return,
                'max_dd': r.max_drawdown,
                'win_rate': r.win_rate,
            }
            for r in self.results
        ])
        
        # Aggregate statistics
        agg = {
            'n_windows': len(self.results),
            'mean_val_auc': results_df['val_auc'].mean(),
            'std_val_auc': results_df['val_auc'].std(),
            'mean_test_auc': results_df['test_auc'].mean(),
            'std_test_auc': results_df['test_auc'].std(),
            'mean_sharpe': results_df['sharpe'].mean(),
            'std_sharpe': results_df['sharpe'].std(),
            'mean_return': results_df['return'].mean(),
            'std_return': results_df['return'].std(),
            'mean_max_dd': results_df['max_dd'].mean(),
            'worst_max_dd': results_df['max_dd'].min(),
            'pct_positive_sharpe': (results_df['sharpe'] > 0).sum() / len(results_df),
            'pct_positive_return': (results_df['return'] > 0).sum() / len(results_df),
        }
        
        return agg, results_df
    
    def print_summary(self):
        """Print walk-forward summary."""
        agg, results_df = self.aggregate_results()
        
        print("\n" + "="*70)
        print("WALK-FORWARD ANALYSIS SUMMARY")
        print("="*70)
        
        print(f"\nNumber of Windows: {agg['n_windows']}")
        
        print(f"\nModel Performance:")
        print(f"  Val AUC:  {agg['mean_val_auc']:.4f} ± {agg['std_val_auc']:.4f}")
        print(f"  Test AUC: {agg['mean_test_auc']:.4f} ± {agg['std_test_auc']:.4f}")
        
        print(f"\nTrading Performance:")
        print(f"  Sharpe Ratio: {agg['mean_sharpe']:.2f} ± {agg['std_sharpe']:.2f}")
        print(f"  Total Return: {agg['mean_return']:.2%} ± {agg['std_return']:.2%}")
        print(f"  Avg Max DD:   {agg['mean_max_dd']:.2%}")
        print(f"  Worst Max DD: {agg['worst_max_dd']:.2%}")
        
        print(f"\nConsistency:")
        print(f"  Positive Sharpe: {agg['pct_positive_sharpe']:.1%} of windows")
        print(f"  Positive Return: {agg['pct_positive_return']:.1%} of windows")
        
        print("\n" + "="*70)
        
        # Per-window breakdown
        print("\nPer-Window Breakdown:")
        print(results_df.to_string())
        
        print("\n" + "="*70)
    
    def save_results(self, output_path: Path):
        """Save results to CSV."""
        _, results_df = self.aggregate_results()
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")

