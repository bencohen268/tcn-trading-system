"""Data loading utilities."""

from pathlib import Path
from typing import Optional, Union
import pandas as pd
import yfinance as yf


def load_raw_data(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str = "1d",
    source: str = "yahoo",
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load raw OHLCV data for a symbol.
    
    Args:
        symbol: Ticker symbol (e.g., "SPY")
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        interval: Data interval (1m, 5m, 1h, 1d, etc.)
        source: Data source ("yahoo" or "file")
        cache_dir: Optional directory to cache downloaded data
        
    Returns:
        DataFrame with columns: open, high, low, close, volume
        Index: datetime
    """
    if source == "yahoo":
        return _load_from_yahoo(symbol, start_date, end_date, interval, cache_dir)
    elif source == "file":
        # Load from CSV file
        if cache_dir is None:
            raise ValueError("cache_dir must be provided for file source")
        filepath = cache_dir / f"{symbol}_{interval}.csv"
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        return df.loc[start_date:end_date]
    else:
        raise ValueError(f"Unknown data source: {source}")


def _load_from_yahoo(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Download data from Yahoo Finance."""
    print(f"Downloading {symbol} data from Yahoo Finance...")
    
    # Download data
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date, interval=interval)
    
    if df.empty:
        raise ValueError(f"No data found for {symbol} from {start_date} to {end_date}")
    
    # Standardize column names
    df.columns = df.columns.str.lower()
    
    # Keep only OHLCV columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    df = df[required_cols]
    
    # Cache if requested
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{symbol}_{interval}.csv"
        df.to_csv(cache_file)
        print(f"Cached data to {cache_file}")
    
    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    
    return df


def save_processed_data(
    df: pd.DataFrame,
    name: str,
    output_dir: Path,
    format: str = "parquet",
) -> Path:
    """
    Save processed data to disk.
    
    Args:
        df: DataFrame to save
        name: Base name for the file
        output_dir: Output directory
        format: File format ("parquet" or "csv")
        
    Returns:
        Path to saved file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if format == "parquet":
        filepath = output_dir / f"{name}.parquet"
        df.to_parquet(filepath, compression='snappy')
    elif format == "csv":
        filepath = output_dir / f"{name}.csv"
        df.to_csv(filepath)
    else:
        raise ValueError(f"Unknown format: {format}")
    
    print(f"Saved {len(df)} rows to {filepath}")
    return filepath


def load_processed_data(
    name: str,
    data_dir: Path,
    format: str = "parquet",
) -> pd.DataFrame:
    """
    Load processed data from disk.
    
    Args:
        name: Base name of the file
        data_dir: Data directory
        format: File format ("parquet" or "csv")
        
    Returns:
        DataFrame
    """
    if format == "parquet":
        filepath = data_dir / f"{name}.parquet"
        df = pd.read_parquet(filepath)
    elif format == "csv":
        filepath = data_dir / f"{name}.csv"
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    else:
        raise ValueError(f"Unknown format: {format}")
    
    print(f"Loaded {len(df)} rows from {filepath}")
    return df

