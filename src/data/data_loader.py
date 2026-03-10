"""
Data Loader cho XAUUSD M1 CSV files.
Đọc data từ MT5 export format: date,time,open,high,low,close,volume (không header).
Lấy cảm hứng từ forbbiden403/tradingbot data pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Union
import logging

logger = logging.getLogger(__name__)


def load_csv(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load 1 file CSV XAUUSD M1.
    Format: date,time,open,high,low,close,volume (không header)
    Ví dụ: 2024.01.02,01:00,2062.77,2064.51,2062.68,2063.9,37
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {filepath}")

    df = pd.read_csv(
        filepath,
        header=None,
        names=["date", "time", "open", "high", "low", "close", "volume"],
        dtype={
            "date": str,
            "time": str,
            "open": float,
            "high": float,
            "low": float,
            "close": float,
            "volume": float,
        },
    )

    # Parse datetime
    df["datetime"] = pd.to_datetime(
        df["date"] + " " + df["time"], format="%Y.%m.%d %H:%M"
    )
    df.set_index("datetime", inplace=True)
    df.drop(columns=["date", "time"], inplace=True)
    df.sort_index(inplace=True)

    # Loại bỏ duplicate timestamps
    df = df[~df.index.duplicated(keep="first")]

    logger.info(f"✅ Loaded {filepath.name}: {len(df)} candles, {df.index[0]} → {df.index[-1]}")
    return df


def load_multiple_csv(
    data_dir: Union[str, Path],
    pattern: str = "XAUUSD_*.csv",
    exclude_patterns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load nhiều file CSV và merge lại.
    Mặc định exclude các file '*_all.csv' để tránh duplicate.
    """
    data_dir = Path(data_dir)
    if exclude_patterns is None:
        exclude_patterns = ["*_all.csv"]

    files = sorted(data_dir.glob(pattern))

    # Filter exclude patterns
    for ep in exclude_patterns:
        exclude_files = set(data_dir.glob(ep))
        files = [f for f in files if f not in exclude_files]

    if not files:
        raise FileNotFoundError(f"Không tìm thấy file nào trong {data_dir} với pattern {pattern}")

    logger.info(f"📂 Tìm thấy {len(files)} files để load")

    dfs = []
    for f in files:
        try:
            df = load_csv(f)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"⚠️ Lỗi khi load {f.name}: {e}")

    if not dfs:
        raise ValueError("Không load được file nào!")

    combined = pd.concat(dfs)
    combined = combined[~combined.index.duplicated(keep="first")]
    combined.sort_index(inplace=True)

    logger.info(
        f"✅ Combined: {len(combined)} candles, "
        f"{combined.index[0]} → {combined.index[-1]}"
    )
    return combined


def load_year_data(
    data_dir: Union[str, Path], year: int
) -> pd.DataFrame:
    """Load toàn bộ data của 1 năm."""
    # Thử dùng file *_all.csv trước (nhanh hơn)
    all_file = Path(data_dir) / f"XAUUSD_{year}_all.csv"
    if all_file.exists():
        return load_csv(all_file)

    # Nếu không có thì load từng tháng
    return load_multiple_csv(
        data_dir,
        pattern=f"XAUUSD_{year}_*.csv",
        exclude_patterns=[],
    )


def load_train_test_data(
    data_dir: Union[str, Path],
    train_years: List[int] = [2024],
    test_years: List[int] = [2025],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load data chia train/test theo năm.
    Mặc định: Train trên 2024, Test trên 2025.
    """
    train_dfs = [load_year_data(data_dir, y) for y in train_years]
    test_dfs = [load_year_data(data_dir, y) for y in test_years]

    train_data = pd.concat(train_dfs).sort_index()
    test_data = pd.concat(test_dfs).sort_index()

    train_data = train_data[~train_data.index.duplicated(keep="first")]
    test_data = test_data[~test_data.index.duplicated(keep="first")]

    logger.info(
        f"📊 Train: {len(train_data)} candles ({train_years}), "
        f"Test: {len(test_data)} candles ({test_years})"
    )
    return train_data, test_data


def validate_data(df: pd.DataFrame) -> dict:
    """Kiểm tra chất lượng data."""
    stats = {
        "total_candles": len(df),
        "date_range": f"{df.index[0]} → {df.index[-1]}",
        "missing_values": df.isnull().sum().to_dict(),
        "price_range": f"{df['low'].min():.2f} → {df['high'].max():.2f}",
        "avg_volume": df["volume"].mean(),
        "zero_volume_pct": (df["volume"] == 0).mean() * 100,
    }

    # Check for gaps
    time_diff = df.index.to_series().diff()
    expected_gap = pd.Timedelta(minutes=1)
    large_gaps = time_diff[time_diff > pd.Timedelta(hours=4)]
    stats["large_gaps_count"] = len(large_gaps)

    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    DATA_DIR = Path(__file__).resolve().parents[2]  # Project root

    # Test loading
    df = load_csv(DATA_DIR / "XAUUSD_2024_01.csv")
    print(f"\n📊 Data shape: {df.shape}")
    print(f"📊 Columns: {df.columns.tolist()}")
    print(f"📊 Head:\n{df.head()}")
    print(f"\n📊 Stats: {validate_data(df)}")
