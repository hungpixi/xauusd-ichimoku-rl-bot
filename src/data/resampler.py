"""
Resampler: Chuyển đổi data M1 sang các timeframe cao hơn.
M1 → M5, M15, H1, H4, D1
Lấy cảm hứng từ forbbiden403/tradingbot multi-timeframe analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict


# Mapping tên timeframe → pandas resample rule
TIMEFRAME_MAP = {
    "M1": "1min",
    "M5": "5min",
    "M15": "15min",
    "M30": "30min",
    "H1": "1h",
    "H4": "4h",
    "D1": "1D",
    "W1": "1W",
}


def resample_ohlcv(df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    """
    Resample OHLCV data sang timeframe mới.
    
    Args:
        df: DataFrame có columns [open, high, low, close, volume] và datetime index
        target_tf: Timeframe đích (M5, M15, H1, H4, D1, ...)
    
    Returns:
        DataFrame resampled
    """
    if target_tf not in TIMEFRAME_MAP:
        raise ValueError(f"Timeframe không hợp lệ: {target_tf}. Hỗ trợ: {list(TIMEFRAME_MAP.keys())}")

    rule = TIMEFRAME_MAP[target_tf]

    resampled = df.resample(rule).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )

    # Loại bỏ candles NaN (weekend, gaps)
    resampled.dropna(subset=["open", "close"], inplace=True)

    return resampled


def create_multi_timeframe(
    df_m1: pd.DataFrame,
    timeframes: list = None,
) -> Dict[str, pd.DataFrame]:
    """
    Tạo dict chứa data từ nhiều timeframe.
    
    Args:
        df_m1: DataFrame M1 gốc
        timeframes: List timeframes cần tạo, mặc định ["M5", "M15", "H1", "H4", "D1"]
    
    Returns:
        Dict[timeframe_name, DataFrame]
    """
    if timeframes is None:
        timeframes = ["M5", "M15", "H1", "H4", "D1"]

    result = {"M1": df_m1}

    for tf in timeframes:
        if tf == "M1":
            continue
        result[tf] = resample_ohlcv(df_m1, tf)

    return result


def merge_multi_timeframe_features(
    df_base: pd.DataFrame,
    mtf_data: Dict[str, pd.DataFrame],
    base_tf: str = "M5",
    higher_tfs: list = None,
) -> pd.DataFrame:
    """
    Merge features từ timeframe cao hơn vào base timeframe.
    Sử dụng forward fill để tránh lookahead bias.
    
    Args:
        df_base: DataFrame base timeframe (đã có features)
        mtf_data: Dict từ create_multi_timeframe
        base_tf: Tên base timeframe
        higher_tfs: List timeframe cao hơn để merge
    
    Returns:
        DataFrame base với cột từ timeframe cao hơn
    """
    if higher_tfs is None:
        higher_tfs = ["H1", "H4", "D1"]

    result = df_base.copy()

    for tf in higher_tfs:
        if tf not in mtf_data:
            continue

        htf = mtf_data[tf]

        # Reindex sang base timeframe dùng forward fill
        cols_to_merge = [c for c in htf.columns if c in ["open", "high", "low", "close", "volume"]]
        htf_reindexed = htf[cols_to_merge].reindex(result.index, method="ffill")

        # Rename columns thêm prefix
        htf_reindexed.columns = [f"{tf}_{c}" for c in htf_reindexed.columns]

        result = pd.concat([result, htf_reindexed], axis=1)

    return result


if __name__ == "__main__":
    from data_loader import load_csv
    from pathlib import Path

    DATA_DIR = Path(__file__).resolve().parents[2]
    df = load_csv(DATA_DIR / "XAUUSD_2024_01.csv")

    # Test resample
    df_h1 = resample_ohlcv(df, "H1")
    print(f"M1: {len(df)} candles → H1: {len(df_h1)} candles")
    print(f"H1 Head:\n{df_h1.head()}")

    # Test multi-timeframe
    mtf = create_multi_timeframe(df)
    for tf, data in mtf.items():
        print(f"{tf}: {len(data)} candles")
