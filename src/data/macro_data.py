"""
Macro Data: DXY + VIX daily từ Yahoo Finance.
Cache local để không tải lại mỗi lần.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "macro_cache"


def download_macro(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Download daily data từ Yahoo Finance, cache local."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{symbol}_{start}_{end}.csv"

    if cache_file.exists():
        logger.info(f"  📂 Cache hit: {cache_file.name}")
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return df

    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end)
        if len(df) > 0:
            df.to_csv(cache_file)
            logger.info(f"  ✅ Downloaded {symbol}: {len(df)} days")
        return df
    except Exception as e:
        logger.warning(f"  ⚠️ Failed to download {symbol}: {e}")
        return pd.DataFrame()


def get_macro_features(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Lấy DXY + VIX, tính features, return daily DataFrame.
    """
    # DXY = US Dollar Index
    dxy = download_macro("DX-Y.NYB", start_date, end_date)
    # VIX = Volatility Index
    vix = download_macro("^VIX", start_date, end_date)

    features = pd.DataFrame()

    if len(dxy) > 0:
        features["dxy_close"] = dxy["Close"]
        features["dxy_ret_1d"] = dxy["Close"].pct_change() * 100
        features["dxy_ret_5d"] = dxy["Close"].pct_change(5) * 100
        features["dxy_ema_21"] = dxy["Close"].ewm(span=21, adjust=False).mean()
        features["dxy_trend"] = (dxy["Close"] > features["dxy_ema_21"]).astype(float)

    if len(vix) > 0:
        features["vix_close"] = vix["Close"]
        features["vix_high"] = (vix["Close"] > 25).astype(float)  # High fear
        features["vix_extreme"] = (vix["Close"] > 35).astype(float)  # Extreme fear
        features["vix_ret_1d"] = vix["Close"].pct_change() * 100

    features.dropna(inplace=True)
    logger.info(f"  📊 Macro features: {len(features.columns)} cols, {len(features)} days")
    return features


def merge_macro_to_intraday(intraday_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill daily macro vào intraday M5 index."""
    if macro_df.empty:
        logger.warning("  ⚠️ No macro data, skipping")
        return intraday_df

    result = intraday_df.copy()
    # Reindex macro to intraday dates (forward fill)
    macro_reindexed = macro_df.reindex(result.index.date)
    macro_reindexed.index = result.index

    for col in macro_reindexed.columns:
        result[col] = macro_reindexed[col].ffill()

    return result
