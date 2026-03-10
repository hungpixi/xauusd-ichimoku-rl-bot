"""
Feature Engineering cho XAUUSD trading.
60+ technical indicators theo phong cách forbbiden403/tradingbot "God Mode Features".
Dùng thư viện `ta` (Technical Analysis) cho indicators chuẩn.
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def add_trend_indicators(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """Thêm indicators xu hướng: EMA, SMA, MACD, ADX."""
    p = prefix

    # Moving Averages
    for period in [9, 21, 50, 100, 200]:
        df[f"{p}ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()
        df[f"{p}sma_{period}"] = df["close"].rolling(window=period).mean()

    # MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df[f"{p}macd"] = ema12 - ema26
    df[f"{p}macd_signal"] = df[f"{p}macd"].ewm(span=9, adjust=False).mean()
    df[f"{p}macd_hist"] = df[f"{p}macd"] - df[f"{p}macd_signal"]

    # ADX (Average Directional Index)
    high = df["high"]
    low = df["low"]
    close = df["close"]
    period = 14

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10))
    df[f"{p}adx"] = dx.rolling(window=period).mean()
    df[f"{p}plus_di"] = plus_di
    df[f"{p}minus_di"] = minus_di

    return df


def add_momentum_indicators(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """Thêm indicators momentum: RSI, Stochastic, CCI, Williams %R."""
    p = prefix

    # RSI
    period = 14
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df[f"{p}rsi"] = 100 - (100 / (1 + rs))

    # Stochastic
    low_14 = df["low"].rolling(window=14).min()
    high_14 = df["high"].rolling(window=14).max()
    df[f"{p}stoch_k"] = 100 * (df["close"] - low_14) / (high_14 - low_14 + 1e-10)
    df[f"{p}stoch_d"] = df[f"{p}stoch_k"].rolling(window=3).mean()

    # CCI (Commodity Channel Index)
    period = 20
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma_tp = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    df[f"{p}cci"] = (tp - sma_tp) / (0.015 * mad + 1e-10)

    # Williams %R
    period = 14
    high_14 = df["high"].rolling(window=period).max()
    low_14 = df["low"].rolling(window=period).min()
    df[f"{p}williams_r"] = -100 * (high_14 - df["close"]) / (high_14 - low_14 + 1e-10)

    # ROC (Rate of Change)
    df[f"{p}roc_10"] = df["close"].pct_change(periods=10) * 100

    return df


def add_volatility_indicators(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """Thêm indicators biến động: ATR, Bollinger Bands, Keltner Channels."""
    p = prefix

    # ATR (Average True Range)
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    df[f"{p}atr"] = tr.rolling(window=14).mean()

    # Bollinger Bands
    period = 20
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    df[f"{p}bb_upper"] = sma + 2 * std
    df[f"{p}bb_lower"] = sma - 2 * std
    df[f"{p}bb_mid"] = sma
    df[f"{p}bb_width"] = (df[f"{p}bb_upper"] - df[f"{p}bb_lower"]) / (sma + 1e-10)
    df[f"{p}bb_pct"] = (close - df[f"{p}bb_lower"]) / (df[f"{p}bb_upper"] - df[f"{p}bb_lower"] + 1e-10)

    # Keltner Channels
    ema20 = close.ewm(span=20, adjust=False).mean()
    df[f"{p}kc_upper"] = ema20 + 2 * df[f"{p}atr"]
    df[f"{p}kc_lower"] = ema20 - 2 * df[f"{p}atr"]

    # Historical Volatility
    df[f"{p}hvol_20"] = close.pct_change().rolling(window=20).std() * np.sqrt(252 * 24 * 60)

    return df


def add_volume_indicators(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """Thêm indicators volume: OBV, MFI, Volume MA."""
    p = prefix

    # OBV (On-Balance Volume)
    direction = np.sign(df["close"].diff())
    df[f"{p}obv"] = (direction * df["volume"]).cumsum()

    # MFI (Money Flow Index)
    period = 14
    tp = (df["high"] + df["low"] + df["close"]) / 3
    mf = tp * df["volume"]
    positive_mf = mf.where(tp > tp.shift(), 0)
    negative_mf = mf.where(tp < tp.shift(), 0)
    mfr = positive_mf.rolling(window=period).sum() / (negative_mf.rolling(window=period).sum() + 1e-10)
    df[f"{p}mfi"] = 100 - (100 / (1 + mfr))

    # Volume MA
    df[f"{p}vol_sma_20"] = df["volume"].rolling(window=20).mean()
    df[f"{p}vol_ratio"] = df["volume"] / (df[f"{p}vol_sma_20"] + 1e-10)

    return df


def add_price_action_features(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """Thêm features price action: candle patterns, S/R levels."""
    p = prefix

    # Candle body & wick
    df[f"{p}body"] = df["close"] - df["open"]
    df[f"{p}body_pct"] = df[f"{p}body"] / (df["open"] + 1e-10) * 100
    df[f"{p}upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df[f"{p}lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
    df[f"{p}candle_range"] = df["high"] - df["low"]

    # Body/Range ratio (Doji detection)
    df[f"{p}body_ratio"] = df[f"{p}body"].abs() / (df[f"{p}candle_range"] + 1e-10)

    # Price returns
    for period in [1, 5, 15, 60]:
        df[f"{p}return_{period}"] = df["close"].pct_change(periods=period) * 100

    # Distance from recent high/low
    df[f"{p}dist_high_20"] = (df["high"].rolling(20).max() - df["close"]) / (df["close"] + 1e-10) * 100
    df[f"{p}dist_low_20"] = (df["close"] - df["low"].rolling(20).min()) / (df["close"] + 1e-10) * 100

    # Support/Resistance (Pivot Points)
    df[f"{p}pivot"] = (df["high"].shift(1) + df["low"].shift(1) + df["close"].shift(1)) / 3
    df[f"{p}r1"] = 2 * df[f"{p}pivot"] - df["low"].shift(1)
    df[f"{p}s1"] = 2 * df[f"{p}pivot"] - df["high"].shift(1)

    return df


def add_session_features(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """Thêm features theo trading session (XAUUSD specific)."""
    p = prefix

    hour = df.index.hour

    # Session indicators (UTC+0 based)
    df[f"{p}session_asian"] = ((hour >= 0) & (hour < 8)).astype(int)
    df[f"{p}session_london"] = ((hour >= 8) & (hour < 16)).astype(int)
    df[f"{p}session_newyork"] = ((hour >= 13) & (hour < 21)).astype(int)
    df[f"{p}session_overlap"] = ((hour >= 13) & (hour < 16)).astype(int)  # London-NY overlap

    # Time features (cyclical encoding)
    df[f"{p}hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df[f"{p}hour_cos"] = np.cos(2 * np.pi * hour / 24)

    dow = df.index.dayofweek
    df[f"{p}dow_sin"] = np.sin(2 * np.pi * dow / 5)
    df[f"{p}dow_cos"] = np.cos(2 * np.pi * dow / 5)

    return df


def compute_all_features(
    df: pd.DataFrame,
    prefix: str = "",
    drop_na: bool = True,
) -> pd.DataFrame:
    """
    Tính toàn bộ features cho 1 timeframe.
    
    Args:
        df: DataFrame OHLCV
        prefix: Prefix thêm vào tên cột (vd: "H1_" cho higher timeframe)
        drop_na: Có loại bỏ rows NaN không
    
    Returns:
        DataFrame với 60+ features
    """
    result = df.copy()

    result = add_trend_indicators(result, prefix)
    result = add_momentum_indicators(result, prefix)
    result = add_volatility_indicators(result, prefix)
    result = add_volume_indicators(result, prefix)
    result = add_price_action_features(result, prefix)
    result = add_session_features(result, prefix)

    if drop_na:
        result.dropna(inplace=True)

    logger.info(f"📊 Features computed: {len(result.columns)} columns, {len(result)} rows")
    return result


def normalize_features(df: pd.DataFrame, exclude_cols: list = None) -> pd.DataFrame:
    """
    Normalize features cho RL input (Z-score normalization).
    Exclude columns OHLCV gốc và session indicators.
    """
    if exclude_cols is None:
        exclude_cols = ["open", "high", "low", "close", "volume"]

    result = df.copy()
    feature_cols = [c for c in result.columns if c not in exclude_cols]

    for col in feature_cols:
        series = result[col]
        mean = series.rolling(window=200, min_periods=50).mean()
        std = series.rolling(window=200, min_periods=50).std()
        result[col] = (series - mean) / (std + 1e-10)

    result.dropna(inplace=True)
    # Clip extreme values
    result[feature_cols] = result[feature_cols].clip(-5, 5)

    return result


if __name__ == "__main__":
    from data_loader import load_csv
    from pathlib import Path

    logging.basicConfig(level=logging.INFO)
    DATA_DIR = Path(__file__).resolve().parents[2]

    df = load_csv(DATA_DIR / "XAUUSD_2024_01.csv")
    df_features = compute_all_features(df)
    print(f"\n📊 Features shape: {df_features.shape}")
    print(f"📊 Feature columns: {sorted(df_features.columns.tolist())}")
    print(f"📊 Sample:\n{df_features.head()}")
