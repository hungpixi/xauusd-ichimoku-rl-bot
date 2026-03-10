"""
Multi-Timeframe Feature Engine cho XAUUSD RL Bot v2.
Tính Ichimoku + indicators trên M5, M15, H1, H4 đồng thời.
Merge tất cả vào M5 base → 100-120 features cho RL.

Điểm khác MoonDev: Ichimoku là core (không phải generic indicators).
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


# =============================================
# INDICATOR FUNCTIONS (dùng numpy cho speed)
# =============================================

def ichimoku(c: np.ndarray, h: np.ndarray, l: np.ndarray,
             tenkan: int = 9, kijun: int = 26, senkou: int = 52) -> Dict[str, np.ndarray]:
    """Ichimoku core - vectorized numpy."""
    n = len(c)

    def donchian_mid(data, period):
        out = np.full(n, np.nan)
        for i in range(period - 1, n):
            out[i] = (np.max(data[i - period + 1:i + 1]) + np.min(data[i - period + 1:i + 1])) / 2
        return out

    tk = donchian_mid(h, tenkan) + donchian_mid(l, tenkan)
    tk = donchian_mid(h, tenkan)  # Fix: use proper calc
    # Recalculate properly
    tenkan_h = pd.Series(h).rolling(tenkan).max().values
    tenkan_l = pd.Series(l).rolling(tenkan).min().values
    tk = (tenkan_h + tenkan_l) / 2

    kijun_h = pd.Series(h).rolling(kijun).max().values
    kijun_l = pd.Series(l).rolling(kijun).min().values
    kj = (kijun_h + kijun_l) / 2

    sa = (tk + kj) / 2
    senkou_h = pd.Series(h).rolling(senkou).max().values
    senkou_l = pd.Series(l).rolling(senkou).min().values
    sb = (senkou_h + senkou_l) / 2

    cloud_top = np.maximum(sa, sb)
    cloud_bot = np.minimum(sa, sb)

    return {
        "tenkan": tk, "kijun": kj,
        "senkou_a": sa, "senkou_b": sb,
        "cloud_top": cloud_top, "cloud_bot": cloud_bot,
        "cloud_width": cloud_top - cloud_bot,
    }


def ema(data: np.ndarray, period: int) -> np.ndarray:
    return pd.Series(data).ewm(span=period, adjust=False).mean().values


def rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).rolling(period).mean().values
    avg_loss = pd.Series(loss).rolling(period).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int = 14) -> np.ndarray:
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    return pd.Series(tr).rolling(period).mean().values


def macd(close: np.ndarray) -> Dict[str, np.ndarray]:
    ema12 = ema(close, 12)
    ema26 = ema(close, 26)
    m = ema12 - ema26
    s = ema(m, 9)
    return {"macd": m, "macd_signal": s, "macd_hist": m - s}


def bollinger(close: np.ndarray, period: int = 20) -> Dict[str, np.ndarray]:
    sma = pd.Series(close).rolling(period).mean().values
    std = pd.Series(close).rolling(period).std().values
    upper = sma + 2 * std
    lower = sma - 2 * std
    pct = (close - lower) / (upper - lower + 1e-10)
    return {"bb_upper": upper, "bb_lower": lower, "bb_pct": pct}


def stochastic(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int = 14) -> np.ndarray:
    hh = pd.Series(h).rolling(period).max().values
    ll = pd.Series(l).rolling(period).min().values
    return (c - ll) / (hh - ll + 1e-10) * 100


def adx(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int = 14) -> np.ndarray:
    """Average Directional Index."""
    up = np.diff(h, prepend=h[0])
    down = -np.diff(l, prepend=l[0])
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = atr(h, l, c, 1)  # True range
    atr_val = pd.Series(tr).rolling(period).mean().values
    plus_di = pd.Series(plus_dm).rolling(period).mean().values / (atr_val + 1e-10) * 100
    minus_di = pd.Series(minus_dm).rolling(period).mean().values / (atr_val + 1e-10) * 100
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100
    return pd.Series(dx).rolling(period).mean().values


# =============================================
# SINGLE TIMEFRAME FEATURES
# =============================================

def compute_tf_features(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """
    Tính tất cả features cho 1 timeframe.
    Returns DataFrame với cột prefix_feature_name.
    """
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values

    features = {}
    p = f"{prefix}_" if prefix else ""

    # 1. Ichimoku (core - điểm khác biệt)
    ichi = ichimoku(c, h, l)
    for k, v in ichi.items():
        features[f"{p}ichi_{k}"] = v

    # Ichimoku signals
    features[f"{p}above_cloud"] = (c > ichi["cloud_top"]).astype(float)
    features[f"{p}below_cloud"] = (c < ichi["cloud_bot"]).astype(float)
    features[f"{p}tk_above_kj"] = (ichi["tenkan"] > ichi["kijun"]).astype(float)
    features[f"{p}cloud_bullish"] = (ichi["senkou_a"] > ichi["senkou_b"]).astype(float)
    features[f"{p}dist_cloud"] = (c - ichi["cloud_top"]) / (ichi["cloud_width"] + 1e-10)

    # Cloud break signals
    prev_c = np.roll(c, 1)
    prev_ct = np.roll(ichi["cloud_top"], 1)
    prev_cb = np.roll(ichi["cloud_bot"], 1)
    features[f"{p}cloud_break_up"] = ((prev_c <= prev_ct) & (c > ichi["cloud_top"])).astype(float)
    features[f"{p}cloud_break_down"] = ((prev_c >= prev_cb) & (c < ichi["cloud_bot"])).astype(float)

    # 2. EMA (34/89 giống EA + thêm 21/200)
    ema_21 = ema(c, 21)
    ema_34 = ema(c, 34)
    ema_89 = ema(c, 89)
    ema_200 = ema(c, 200)
    features[f"{p}ema_34_89_bull"] = (ema_34 > ema_89).astype(float)
    features[f"{p}ema_21_200_diff"] = (ema_21 - ema_200) / (c + 1e-10) * 100
    features[f"{p}price_vs_ema200"] = (c - ema_200) / (c + 1e-10) * 100

    # 3. RSI
    rsi_val = rsi(c, 14)
    features[f"{p}rsi"] = rsi_val
    features[f"{p}rsi_oversold"] = (rsi_val < 30).astype(float)
    features[f"{p}rsi_overbought"] = (rsi_val > 70).astype(float)

    # 4. MACD
    m = macd(c)
    features[f"{p}macd_hist"] = m["macd_hist"]
    features[f"{p}macd_bull"] = (m["macd"] > m["macd_signal"]).astype(float)

    # 5. ATR (volatility)
    atr_val = atr(h, l, c, 14)
    atr_50 = atr(h, l, c, 50)
    features[f"{p}atr_14"] = atr_val
    features[f"{p}vol_ratio"] = atr_val / (atr_50 + 1e-10)

    # 6. Bollinger
    bb = bollinger(c)
    features[f"{p}bb_pct"] = bb["bb_pct"]

    # 7. Stochastic
    features[f"{p}stoch"] = stochastic(h, l, c)

    # 8. ADX (trend strength)
    features[f"{p}adx"] = adx(h, l, c)

    # 9. Price returns
    for period in [1, 5, 15]:
        ret = np.zeros_like(c)
        ret[period:] = (c[period:] - c[:-period]) / (c[:-period] + 1e-10) * 100
        features[f"{p}ret_{period}"] = ret

    result = pd.DataFrame(features, index=df.index)
    return result


# =============================================
# MULTI-TIMEFRAME ENGINE
# =============================================

def build_multi_tf_features(
    df_m1: pd.DataFrame,
    timeframes: List[str] = None,
) -> pd.DataFrame:
    """
    Build features đa timeframe từ M1 data.
    
    Process:
    1. Resample M1 → M5, M15, H1, H4
    2. Tính features trên MỖI TF
    3. Merge tất cả vào M5 base (forward-fill)
    4. Thêm session features
    
    Returns: M5 DataFrame với 100-120 features
    """
    from src.data.resampler import resample_ohlcv

    if timeframes is None:
        timeframes = ["M5", "M15", "H1", "H4"]

    # Resample
    tf_data = {}
    for tf in timeframes:
        tf_data[tf] = resample_ohlcv(df_m1, tf)
        logger.info(f"  📊 {tf}: {len(tf_data[tf])} bars")

    # Base = M5
    base_tf = timeframes[0]
    base_df = tf_data[base_tf].copy()

    # Compute features per timeframe
    all_features = []
    for tf in timeframes:
        prefix = tf.lower()
        feats = compute_tf_features(tf_data[tf], prefix=prefix)
        all_features.append((tf, feats))
        logger.info(f"  🔧 {tf}: {len(feats.columns)} features")

    # Merge: M5 features join as-is, higher TF forward-fill vào M5 index
    result = base_df[["open", "high", "low", "close", "volume"]].copy()

    for tf, feats in all_features:
        if tf == base_tf:
            result = result.join(feats, how="left")
        else:
            # Reindex higher TF features to M5, forward fill
            feats_reindexed = feats.reindex(result.index, method="ffill")
            result = result.join(feats_reindexed, how="left")

    # Session features
    hour = result.index.hour
    result["session_asian"] = ((hour >= 0) & (hour < 8)).astype(float)
    result["session_london"] = ((hour >= 8) & (hour < 16)).astype(float)
    result["session_ny"] = ((hour >= 13) & (hour < 21)).astype(float)
    result["session_overlap"] = ((hour >= 13) & (hour < 16)).astype(float)
    result["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    result["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    dow = result.index.dayofweek
    result["dow_sin"] = np.sin(2 * np.pi * dow / 5)
    result["dow_cos"] = np.cos(2 * np.pi * dow / 5)

    # Drop NaN rows
    result.dropna(inplace=True)

    feature_cols = [c for c in result.columns if c not in ["open", "high", "low", "close", "volume"]]
    logger.info(f"✅ Multi-TF features: {len(feature_cols)} features, {len(result)} bars")

    return result


def normalize_features(df: pd.DataFrame, window: int = 200) -> pd.DataFrame:
    """Rolling Z-score normalization (chỉ continuous features)."""
    result = df.copy()
    ohlcv = ["open", "high", "low", "close", "volume"]
    binary_prefixes = ["above_", "below_", "cloud_break", "tk_above", "cloud_bullish",
                       "ema_34_89_bull", "macd_bull", "rsi_over", "session_"]

    continuous_cols = [c for c in result.columns
                       if c not in ohlcv
                       and not any(c.endswith(p) or any(c.startswith(bp) or bp in c for bp in binary_prefixes)
                                   for p in [])]

    # Filter to only normalize non-binary columns
    to_normalize = []
    for col in result.columns:
        if col in ohlcv:
            continue
        is_binary = any(bp in col for bp in binary_prefixes)
        if not is_binary:
            to_normalize.append(col)

    for col in to_normalize:
        s = result[col]
        mean = s.rolling(window, min_periods=50).mean()
        std = s.rolling(window, min_periods=50).std()
        result[col] = (s - mean) / (std + 1e-10)

    result.dropna(inplace=True)
    result[to_normalize] = result[to_normalize].clip(-5, 5)

    return result


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from data.data_loader import load_csv
    DATA_DIR = Path(__file__).resolve().parents[2]

    df_m1 = load_csv(DATA_DIR / "XAUUSD_2026_01.csv")
    df = build_multi_tf_features(df_m1)
    df = normalize_features(df)

    feature_cols = [c for c in df.columns if c not in ["open", "high", "low", "close", "volume"]]
    print(f"\n✅ Total: {len(feature_cols)} features, {len(df)} bars")
    print(f"📋 Features:\n  {', '.join(sorted(feature_cols))}")
