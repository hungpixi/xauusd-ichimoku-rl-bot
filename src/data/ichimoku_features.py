"""
Ichimoku Feature Engine cho XAUUSD.
Tái tạo logic EA IchiDCA_CCBSN_PropFirm.mq5 dưới dạng features cho RL.
Ichimoku Cloud Break + EMA Filter + DCA awareness.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def ichimoku(df: pd.DataFrame, tenkan: int = 9, kijun: int = 26, senkou: int = 52) -> pd.DataFrame:
    """
    Tính Ichimoku Kinko Hyo giống MT5.
    - Tenkan-sen: (highest high + lowest low) / 2 trong N periods
    - Kijun-sen: tương tự với M periods
    - Senkou Span A: (Tenkan + Kijun) / 2 shift ahead 26
    - Senkou Span B: (highest + lowest) / 2 trong K periods, shift ahead 26
    - Chikou Span: close shift back 26
    """
    result = df.copy()
    high = result["high"]
    low = result["low"]
    close = result["close"]

    # Tenkan-sen
    result["tenkan"] = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2

    # Kijun-sen
    result["kijun"] = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2

    # Senkou Span A (shift ahead - nhưng cho RL, dùng giá trị hiện tại, không shift)
    result["senkou_a"] = ((result["tenkan"] + result["kijun"]) / 2)

    # Senkou Span B
    result["senkou_b"] = (high.rolling(senkou).max() + low.rolling(senkou).min()) / 2

    # Chikou Span (close shift back 26 → cho RL, dùng close từ 26 bars trước)
    result["chikou"] = close.shift(kijun)

    # Cloud boundaries (mây trên/dưới)
    result["cloud_top"] = result[["senkou_a", "senkou_b"]].max(axis=1)
    result["cloud_bot"] = result[["senkou_a", "senkou_b"]].min(axis=1)
    result["cloud_width"] = result["cloud_top"] - result["cloud_bot"]

    return result


def add_ichimoku_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tái tạo logic signal từ EA IchiDCA_CCBSN:
    - Cloud Break: giá đóng trên/dưới mây
    - Tenkan/Kijun cross
    - Khoảng cách giá-mây
    """
    result = df.copy()
    close = result["close"]

    # === Vị trí giá so với mây ===
    result["above_cloud"] = (close > result["cloud_top"]).astype(int)
    result["below_cloud"] = (close < result["cloud_bot"]).astype(int)
    result["in_cloud"] = ((close >= result["cloud_bot"]) & (close <= result["cloud_top"])).astype(int)

    # Cloud Break signals (giống GetIchimokuSignal trong EA)
    prev_close = close.shift(1)
    prev_cloud_top = result["cloud_top"].shift(1)
    prev_cloud_bot = result["cloud_bot"].shift(1)

    # Buy break: close trước <= cloud_top trước, close hiện tại > cloud_top hiện tại
    result["cloud_break_up"] = (
        (prev_close <= prev_cloud_top) & (close > result["cloud_top"])
    ).astype(int)

    # Sell break: close trước >= cloud_bot trước, close hiện tại < cloud_bot hiện tại
    result["cloud_break_down"] = (
        (prev_close >= prev_cloud_bot) & (close < result["cloud_bot"])
    ).astype(int)

    # === Tenkan/Kijun relationship ===
    result["tk_above_kj"] = (result["tenkan"] > result["kijun"]).astype(int)
    result["tk_below_kj"] = (result["tenkan"] < result["kijun"]).astype(int)

    # TK Cross
    prev_tenkan = result["tenkan"].shift(1)
    prev_kijun = result["kijun"].shift(1)
    result["tk_cross_up"] = (
        (prev_tenkan <= prev_kijun) & (result["tenkan"] > result["kijun"])
    ).astype(int)
    result["tk_cross_down"] = (
        (prev_tenkan >= prev_kijun) & (result["tenkan"] < result["kijun"])
    ).astype(int)

    # === Khoảng cách giá-mây (normalized) ===
    result["dist_to_cloud_top"] = (close - result["cloud_top"]) / (result["cloud_width"] + 1e-10)
    result["dist_to_cloud_bot"] = (close - result["cloud_bot"]) / (result["cloud_width"] + 1e-10)

    # === Cloud color (bullish/bearish) ===
    result["cloud_bullish"] = (result["senkou_a"] > result["senkou_b"]).astype(int)

    # === Chikou vs price ===
    if "chikou" in result.columns:
        result["chikou_above_price"] = (result["chikou"] > close).astype(int)

    return result


def add_ema_filter(df: pd.DataFrame, fast: int = 34, slow: int = 89) -> pd.DataFrame:
    """EMA Filter giống EA: EMA 34/89 cho trend confirmation."""
    result = df.copy()
    close = result["close"]

    result["ema_fast"] = close.ewm(span=fast, adjust=False).mean()
    result["ema_slow"] = close.ewm(span=slow, adjust=False).mean()
    result["ema_bullish"] = (result["ema_fast"] > result["ema_slow"]).astype(int)
    result["ema_diff"] = (result["ema_fast"] - result["ema_slow"]) / (close + 1e-10) * 100

    return result


def add_dca_awareness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features hỗ trợ DCA logic:
    - Volatility ngắn hạn (có nên DCA?)
    - Price deviation từ moving average (giá xa trung bình?)
    - Momentum (giá đang recovery hay tiếp tục xuống?)
    """
    result = df.copy()
    close = result["close"]

    # ATR (volatility) - cần cho DCA distance
    high = result["high"]
    low = result["low"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    result["atr_14"] = tr.rolling(14).mean()
    result["atr_50"] = tr.rolling(50).mean()
    result["vol_ratio"] = result["atr_14"] / (result["atr_50"] + 1e-10)

    # Price deviation từ Kijun-sen (cần cho DCA decision)
    if "kijun" in result.columns:
        result["price_dev_kijun"] = (close - result["kijun"]) / (result["atr_14"] + 1e-10)

    # RSI (momentum - giá oversold? ripe for DCA)
    period = 14
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    result["rsi"] = 100 - (100 / (1 + rs))

    # Stochastic RSI
    rsi = result["rsi"]
    rsi_min = rsi.rolling(14).min()
    rsi_max = rsi.rolling(14).max()
    result["stoch_rsi"] = (rsi - rsi_min) / (rsi_max - rsi_min + 1e-10)

    # MACD (secondary trend filter)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    result["macd"] = ema12 - ema26
    result["macd_signal"] = result["macd"].ewm(span=9, adjust=False).mean()
    result["macd_hist"] = result["macd"] - result["macd_signal"]

    # Bollinger Band position (hỗ trợ entry timing)
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    result["bb_pct"] = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)

    # Price returns (momentum)
    for p in [1, 5, 15, 60]:
        result[f"ret_{p}"] = close.pct_change(p) * 100

    return result


def add_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """Trading session detection (XAUUSD specific)."""
    result = df.copy()
    hour = result.index.hour

    result["session_asian"] = ((hour >= 0) & (hour < 8)).astype(int)
    result["session_london"] = ((hour >= 8) & (hour < 16)).astype(int)
    result["session_newyork"] = ((hour >= 13) & (hour < 21)).astype(int)
    result["session_overlap"] = ((hour >= 13) & (hour < 16)).astype(int)

    # Cyclical encoding
    result["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    result["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    dow = result.index.dayofweek
    result["dow_sin"] = np.sin(2 * np.pi * dow / 5)
    result["dow_cos"] = np.cos(2 * np.pi * dow / 5)

    return result


def compute_ichimoku_features(
    df: pd.DataFrame,
    tenkan: int = 9,
    kijun: int = 26,
    senkou: int = 52,
    ema_fast: int = 34,
    ema_slow: int = 89,
    drop_na: bool = True,
) -> pd.DataFrame:
    """
    Pipeline hoàn chỉnh: Ichimoku + EMA + DCA awareness + Sessions.
    Giống hệt cách EA IchiDCA_CCBSN xử lý signals.
    """
    result = df.copy()

    # 1. Ichimoku core
    result = ichimoku(result, tenkan, kijun, senkou)

    # 2. Ichimoku signals
    result = add_ichimoku_signals(result)

    # 3. EMA filter (giống EA: 34/89)
    result = add_ema_filter(result, ema_fast, ema_slow)

    # 4. DCA awareness (ATR, RSI, MACD, BB)
    result = add_dca_awareness(result)

    # 5. Sessions
    result = add_session_features(result)

    if drop_na:
        result.dropna(inplace=True)

    logger.info(f"📊 Ichimoku features: {len(result.columns)} cols, {len(result)} rows")
    return result


def normalize_features(df: pd.DataFrame, exclude_cols: list = None) -> pd.DataFrame:
    """Normalize features cho RL (rolling Z-score)."""
    if exclude_cols is None:
        exclude_cols = ["open", "high", "low", "close", "volume"]

    result = df.copy()

    # Columns cần normalize (continuous values)
    continuous_cols = [c for c in result.columns if c not in exclude_cols
                       and not any(c.startswith(p) for p in [
                           "above_", "below_", "in_", "cloud_break", "tk_above",
                           "tk_below", "tk_cross", "cloud_bullish", "chikou_above",
                           "ema_bullish", "session_"
                       ])]

    for col in continuous_cols:
        s = result[col]
        mean = s.rolling(200, min_periods=50).mean()
        std = s.rolling(200, min_periods=50).std()
        result[col] = (s - mean) / (std + 1e-10)

    result.dropna(inplace=True)
    result[continuous_cols] = result[continuous_cols].clip(-5, 5)

    return result


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
    from data.data_loader import load_csv
    from data.resampler import resample_ohlcv
    from pathlib import Path

    logging.basicConfig(level=logging.INFO)
    DATA_DIR = Path(__file__).resolve().parents[2]

    df = load_csv(DATA_DIR / "XAUUSD_2026_01.csv")
    df_m5 = resample_ohlcv(df, "M5")
    df_feat = compute_ichimoku_features(df_m5)
    df_norm = normalize_features(df_feat)

    print(f"\n📊 Shape: {df_norm.shape}")
    print(f"📊 Features: {sorted(df_norm.columns.tolist())}")
    print(f"📊 Sample:\n{df_norm.head()}")
