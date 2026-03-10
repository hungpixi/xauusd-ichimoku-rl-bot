"""
Ichimoku Strategy Engine - Rule-based, giống EA IchiDCA_CCBSN_PropFirm.
Tất cả params (SL/TP/DCA/Cloud/EMA) đều tunable qua grid search.

Cải tiến so với MQL5 backtester:
1. Trade-by-trade log chi tiết (entry reason, exit reason)
2. Bar-by-bar equity curve 
3. Multi-param optimization (grid search / Bayesian)
4. Statistical significance tests
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class StrategyParams:
    """Tất cả tham số chiến lược - đều có thể tối ưu."""
    # Ichimoku periods
    tenkan_period: int = 9
    kijun_period: int = 26
    senkou_period: int = 52

    # EMA Filter
    ema_fast: int = 34
    ema_slow: int = 89
    use_ema_filter: bool = True

    # Entry signals
    use_cloud_break: bool = True      # Entry khi price phá mây
    use_tk_cross: bool = False        # Entry khi Tenkan cross Kijun
    require_cloud_color: bool = True  # Mây phải bullish/bearish đúng hướng

    # Risk Management
    stop_loss: float = 5.0            # $ (XAUUSD points)
    take_profit: float = 3.0          # $
    use_trailing_stop: bool = True    # Trailing stop giống EA
    trailing_start: float = 1.5       # Bắt đầu trail khi lời $1.5
    trailing_step: float = 0.5        # Trail step $0.5

    # DCA (Dollar Cost Averaging)
    use_dca: bool = False
    dca_distance: float = 2.0        # DCA khi lỗ thêm $2
    max_dca_orders: int = 2           # Tối đa 2 lệnh DCA

    # Position sizing
    lot_size: float = 0.01
    spread: float = 0.36              # Exness XAUUSD

    # Trade management
    cooldown_bars: int = 3            # Chờ N bars sau khi đóng
    max_trades_per_day: int = 10      # Giới hạn trades/ngày


@dataclass
class Trade:
    """Chi tiết 1 trade - nhiều info hơn MQL5."""
    ticket: int
    type: str                    # "LONG" / "SHORT"
    entry_time: str
    entry_price: float
    entry_reason: str           # "cloud_break_up", "tk_cross_up", etc.
    exit_time: str = ""
    exit_price: float = 0.0
    exit_reason: str = ""       # "TP", "SL", "trailing", "signal_reverse", "manual"
    pnl: float = 0.0
    balance_after: float = 0.0
    bars_held: int = 0
    max_favorable: float = 0.0  # Max favorable excursion (MFE)
    max_adverse: float = 0.0    # Max adverse excursion (MAE)


def compute_ichimoku_raw(df: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
    """Tính Ichimoku KHÔNG normalize (dùng giá raw cho backtest)."""
    result = df.copy()
    h, l, c = result["high"], result["low"], result["close"]

    # Ichimoku core
    result["tenkan"] = (h.rolling(params.tenkan_period).max() + l.rolling(params.tenkan_period).min()) / 2
    result["kijun"] = (h.rolling(params.kijun_period).max() + l.rolling(params.kijun_period).min()) / 2
    result["senkou_a"] = (result["tenkan"] + result["kijun"]) / 2
    result["senkou_b"] = (h.rolling(params.senkou_period).max() + l.rolling(params.senkou_period).min()) / 2
    result["cloud_top"] = result[["senkou_a", "senkou_b"]].max(axis=1)
    result["cloud_bot"] = result[["senkou_a", "senkou_b"]].min(axis=1)

    # EMA
    if params.use_ema_filter:
        result["ema_fast"] = c.ewm(span=params.ema_fast, adjust=False).mean()
        result["ema_slow"] = c.ewm(span=params.ema_slow, adjust=False).mean()

    result.dropna(inplace=True)
    return result


def run_backtest(
    df: pd.DataFrame,
    params: StrategyParams,
    initial_balance: float = 10000.0,
) -> dict:
    """
    Backtest rule-based Ichimoku strategy.
    Chạy cực nhanh (loop qua numpy arrays).
    
    Returns: {trades, equity_curve, metrics, params}
    """
    # Pre-compute indicators
    data = compute_ichimoku_raw(df, params)

    # Convert to numpy for speed
    close = data["close"].values
    high = data["high"].values
    low = data["low"].values
    tenkan = data["tenkan"].values
    kijun = data["kijun"].values
    cloud_top = data["cloud_top"].values
    cloud_bot = data["cloud_bot"].values
    senkou_a = data["senkou_a"].values
    senkou_b = data["senkou_b"].values
    timestamps = data.index

    if params.use_ema_filter:
        ema_fast = data["ema_fast"].values
        ema_slow = data["ema_slow"].values
    else:
        ema_fast = ema_slow = None

    n = len(close)
    spread = params.spread
    lot = params.lot_size

    # State
    balance = initial_balance
    position = 0  # 1=long, -1=short
    entry_price = 0.0
    entry_idx = 0
    entry_reason = ""
    trail_price = 0.0  # Trailing stop level
    bars_since_close = 99
    daily_trades = 0
    current_day = -1
    ticket_counter = 0
    mfe = 0.0  # max favorable excursion
    mae = 0.0  # max adverse excursion

    # DCA state
    dca_count = 0
    dca_entry_prices = []
    dca_lot_total = 0.0

    # Results
    trades = []
    equity_curve = np.zeros(n, dtype=np.float64)

    for i in range(1, n):
        # Reset daily counter
        day = timestamps[i].day if hasattr(timestamps[i], 'day') else -1
        if day != current_day:
            daily_trades = 0
            current_day = day

        bars_since_close += 1
        p = close[i]
        h_bar = high[i]
        l_bar = low[i]

        # === SIGNALS (giống EA) ===
        buy_signal = False
        sell_signal = False
        signal_reason = ""

        # Cloud Break
        if params.use_cloud_break:
            # Buy: close vừa phá qua cloud_top (close[i-1] <= cloud_top[i-1], close[i] > cloud_top[i])
            if close[i - 1] <= cloud_top[i - 1] and p > cloud_top[i]:
                if not params.require_cloud_color or senkou_a[i] > senkou_b[i]:
                    buy_signal = True
                    signal_reason = "cloud_break_up"

            # Sell: close vừa phá xuống cloud_bot
            if close[i - 1] >= cloud_bot[i - 1] and p < cloud_bot[i]:
                if not params.require_cloud_color or senkou_a[i] < senkou_b[i]:
                    sell_signal = True
                    signal_reason = "cloud_break_down"

        # TK Cross
        if params.use_tk_cross and not (buy_signal or sell_signal):
            if tenkan[i - 1] <= kijun[i - 1] and tenkan[i] > kijun[i]:
                buy_signal = True
                signal_reason = "tk_cross_up"
            if tenkan[i - 1] >= kijun[i - 1] and tenkan[i] < kijun[i]:
                sell_signal = True
                signal_reason = "tk_cross_down"

        # EMA Filter
        if params.use_ema_filter and ema_fast is not None:
            if buy_signal and ema_fast[i] < ema_slow[i]:
                buy_signal = False
            if sell_signal and ema_fast[i] > ema_slow[i]:
                sell_signal = False

        # === MANAGE OPEN POSITION ===
        if position != 0:
            # Track MFE/MAE
            if position == 1:
                fav = (h_bar - entry_price)
                adv = (entry_price - l_bar)
            else:
                fav = (entry_price - l_bar)
                adv = (h_bar - entry_price)
            if fav > mfe:
                mfe = fav
            if adv > mae:
                mae = adv

            # Check SL
            sl_hit = False
            tp_hit = False
            exit_price = 0.0
            exit_reason = ""

            if position == 1:
                if l_bar <= entry_price - params.stop_loss:
                    sl_hit = True
                    exit_price = entry_price - params.stop_loss
                    exit_reason = "SL"
                elif h_bar >= entry_price + params.take_profit:
                    tp_hit = True
                    exit_price = entry_price + params.take_profit
                    exit_reason = "TP"
            elif position == -1:
                if h_bar >= entry_price + params.stop_loss:
                    sl_hit = True
                    exit_price = entry_price + params.stop_loss
                    exit_reason = "SL"
                elif l_bar <= entry_price - params.take_profit:
                    tp_hit = True
                    exit_price = entry_price - params.take_profit
                    exit_reason = "TP"

            # Trailing stop
            if params.use_trailing_stop and not sl_hit and not tp_hit:
                if position == 1:
                    current_profit = p - entry_price
                    if current_profit >= params.trailing_start:
                        new_trail = p - params.trailing_step
                        if new_trail > trail_price:
                            trail_price = new_trail
                        if l_bar <= trail_price and trail_price > 0:
                            sl_hit = True
                            exit_price = trail_price
                            exit_reason = "trailing"
                elif position == -1:
                    current_profit = entry_price - p
                    if current_profit >= params.trailing_start:
                        new_trail = p + params.trailing_step
                        if trail_price == 0 or new_trail < trail_price:
                            trail_price = new_trail
                        if h_bar >= trail_price and trail_price > 0:
                            sl_hit = True
                            exit_price = trail_price
                            exit_reason = "trailing"

            # Signal reverse close
            if not sl_hit and not tp_hit:
                if position == 1 and sell_signal:
                    sl_hit = True
                    exit_price = p
                    exit_reason = "signal_reverse"
                elif position == -1 and buy_signal:
                    sl_hit = True
                    exit_price = p
                    exit_reason = "signal_reverse"

            # Close position
            if sl_hit or tp_hit:
                # Calculate PnL (spread đã tính trong entry)
                if position == 1:
                    actual_exit = exit_price - spread * 0.5
                else:
                    actual_exit = exit_price + spread * 0.5

                total_lot = lot * (1 + dca_count)
                pnl = (actual_exit - entry_price) * position * total_lot * 100

                balance += pnl

                trade = Trade(
                    ticket=ticket_counter,
                    type="LONG" if position == 1 else "SHORT",
                    entry_time=str(timestamps[entry_idx]),
                    entry_price=entry_price,
                    entry_reason=entry_reason,
                    exit_time=str(timestamps[i]),
                    exit_price=exit_price,
                    exit_reason=exit_reason,
                    pnl=pnl,
                    balance_after=balance,
                    bars_held=i - entry_idx,
                    max_favorable=mfe,
                    max_adverse=mae,
                )
                trades.append(trade)
                ticket_counter += 1

                position = 0
                entry_price = 0.0
                trail_price = 0.0
                bars_since_close = 0
                dca_count = 0
                dca_entry_prices = []

        # === OPEN NEW POSITION ===
        if position == 0 and bars_since_close >= params.cooldown_bars and daily_trades < params.max_trades_per_day:
            if buy_signal:
                position = 1
                entry_price = p + spread * 0.5  # Buy at ask
                entry_idx = i
                entry_reason = signal_reason
                trail_price = 0.0
                mfe = 0.0
                mae = 0.0
                daily_trades += 1
                dca_count = 0

            elif sell_signal:
                position = -1
                entry_price = p - spread * 0.5  # Sell at bid
                entry_idx = i
                entry_reason = signal_reason
                trail_price = 0.0
                mfe = 0.0
                mae = 0.0
                daily_trades += 1
                dca_count = 0

        # === Update equity ===
        unrealized = 0.0
        if position != 0:
            if position == 1:
                unrealized = (p - spread * 0.5 - entry_price) * lot * (1 + dca_count) * 100
            else:
                unrealized = (entry_price - p - spread * 0.5) * lot * (1 + dca_count) * 100
        equity_curve[i] = balance + unrealized

    # Close any remaining position
    if position != 0:
        p = close[-1]
        if position == 1:
            actual_exit = p - spread * 0.5
        else:
            actual_exit = p + spread * 0.5
        total_lot = lot * (1 + dca_count)
        pnl = (actual_exit - entry_price) * position * total_lot * 100
        balance += pnl
        trades.append(Trade(
            ticket=ticket_counter, type="LONG" if position == 1 else "SHORT",
            entry_time=str(timestamps[entry_idx]), entry_price=entry_price,
            entry_reason=entry_reason, exit_time=str(timestamps[-1]),
            exit_price=p, exit_reason="end_of_data", pnl=pnl,
            balance_after=balance, bars_held=n - 1 - entry_idx,
            max_favorable=mfe, max_adverse=mae,
        ))
        equity_curve[-1] = balance

    # Fix equity curve (fill forward)
    eq = equity_curve.copy()
    eq[0] = initial_balance
    for i in range(1, len(eq)):
        if eq[i] == 0:
            eq[i] = eq[i - 1]

    return {
        "trades": trades,
        "equity_curve": eq,
        "final_balance": balance,
        "initial_balance": initial_balance,
        "total_bars": n,
        "params": asdict(params),
    }


def optimize_params(
    df: pd.DataFrame,
    param_grid: dict,
    initial_balance: float = 10000.0,
    sort_by: str = "profit_factor",
    top_n: int = 10,
) -> list:
    """
    Grid search optimize tham số SL/TP/trailing.
    Tốt hơn MQL5 optimizer vì chạy cực nhanh + sort theo bất kỳ metric.
    """
    from itertools import product

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(product(*values))

    logger.info(f"🔍 Optimizing: {len(combinations)} combinations")

    results = []
    for combo in combinations:
        params = StrategyParams(**dict(zip(keys, combo)))
        result = run_backtest(df, params, initial_balance)
        trades = result["trades"]

        if len(trades) == 0:
            continue

        pnls = np.array([t.pnl for t in trades])
        gp = pnls[pnls > 0].sum()
        gl = abs(pnls[pnls < 0].sum())
        pf = gp / (gl + 1e-10)
        wr = (pnls > 0).sum() / len(pnls) * 100
        net = pnls.sum()
        eq = result["equity_curve"]
        peak = np.maximum.accumulate(eq)
        dd = ((peak - eq) / (peak + 1e-10) * 100).max()

        # Sharpe
        if len(eq) > 1:
            rets = np.diff(eq) / (np.abs(eq[:-1]) + 1e-10)
            sharpe = rets.mean() / (rets.std() + 1e-10) * np.sqrt(252 * 24 * 12)
        else:
            sharpe = 0

        # Recovery factor
        rf = net / (dd * initial_balance / 100 + 1e-10)

        results.append({
            "params": dict(zip(keys, combo)),
            "net_profit": net,
            "profit_factor": pf,
            "win_rate": wr,
            "total_trades": len(trades),
            "max_dd": dd,
            "sharpe": sharpe,
            "recovery_factor": rf,
            "avg_win": pnls[pnls > 0].mean() if (pnls > 0).any() else 0,
            "avg_loss": pnls[pnls < 0].mean() if (pnls < 0).any() else 0,
        })

    results.sort(key=lambda x: x.get(sort_by, 0), reverse=True)
    return results[:top_n]


def print_optimization_results(results: list, title: str = "Optimization Results"):
    """In bảng kết quả optimization đẹp."""
    print(f"\n{'='*120}")
    print(f"  {title}")
    print(f"{'='*120}")
    print(f"  {'#':>3} | {'Net Profit':>12} | {'PF':>6} | {'WR':>6} | {'Trades':>6} | {'MaxDD':>7} | {'Sharpe':>7} | {'RF':>6} | Params")
    print(f"  {'─'*3}─┼─{'─'*12}─┼─{'─'*6}─┼─{'─'*6}─┼─{'─'*6}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*6}─┼─{'─'*40}")

    for i, r in enumerate(results):
        params_str = ", ".join(f"{k}={v}" for k, v in r["params"].items())
        print(
            f"  {i+1:>3} | ${r['net_profit']:>10.2f} | {r['profit_factor']:>5.2f} | {r['win_rate']:>5.1f}% | "
            f"{r['total_trades']:>6} | {r['max_dd']:>6.2f}% | {r['sharpe']:>6.2f} | {r['recovery_factor']:>5.2f} | {params_str}"
        )
    print(f"{'='*120}\n")
