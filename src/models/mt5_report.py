"""
MT5 Strategy Tester Report - Output giống hệt MQL5 Tester.
Tính tất cả metrics: Profit Factor, Recovery Factor, DD, consecutive wins/losses, etc.
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def compute_mt5_report(
    trade_log: pd.DataFrame,
    equity_history: list,
    initial_balance: float = 10000.0,
    total_bars: int = 0,
) -> dict:
    """
    Tính toàn bộ metrics giống MT5 Strategy Tester.
    
    Args:
        trade_log: DataFrame với columns [step, type, entry, exit, pnl, balance]
        equity_history: List equity qua từng step
        initial_balance: Vốn ban đầu
        total_bars: Tổng số bars
    """
    report = {}

    # === Basic info ===
    report["bars"] = total_bars
    report["initial_deposit"] = initial_balance

    if trade_log.empty:
        report["total_net_profit"] = 0
        return report

    # === Profits ===
    all_pnl = trade_log["pnl"].values
    report["total_net_profit"] = float(np.sum(all_pnl))
    report["gross_profit"] = float(np.sum(all_pnl[all_pnl > 0])) if len(all_pnl[all_pnl > 0]) > 0 else 0
    report["gross_loss"] = float(np.sum(all_pnl[all_pnl < 0])) if len(all_pnl[all_pnl < 0]) > 0 else 0

    # === Drawdown ===
    equity = np.array(equity_history, dtype=float)
    if len(equity) > 0:
        # Balance Drawdown (từ balance history trong trade_log)
        balances = trade_log["balance"].values
        if len(balances) > 0:
            peak_bal = np.maximum.accumulate(np.concatenate([[initial_balance], balances]))
            dd_bal = peak_bal[1:] - balances
            dd_bal_pct = dd_bal / (peak_bal[1:] + 1e-10) * 100

            report["balance_dd_absolute"] = float(max(initial_balance - min(balances), 0))
            report["balance_dd_maximal"] = float(np.max(dd_bal)) if len(dd_bal) > 0 else 0
            report["balance_dd_maximal_pct"] = float(np.max(dd_bal_pct)) if len(dd_bal_pct) > 0 else 0
            report["balance_dd_relative"] = f"{report['balance_dd_maximal_pct']:.2f}% ({report['balance_dd_maximal']:.2f})"
        else:
            report["balance_dd_absolute"] = 0
            report["balance_dd_maximal"] = 0
            report["balance_dd_maximal_pct"] = 0

        # Equity Drawdown
        peak_eq = np.maximum.accumulate(equity)
        dd_eq = peak_eq - equity
        dd_eq_pct = dd_eq / (peak_eq + 1e-10) * 100

        report["equity_dd_absolute"] = float(max(initial_balance - np.min(equity), 0))
        report["equity_dd_maximal"] = float(np.max(dd_eq))
        report["equity_dd_maximal_pct"] = float(np.max(dd_eq_pct))
        report["equity_dd_relative"] = f"{report['equity_dd_maximal_pct']:.2f}% ({report['equity_dd_maximal']:.2f})"

    # === Ratios ===
    report["profit_factor"] = abs(report["gross_profit"] / (report["gross_loss"] + 1e-10))
    report["expected_payoff"] = float(np.mean(all_pnl))

    # Recovery Factor = Net Profit / Max DD
    max_dd = report.get("equity_dd_maximal", 0)
    report["recovery_factor"] = report["total_net_profit"] / (max_dd + 1e-10)

    # Sharpe Ratio
    if len(equity) > 1:
        returns = np.diff(equity) / (equity[:-1] + 1e-10)
        report["sharpe_ratio"] = float(np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252 * 24 * 12))  # M5
    else:
        report["sharpe_ratio"] = 0

    # AHPR (Average Holding Period Return)
    if len(all_pnl) > 0:
        trade_returns = all_pnl / initial_balance
        report["ahpr"] = float(1 + np.mean(trade_returns))
        # GHPR (Geometric)
        pos_returns = 1 + trade_returns
        pos_returns = pos_returns[pos_returns > 0]  # Avoid negative
        if len(pos_returns) > 0:
            report["ghpr"] = float(np.exp(np.mean(np.log(pos_returns))))
        else:
            report["ghpr"] = 0
    else:
        report["ahpr"] = 1
        report["ghpr"] = 1

    # Z-Score
    if len(all_pnl) > 1:
        wins = (all_pnl > 0).astype(int)
        n = len(wins)
        n_wins = np.sum(wins)
        n_losses = n - n_wins
        runs = 1 + np.sum(np.diff(wins) != 0)
        expected_runs = (2 * n_wins * n_losses) / (n + 1e-10) + 1
        std_runs = np.sqrt((2 * n_wins * n_losses * (2 * n_wins * n_losses - n)) / (n * n * (n - 1) + 1e-10))
        report["z_score"] = (runs - expected_runs) / (std_runs + 1e-10)
    else:
        report["z_score"] = 0

    # LR Correlation & Standard Error
    if len(equity) > 2:
        x = np.arange(len(equity))
        corr_matrix = np.corrcoef(x, equity)
        report["lr_correlation"] = float(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0
        # Standard error
        slope, intercept = np.polyfit(x, equity, 1)
        predicted = slope * x + intercept
        residuals = equity - predicted
        report["lr_std_error"] = float(np.std(residuals))
    else:
        report["lr_correlation"] = 0
        report["lr_std_error"] = 0

    # === Trade Statistics ===
    report["total_trades"] = len(trade_log)
    report["total_deals"] = len(trade_log) * 2  # Open + Close

    # Long/Short breakdown
    long_trades = trade_log[trade_log["type"] == "LONG"]
    short_trades = trade_log[trade_log["type"] == "SHORT"]

    long_won = long_trades[long_trades["pnl"] > 0]
    short_won = short_trades[short_trades["pnl"] > 0]

    report["short_trades_total"] = len(short_trades)
    report["short_trades_won"] = len(short_won)
    report["short_trades_won_pct"] = len(short_won) / (len(short_trades) + 1e-10) * 100

    report["long_trades_total"] = len(long_trades)
    report["long_trades_won"] = len(long_won)
    report["long_trades_won_pct"] = len(long_won) / (len(long_trades) + 1e-10) * 100

    # Profit/Loss trades
    profit_trades = trade_log[trade_log["pnl"] > 0]
    loss_trades = trade_log[trade_log["pnl"] < 0]

    report["profit_trades"] = len(profit_trades)
    report["profit_trades_pct"] = len(profit_trades) / (len(trade_log) + 1e-10) * 100
    report["loss_trades"] = len(loss_trades)
    report["loss_trades_pct"] = len(loss_trades) / (len(trade_log) + 1e-10) * 100

    # Largest
    report["largest_profit_trade"] = float(all_pnl.max()) if len(all_pnl) > 0 else 0
    report["largest_loss_trade"] = float(all_pnl.min()) if len(all_pnl) > 0 else 0

    # Average
    report["average_profit_trade"] = float(profit_trades["pnl"].mean()) if len(profit_trades) > 0 else 0
    report["average_loss_trade"] = float(loss_trades["pnl"].mean()) if len(loss_trades) > 0 else 0

    # === Consecutive Stats ===
    if len(all_pnl) > 0:
        wins_mask = all_pnl > 0

        # Consecutive wins
        max_consec_wins, max_consec_wins_profit = _max_consecutive(all_pnl, True)
        max_consec_losses, max_consec_losses_loss = _max_consecutive(all_pnl, False)

        # Max consecutive profit ($) and its count
        max_consec_profit, max_consec_profit_count = _max_consecutive_profit(all_pnl, True)
        max_consec_loss, max_consec_loss_count = _max_consecutive_profit(all_pnl, False)

        report["max_consecutive_wins"] = max_consec_wins
        report["max_consecutive_wins_profit"] = max_consec_wins_profit
        report["max_consecutive_losses"] = max_consec_losses
        report["max_consecutive_losses_loss"] = max_consec_losses_loss

        report["maximal_consecutive_profit"] = max_consec_profit
        report["maximal_consecutive_profit_count"] = max_consec_profit_count
        report["maximal_consecutive_loss"] = max_consec_loss
        report["maximal_consecutive_loss_count"] = max_consec_loss_count

        # Average consecutive
        report["avg_consecutive_wins"] = _avg_consecutive(all_pnl, True)
        report["avg_consecutive_losses"] = _avg_consecutive(all_pnl, False)

    return report


def _max_consecutive(pnl: np.ndarray, is_win: bool) -> tuple:
    """Max consecutive wins/losses count and their total profit."""
    mask = pnl > 0 if is_win else pnl < 0
    max_count = 0
    max_total = 0
    current_count = 0
    current_total = 0

    for i in range(len(pnl)):
        if mask[i]:
            current_count += 1
            current_total += pnl[i]
            if current_count > max_count:
                max_count = current_count
                max_total = current_total
        else:
            current_count = 0
            current_total = 0

    return max_count, max_total


def _max_consecutive_profit(pnl: np.ndarray, is_win: bool) -> tuple:
    """Max consecutive profit/loss ($) and its count."""
    mask = pnl > 0 if is_win else pnl < 0
    max_profit = 0
    max_count = 0
    current_profit = 0
    current_count = 0

    for i in range(len(pnl)):
        if mask[i]:
            current_count += 1
            current_profit += pnl[i]
            if (is_win and current_profit > max_profit) or (not is_win and current_profit < max_profit):
                max_profit = current_profit
                max_count = current_count
        else:
            current_count = 0
            current_profit = 0

    return max_profit, max_count


def _avg_consecutive(pnl: np.ndarray, is_win: bool) -> float:
    """Average consecutive wins/losses."""
    mask = pnl > 0 if is_win else pnl < 0
    streaks = []
    current = 0

    for m in mask:
        if m:
            current += 1
        else:
            if current > 0:
                streaks.append(current)
            current = 0
    if current > 0:
        streaks.append(current)

    return float(np.mean(streaks)) if streaks else 0


def print_mt5_report(report: dict, title: str = "Strategy Tester Report"):
    """In report giống MT5 Strategy Tester layout."""
    print()
    print(f"{'='*90}")
    print(f"  {title}")
    print(f"{'='*90}")

    # Row 1
    _row("Bars", report.get("bars", 0),
         "", "",
         "", "")

    print(f"{'─'*90}")

    # Row 2: Deposits & DD
    _row("Initial Deposit", f"${report.get('initial_deposit', 0):,.2f}",
         "", "",
         "", "")

    _row("Total Net Profit", f"${report.get('total_net_profit', 0):,.2f}",
         "Balance DD Absolute", f"${report.get('balance_dd_absolute', 0):,.2f}",
         "Equity DD Absolute", f"${report.get('equity_dd_absolute', 0):,.2f}")

    _row("Gross Profit", f"${report.get('gross_profit', 0):,.2f}",
         "Balance DD Maximal", f"${report.get('balance_dd_maximal', 0):,.2f} ({report.get('balance_dd_maximal_pct', 0):.2f}%)",
         "Equity DD Maximal", f"${report.get('equity_dd_maximal', 0):,.2f} ({report.get('equity_dd_maximal_pct', 0):.2f}%)")

    _row("Gross Loss", f"${report.get('gross_loss', 0):,.2f}",
         "Balance DD Relative", report.get("balance_dd_relative", "0"),
         "Equity DD Relative", report.get("equity_dd_relative", "0"))

    print(f"{'─'*90}")

    # Row 3: Ratios
    _row("Profit Factor", f"{report.get('profit_factor', 0):.2f}",
         "Expected Payoff", f"${report.get('expected_payoff', 0):.2f}",
         "Margin Level", "")

    _row("Recovery Factor", f"{report.get('recovery_factor', 0):.2f}",
         "Sharpe Ratio", f"{report.get('sharpe_ratio', 0):.2f}",
         "Z-Score", f"{report.get('z_score', 0):.2f}")

    _row("AHPR", f"{report.get('ahpr', 1):.4f}",
         "LR Correlation", f"{report.get('lr_correlation', 0):.2f}",
         "", "")

    _row("GHPR", f"{report.get('ghpr', 1):.4f}",
         "LR Standard Error", f"{report.get('lr_std_error', 0):.2f}",
         "", "")

    print(f"{'─'*90}")

    # Row 4: Trade stats
    _row("Total Trades", report.get("total_trades", 0),
         f"Short Trades (won %)",
         f"{report.get('short_trades_won', 0)} ({report.get('short_trades_won_pct', 0):.2f}%)",
         f"Long Trades (won %)",
         f"{report.get('long_trades_won', 0)} ({report.get('long_trades_won_pct', 0):.2f}%)")

    _row("Total Deals", report.get("total_deals", 0),
         "Profit Trades (% of total)",
         f"{report.get('profit_trades', 0)} ({report.get('profit_trades_pct', 0):.2f}%)",
         "Loss Trades (% of total)",
         f"{report.get('loss_trades', 0)} ({report.get('loss_trades_pct', 0):.2f}%)")

    print(f"{'─'*90}")

    _row("", "",
         "Largest", "",
         "", "")
    _row("", "",
         "  profit trade", f"${report.get('largest_profit_trade', 0):.2f}",
         "  loss trade", f"${report.get('largest_loss_trade', 0):.2f}")

    _row("", "",
         "Average", "",
         "", "")
    _row("", "",
         "  profit trade", f"${report.get('average_profit_trade', 0):.2f}",
         "  loss trade", f"${report.get('average_loss_trade', 0):.2f}")

    print(f"{'─'*90}")

    _row("", "",
         "Maximum", "",
         "", "")
    _row("", "",
         f"  consecutive wins ($)",
         f"{report.get('max_consecutive_wins', 0)} (${report.get('max_consecutive_wins_profit', 0):.2f})",
         f"  consecutive losses ($)",
         f"{report.get('max_consecutive_losses', 0)} (${report.get('max_consecutive_losses_loss', 0):.2f})")

    _row("", "",
         "Maximal", "",
         "", "")
    _row("", "",
         f"  consecutive profit (count)",
         f"${report.get('maximal_consecutive_profit', 0):.2f} ({report.get('maximal_consecutive_profit_count', 0)})",
         f"  consecutive loss (count)",
         f"${report.get('maximal_consecutive_loss', 0):.2f} ({report.get('maximal_consecutive_loss_count', 0)})")

    _row("", "",
         "Average", "",
         "", "")
    _row("", "",
         f"  consecutive wins",
         f"{report.get('avg_consecutive_wins', 0):.0f}",
         f"  consecutive losses",
         f"{report.get('avg_consecutive_losses', 0):.0f}")

    print(f"{'='*90}")
    print()


def _row(c1, v1, c2="", v2="", c3="", v3=""):
    """Print 1 row với 3 columns giống MT5."""
    col1 = f"  {c1:<22}{str(v1):>8}" if c1 else " " * 30
    col2 = f"  {c2:<28}{str(v2):>12}" if c2 else " " * 40
    col3 = f"  {c3:<28}{str(v3):>12}" if c3 else ""
    print(f"{col1}{col2}{col3}")
