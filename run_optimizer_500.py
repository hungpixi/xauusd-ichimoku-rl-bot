"""
Rule-based Ichimoku Optimizer — Config $500 / Đòn bẩy 200x.
=============================================================
Target: +$500/tháng (100% return) trên tài khoản $500.

Pipeline:
1. Grid search Ichimoku params trên TRAIN data (Oct-Nov-Dec 2025)
2. Rank top N theo composite score: balance > profit_factor > recovery_factor > max_dd
3. Multi-month backtest top 3 trên TẤT CẢ tháng available
4. Export top3_configs.json + in báo cáo đầy đủ
5. Gọi generate_mql5.py để tạo MQL5 EA
"""

import sys
import json
import logging
from pathlib import Path
from itertools import product
from dataclasses import asdict

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_loader import load_csv
from src.data.resampler import resample_ohlcv
from src.strategy.ichimoku_strategy import StrategyParams, run_backtest

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
INITIAL_BALANCE = 500.0
LOT_SIZE = 0.05          # $500 × 200x → ~$100k margin → 0.05 lot XAU (conservative)
TIMEFRAME = "M5"

TRAIN_YEAR = 2025
TRAIN_MONTHS = ["10", "11", "12"]  # Q4 2025

# Tất cả tháng có data để multi-month test
ALL_TEST_PERIODS = [
    (2024, m) for m in ["01","02","03","04","05","06","07","08","09","10","11","12"]
] + [
    (2025, m) for m in ["01","02","03","04","05","06","07","08","09","10","11","12"]
] + [
    (2026, "01")
]

OUTPUT_PATH = PROJECT_ROOT / "models" / "top3_configs.json"

# ─────────────────────────────────────────────
# GRID SEARCH PARAMS (~320 combos, ~30s chạy)
# Giảm từ 1920 bằng cách cố định trailing params (default đủ tốt)
# ─────────────────────────────────────────────
PARAM_GRID = {
    "stop_loss":         [3.0, 5.0, 8.0, 10.0, 15.0],
    "take_profit":       [3.0, 5.0, 8.0, 10.0],
    "use_trailing_stop": [True, False],
    "trailing_start":    [1.5],      # fixed default
    "trailing_step":     [0.5],      # fixed default
    "cooldown_bars":     [3, 5, 10],
    "use_tk_cross":      [False, True],
    "use_ema_filter":    [True, False],
}


def load_month_m5(year: int, month: str) -> pd.DataFrame | None:
    """Load 1 tháng M1 → resample M5."""
    path = PROJECT_ROOT / f"XAUUSD_{year}_{month}.csv"
    if not path.exists():
        return None
    df_m1 = load_csv(path)
    if TIMEFRAME != "M1":
        return resample_ohlcv(df_m1, TIMEFRAME)
    return df_m1


def compute_metrics(result: dict) -> dict:
    """Tính composite score + tất cả metrics từ backtest result."""
    trades = result["trades"]
    eq = result["equity_curve"]
    initial = result["initial_balance"]
    final_bal = result["final_balance"]

    if not trades:
        return {
            "net_profit": 0, "final_balance": initial, "return_pct": 0,
            "profit_factor": 0, "win_rate": 0, "total_trades": 0,
            "max_dd_pct": 0, "recovery_factor": 0, "sharpe": 0,
            "composite_score": -9999, "monthly_stats": {}
        }

    pnls = np.array([t.pnl for t in trades])
    net = pnls.sum()
    gp = pnls[pnls > 0].sum()
    gl = abs(pnls[pnls < 0].sum())
    pf = gp / (gl + 1e-10)
    wr = (pnls > 0).sum() / len(pnls) * 100

    peak = np.maximum.accumulate(eq)
    dd_arr = (peak - eq) / (peak + 1e-10) * 100
    max_dd = dd_arr.max()

    rf = net / (max_dd * initial / 100 + 1e-10)

    if len(eq) > 1:
        rets = np.diff(eq) / (np.abs(eq[:-1]) + 1e-10)
        sharpe = float(rets.mean() / (rets.std() + 1e-10) * np.sqrt(252 * 24 * 12))
    else:
        sharpe = 0.0

    return_pct = (final_bal - initial) / initial * 100

    # Composite: normalize từng thành phần rồi weighted sum
    # Ưu tiên: balance > PF > RF > max_dd (theo yêu cầu)
    bal_score = return_pct / 100          # 0.0 → ~1.0 nếu 100% return
    pf_score = min(pf / 5.0, 1.0)        # cap tại PF=5
    rf_score = min(rf / 10.0, 1.0)       # cap tại RF=10
    dd_penalty = max_dd / 100            # 0.0 (no DD) → 1.0 (100% DD)

    composite = bal_score * 0.40 + pf_score * 0.30 + rf_score * 0.20 - dd_penalty * 0.10

    return {
        "net_profit": round(float(net), 2),
        "final_balance": round(float(final_bal), 2),
        "return_pct": round(float(return_pct), 3),
        "profit_factor": round(float(pf), 4),
        "win_rate": round(float(wr), 2),
        "total_trades": len(trades),
        "max_dd_pct": round(float(max_dd), 3),
        "recovery_factor": round(float(rf), 4),
        "sharpe": round(float(sharpe), 4),
        "composite_score": round(float(composite), 6),
    }


def run_grid_search(train_df: pd.DataFrame) -> list:
    """
    Grid search toàn bộ param combos trên train data.
    Returns list sorted by composite_score desc.
    """
    keys = list(PARAM_GRID.keys())
    values = [PARAM_GRID[k] for k in keys]
    combos = list(product(*values))
    total = len(combos)
    logger.info(f"🔍 Grid search: {total} combinations trên {len(train_df)} bars M5")

    results = []
    for i, combo in enumerate(combos):
        params_dict = dict(zip(keys, combo))
        params = StrategyParams(
            lot_size=LOT_SIZE,
            **params_dict
        )

        result = run_backtest(train_df, params, INITIAL_BALANCE)
        m = compute_metrics(result)
        m["params"] = params_dict

        if m["total_trades"] >= 5:  # Lọc bỏ configs trade quá ít
            results.append(m)

        if (i + 1) % 100 == 0:
            logger.info(f"  Progress: {i+1}/{total}")

    results.sort(key=lambda x: x["composite_score"], reverse=True)
    logger.info(f"✅ Done. {len(results)} valid configs.")
    return results


def multi_month_backtest(params_dict: dict, label: str) -> dict:
    """
    Test 1 config trên tất cả tháng available.
    Returns { "YYYY-MM": metrics_dict }
    """
    params = StrategyParams(lot_size=LOT_SIZE, **params_dict)
    monthly = {}
    blow_up_months = []

    for year, month in ALL_TEST_PERIODS:
        df = load_month_m5(year, month)
        if df is None or len(df) < 100:
            continue

        result = run_backtest(df, params, INITIAL_BALANCE)
        m = compute_metrics(result)
        key = f"{year}-{month}"
        monthly[key] = {
            "return_pct": m["return_pct"],
            "net_profit": m["net_profit"],
            "profit_factor": m["profit_factor"],
            "max_dd_pct": m["max_dd_pct"],
            "total_trades": m["total_trades"],
            "final_balance": m["final_balance"],
        }
        if m["return_pct"] <= -50:
            blow_up_months.append(key)

    # Summary stats
    returns = [v["return_pct"] for v in monthly.values()]
    pfs = [v["profit_factor"] for v in monthly.values()]
    dds = [v["max_dd_pct"] for v in monthly.values()]

    summary = {
        "avg_monthly_return": round(float(np.mean(returns)), 2) if returns else 0,
        "median_monthly_return": round(float(np.median(returns)), 2) if returns else 0,
        "best_month": round(float(max(returns)), 2) if returns else 0,
        "worst_month": round(float(min(returns)), 2) if returns else 0,
        "avg_profit_factor": round(float(np.mean(pfs)), 3) if pfs else 0,
        "avg_max_dd": round(float(np.mean(dds)), 2) if dds else 0,
        "profitable_months": int(sum(1 for r in returns if r > 0)),
        "total_months_tested": len(returns),
        "blow_up_months": blow_up_months,
        "win_rate_months": round(sum(1 for r in returns if r > 0) / (len(returns) + 1e-10) * 100, 1),
    }

    return {"monthly": monthly, "summary": summary}


def print_summary_table(top3: list):
    """In bảng so sánh top 3 dạng dễ đọc."""
    print(f"\n{'='*120}")
    print(f"  🏆 TOP 3 ICHIMOKU CONFIGS — XAUUSD $500/200x | TRAIN: Q4 2025 | SORT: balance>PF>RF>DD")
    print(f"{'='*120}")
    header = f"  {'Rank':<5} | {'Net $':>8} | {'Ret %':>7} | {'PF':>6} | {'WR':>6} | {'DD':>7} | {'RF':>6} | {'Score':>8} | Params"
    print(header)
    print(f"  {'─'*5}─┼─{'─'*8}─┼─{'─'*7}─┼─{'─'*6}─┼─{'─'*6}─┼─{'─'*7}─┼─{'─'*6}─┼─{'─'*8}─┼─{'─'*50}")

    for i, cfg in enumerate(top3):
        p = cfg["train_metrics"]
        params_str = ", ".join(f"{k}={v}" for k, v in cfg["params"].items()
                               if k not in ["lot_size", "tenkan_period", "kijun_period", "senkou_period",
                                            "ema_fast", "ema_slow", "use_cloud_break", "require_cloud_color",
                                            "use_dca", "dca_distance", "max_dca_orders", "spread",
                                            "max_trades_per_day"])
        print(f"  #{i+1:<4} | ${p['net_profit']:>7.2f} | {p['return_pct']:>6.2f}% | {p['profit_factor']:>5.3f} | "
              f"{p['win_rate']:>5.1f}% | {p['max_dd_pct']:>6.2f}% | {p['recovery_factor']:>5.2f} | "
              f"{p['composite_score']:>8.5f} | {params_str}")

    print(f"{'='*120}")

    print(f"\n{'='*120}")
    print(f"  📅 MULTI-MONTH TEST (tất cả tháng available)")
    print(f"{'='*120}")

    # Print monthly comparison
    for i, cfg in enumerate(top3):
        s = cfg["multi_month"]["summary"]
        blow = cfg["multi_month"]["summary"]["blow_up_months"]
        blow_str = ", ".join(blow) if blow else "none 🟢"
        print(f"\n  Config #{i+1}:")
        print(f"    Avg return/tháng : {s['avg_monthly_return']:>7.2f}%  |  Median: {s['median_monthly_return']:>7.2f}%")
        print(f"    Best/Worst tháng : {s['best_month']:>7.2f}% / {s['worst_month']:>7.2f}%")
        print(f"    Tháng có lời     : {s['profitable_months']}/{s['total_months_tested']} ({s['win_rate_months']:.1f}%)")
        print(f"    Avg PF / Avg DD  : {s['avg_profit_factor']:>5.2f} / {s['avg_max_dd']:>6.2f}%")
        print(f"    ⚠️  Cháy tài khoản (DD>50%): {blow_str}")

        # Monthly breakdown table
        monthly = cfg["multi_month"]["monthly"]
        months = sorted(monthly.keys())
        print(f"\n    {'Month':<9} | {'Return':>8} | {'Net $':>8} | {'PF':>6} | {'DD':>7} | {'Trades':>6}")
        print(f"    {'─'*9}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*6}─┼─{'─'*7}─┼─{'─'*6}")
        for mo in months:
            mv = monthly[mo]
            ret_flag = "🟢" if mv["return_pct"] > 5 else ("🔴" if mv["return_pct"] < -20 else "🟡")
            print(f"    {mo:<9} | {mv['return_pct']:>7.2f}% | ${mv['net_profit']:>7.2f} | "
                  f"{mv['profit_factor']:>5.2f} | {mv['max_dd_pct']:>6.2f}% | {mv['total_trades']:>6} {ret_flag}")

    print(f"\n{'='*120}")


def main():
    logger.info("="*60)
    logger.info("🚀 ICHIMOKU OPTIMIZER — $500/200x Config")
    logger.info("="*60)

    # ── Load & concat train data ──────────────────────────
    logger.info("📂 Loading train data: Q4 2025 (Oct-Nov-Dec)...")
    train_dfs = []
    for m in TRAIN_MONTHS:
        df = load_month_m5(TRAIN_YEAR, m)
        if df is not None:
            train_dfs.append(df)
            logger.info(f"  Loaded 2025-{m}: {len(df)} bars M5")
        else:
            logger.warning(f"  ⚠️ Missing: XAUUSD_{TRAIN_YEAR}_{m}.csv")

    if not train_dfs:
        logger.error("❌ No train data found!")
        return

    train_df = pd.concat(train_dfs).sort_index()
    train_df = train_df[~train_df.index.duplicated(keep="first")]
    logger.info(f"✅ Train data: {len(train_df)} bars total")

    # ── Grid search ───────────────────────────────────────
    logger.info("\n🔍 Running grid search...")
    all_results = run_grid_search(train_df)

    if not all_results:
        logger.error("❌ Không có config nào hợp lệ!")
        return

    top3_raw = all_results[:3]
    logger.info(f"✅ Top 3 configs identified.")

    # ── Multi-month backtest ──────────────────────────────
    logger.info("\n📅 Running multi-month backtest for top 3...")
    top3_final = []
    for i, cfg in enumerate(top3_raw):
        logger.info(f"  Testing Config #{i+1}...")
        mm_result = multi_month_backtest(cfg["params"], f"Config #{i+1}")

        top3_final.append({
            "rank": i + 1,
            "params": cfg["params"],
            "lot_size": LOT_SIZE,
            "initial_balance": INITIAL_BALANCE,
            "train_metrics": {k: v for k, v in cfg.items() if k != "params"},
            "multi_month": mm_result,
        })

    # ── Print summary ─────────────────────────────────────
    print_summary_table(top3_final)

    # ── Save JSON ─────────────────────────────────────────
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(top3_final, f, indent=2, ensure_ascii=False)
    logger.info(f"\n💾 Saved: {OUTPUT_PATH}")

    # ── Generate MQL5 ─────────────────────────────────────
    logger.info("\n🖊️  Generating MQL5 EAs...")
    try:
        import importlib.util
        gen_path = PROJECT_ROOT / "generate_mql5.py"
        if gen_path.exists():
            spec = importlib.util.spec_from_file_location("generate_mql5", gen_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod.generate_all(top3_final)
            logger.info("✅ MQL5 EAs generated!")
        else:
            logger.warning("⚠️  generate_mql5.py không tìm thấy, bỏ qua.")
    except Exception as e:
        logger.error(f"❌ MQL5 gen lỗi: {e}")

    logger.info("\n✅ ALL DONE!")
    logger.info(f"   JSON output: {OUTPUT_PATH}")
    logger.info(f"   MQL5 output: {PROJECT_ROOT / 'mql5_output/'}")


if __name__ == "__main__":
    main()
