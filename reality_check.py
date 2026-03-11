"""
Reality Check ~5 phút:
Test model rl_v2_ichi_mtf (best known) trên 25 tháng data.
Rồi so sánh với 3 cải tiến anti-overfitting để thấy hướng đi.
"""
import sys, json, time
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO
from src.data.data_loader import load_csv
from src.data.multi_tf_engine import build_multi_tf_features, normalize_features
from src.env.trading_env import XAUUSDTradingEnv
import logging
logging.disable(logging.CRITICAL)

INITIAL_BALANCE = 500.0
LOT_SIZE = 0.05
SL, TP = 5.0, 3.0

ALL_MONTHS = (
    [(2024, f"{m:02d}") for m in range(1, 13)] +
    [(2025, f"{m:02d}") for m in range(1, 13)] +
    [(2026, "01")]
)

def load_and_build(year, month):
    f = PROJECT_ROOT / f"XAUUSD_{year}_{month}.csv"
    if not f.exists():
        return None, None
    df_m1 = load_csv(f)
    df = build_multi_tf_features(df_m1, timeframes=["M5", "M15", "H1", "H4"])
    df = normalize_features(df)
    feat_cols = [c for c in df.columns if c not in ["open","high","low","close","volume"]]
    return df, feat_cols

def run_month(model, year, month, balance=INITIAL_BALANCE, lot=LOT_SIZE,
              max_trades_day=999, cost_per_trade=0.0):
    df, feat_cols = load_and_build(year, month)
    if df is None:
        return None
    env = XAUUSDTradingEnv(
        df=df, feature_columns=feat_cols,
        initial_balance=balance, lot_size=lot,
        spread=0.36, reward_mode="sharpe",
        stop_loss=SL, take_profit=TP,
        use_sl_tp=True, trade_cooldown=5,
        random_start=False,
    )
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, t, tr, _ = env.step(action)
        done = t or tr
    s = env.get_performance_summary()
    # Apply extra transaction cost if specified
    extra_cost = s["total_trades"] * cost_per_trade
    return {
        "return_pct": round(s["total_return"] - extra_cost/balance*100, 2),
        "pnl": round(s["total_pnl"] - extra_cost, 2),
        "win_rate": round(s["win_rate"], 1),
        "profit_factor": round(s["profit_factor"], 3),
        "max_dd": round(s["max_drawdown"], 2),
        "trades": s["total_trades"],
    }

def print_table(results, label):
    months_sorted = sorted(results.keys())
    rets = [v["return_pct"] for v in results.values() if v]
    wins = sum(1 for r in rets if r > 0)
    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"{'='*80}")
    print(f"  {'Month':<9} | {'Ret%':>7} | {'PnL$':>8} | {'WR':>6} | {'PF':>5} | {'DD':>6} | {'Trades':>6}")
    print(f"  {'-'*9}-+-{'-'*7}-+-{'-'*8}-+-{'-'*6}-+-{'-'*5}-+-{'-'*6}-+-{'-'*6}")
    for mo in months_sorted:
        v = results.get(mo)
        if not v:
            continue
        flag = "🟢" if v["return_pct"] > 5 else ("🔴" if v["return_pct"] < -10 else "🟡")
        print(f"  {mo:<9} | {v['return_pct']:>6.2f}% | ${v['pnl']:>7.2f} | "
              f"{v['win_rate']:>5.1f}% | {v['profit_factor']:>4.2f} | "
              f"{v['max_dd']:>5.2f}% | {v['trades']:>6} {flag}")
    if rets:
        arr = np.array(rets)
        cháy = (arr < -30).sum()
        print(f"\n  {'─'*78}")
        print(f"  📊 Tổng hợp: {wins}/{len(rets)} tháng có lời ({wins/len(rets)*100:.0f}%)")
        print(f"  📈 Avg/Median: {arr.mean():.2f}% / {np.median(arr):.2f}%")
        print(f"  🏆 Best: {arr.max():.2f}%  |  💀 Worst: {arr.min():.2f}%")
        print(f"  ☠️  Cháy (<-30%): {cháy} tháng")

# ─── Load models ──────────────────────────────────────────────
print("\n⏳ Loading models...")
t0 = time.time()

model_ichi = None
model_best = None

p1 = PROJECT_ROOT / "models" / "rl_v2_ichi_mtf.zip"
p2 = PROJECT_ROOT / "models" / "rl_v2_BEST.zip"
if p1.exists():
    model_ichi = PPO.load(str(p1))
    print(f"  ✅ Loaded: rl_v2_ichi_mtf")
if p2.exists():
    model_best = PPO.load(str(p2))
    print(f"  ✅ Loaded: rl_v2_BEST (current training)")

# ─── Test 1: ichi_mtf trên tất cả tháng (baseline) ───────────
if model_ichi:
    print(f"\n🔬 Testing rl_v2_ichi_mtf trên {len(ALL_MONTHS)} tháng...")
    res_ichi = {}
    for year, month in ALL_MONTHS:
        r = run_month(model_ichi, year, month)
        if r:
            res_ichi[f"{year}-{month}"] = r
            ret = r['return_pct']
            flag = "🟢" if ret > 5 else ("🔴" if ret < -10 else "🟡")
            print(f"  {year}-{month}: {ret:>7.2f}%  PF={r['profit_factor']:.2f}  trades={r['trades']} {flag}")
    print_table(res_ichi, "rl_v2_ichi_mtf — BASELINE (500$ / 0.05 lot)")

# ─── Test 2: ichi_mtf với transaction cost $1/trade penalty ───
if model_ichi:
    print(f"\n🔬 Testing ichi_mtf + Transaction Cost ($1/trade extra penalty)...")
    res_cost = {}
    for year, month in ALL_MONTHS:
        r = run_month(model_ichi, year, month, cost_per_trade=1.0)
        if r:
            res_cost[f"{year}-{month}"] = r
    print_table(res_cost, "rl_v2_ichi_mtf + $1/trade cost — Realistic Cost Scenario")

# ─── Test 3: rl_v2_BEST (current training) vs ichi_mtf ────────
if model_best and model_ichi:
    print(f"\n🔬 Quick compare Jan 2026: NEW vs OLD model...")
    r_new = run_month(model_best, 2026, "01")
    r_old = run_month(model_ichi, 2026, "01")
    print(f"\n  Model              | Return  | PF    | WR    | DD    | Trades")
    print(f"  {'-'*18}-+-{'-'*7}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}-+-{'-'*6}")
    if r_old: print(f"  ichi_mtf (old)     | {r_old['return_pct']:>6.2f}% | {r_old['profit_factor']:>5.2f} | {r_old['win_rate']:>5.1f}% | {r_old['max_dd']:>5.2f}% | {r_old['trades']}")
    if r_new: print(f"  BEST (new $500)    | {r_new['return_pct']:>6.2f}% | {r_new['profit_factor']:>5.2f} | {r_new['win_rate']:>5.1f}% | {r_new['max_dd']:>5.2f}% | {r_new['trades']}")

# ─── Save JSON cho walkthrough ─────────────────────────────────
out = {"ichi_mtf_all_months": res_ichi if model_ichi else {}}
with open("reality_check_results.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

elapsed = time.time() - t0
print(f"\n⏱️  Done in {elapsed:.0f}s ({elapsed/60:.1f} phút)")
print(f"💾  Saved: reality_check_results.json")
print(f"\n💡 HƯỚNG CẢI TIẾN RÕ NHẤT:")
if model_ichi and res_ichi:
    rets = np.array([v["return_pct"] for v in res_ichi.values()])
    bad_months = sum(1 for r in rets if r < -10)
    if bad_months > len(rets) * 0.4:
        print("   → Bot KHÔNG stable: >40% tháng thua >10%. Cần Walk-Forward + Validation set.")
    elif rets.mean() < 3:
        print("   → Bot profitable nhưng yếu (~3%/tháng). Tăng SL/TP hoặc lotsize.")
    else:
        print("   → Bot ổn định! Focus vào live trading với risk management.")
