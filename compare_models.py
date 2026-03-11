"""
So sánh 2 model RL trên test data Jan 2026.
- rl_v2_BEST.zip  (train Q4 2025, auto-saved best checkpoint)
- rl_v2_ichi_mtf.zip (progressive training)
"""

import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO
from src.data.data_loader import load_csv
from src.data.multi_tf_engine import build_multi_tf_features, normalize_features
from src.env.trading_env import XAUUSDTradingEnv
from src.models.mt5_report import compute_mt5_report, print_mt5_report

logging.basicConfig(level=logging.WARNING)

def test_model(model_path: Path, label: str, sl=5.0, tp=3.0):
    print(f"\n{'='*60}")
    print(f"  🧪 Testing: {label}")
    print(f"  File: {model_path.name}")
    print(f"{'='*60}")

    # Load data Jan 2026
    data_dir = PROJECT_ROOT
    try:
        from src.data.data_loader import load_csv
        df_m1 = load_csv(data_dir / "XAUUSD_2026_01.csv")
    except Exception as e:
        print(f"  ❌ Load data lỗi: {e}")
        return None

    # Build features
    print("  🔧 Building features...")
    df = build_multi_tf_features(df_m1, timeframes=["M5", "M15", "H1", "H4"])
    df = normalize_features(df)
    feature_cols = [c for c in df.columns if c not in ["open","high","low","close","volume"]]
    print(f"  ✅ {len(feature_cols)} features, {len(df)} bars")

    # Load model
    try:
        model = PPO.load(str(model_path))
    except Exception as e:
        print(f"  ❌ Load model lỗi: {e}")
        return None

    # Run test
    env = XAUUSDTradingEnv(
        df=df, feature_columns=feature_cols,
        initial_balance=10000, lot_size=0.01,
        spread=0.36, reward_mode="sharpe",
        stop_loss=sl, take_profit=tp,
        use_sl_tp=True, trade_cooldown=5,
        random_start=False,
    )

    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    # Report
    trade_log = env.get_trade_log()
    mt5 = compute_mt5_report(
        trade_log=trade_log,
        equity_history=env.equity_history,
        initial_balance=10000,
        total_bars=len(df),
    )
    print_mt5_report(mt5, title=f"{label} | XAUUSD Jan 2026")

    summary = env.get_performance_summary()
    return summary


if __name__ == "__main__":
    models_dir = PROJECT_ROOT / "models"

    results = {}

    # Test rl_v2_ichi_mtf (train progressive)
    p1 = models_dir / "rl_v2_ichi_mtf.zip"
    if p1.exists():
        results["ichi_mtf"] = test_model(p1, "RL v2 Ichi-MTF (progressive)")

    # Test rl_v2_BEST
    p2 = models_dir / "rl_v2_BEST.zip"
    if p2.exists():
        results["best"] = test_model(p2, "RL v2 BEST (Q4 2025 checkpoint)")

    # Summary comparison
    print(f"\n{'='*70}")
    print(f"  📊 COMPARISON SUMMARY — Jan 2026")
    print(f"  {'Model':<30} | {'Return':>8} | {'WR':>7} | {'PF':>6} | {'DD':>7} | {'Trades':>6}")
    print(f"  {'─'*30}─┼─{'─'*8}─┼─{'─'*7}─┼─{'─'*6}─┼─{'─'*7}─┼─{'─'*6}")
    for name, r in results.items():
        if r:
            print(f"  {name:<30} | {r['total_return']:>7.2f}% | {r['win_rate']:>6.1f}% | {r['profit_factor']:>6.2f} | {r['max_drawdown']:>6.1f}% | {r['total_trades']:>6}")
    print(f"{'='*70}")
