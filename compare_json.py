"""Test 2 model → output JSON"""
import sys, json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO
from src.data.data_loader import load_csv
from src.data.multi_tf_engine import build_multi_tf_features, normalize_features
from src.env.trading_env import XAUUSDTradingEnv
import logging
logging.disable(logging.CRITICAL)

def test_model(model_path, sl=5.0, tp=3.0):
    df_m1 = load_csv(PROJECT_ROOT / "XAUUSD_2026_01.csv")
    df = build_multi_tf_features(df_m1, timeframes=["M5", "M15", "H1", "H4"])
    df = normalize_features(df)
    feature_cols = [c for c in df.columns if c not in ["open","high","low","close","volume"]]

    model = PPO.load(str(model_path))
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

    s = env.get_performance_summary()
    tl = env.get_trade_log()
    
    # Action distribution
    env2 = XAUUSDTradingEnv(
        df=df, feature_columns=feature_cols,
        initial_balance=10000, lot_size=0.01,
        spread=0.36, reward_mode="sharpe",
        stop_loss=sl, take_profit=tp,
        use_sl_tp=True, trade_cooldown=5,
        random_start=False,
    )
    obs2, _ = env2.reset()
    done2 = False
    actions = []
    while not done2:
        action2, _ = model.predict(obs2, deterministic=True)
        actions.append(int(action2))
        obs2, _, t2, tr2, _ = env2.step(action2)
        done2 = t2 or tr2
    
    from collections import Counter
    action_dist = Counter(actions)
    
    return {
        "total_return": round(float(s["total_return"]), 3),
        "total_pnl": round(float(s["total_pnl"]), 2),
        "win_rate": round(float(s["win_rate"]), 2),
        "profit_factor": round(float(s["profit_factor"]), 4),
        "max_drawdown": round(float(s["max_drawdown"]), 3),
        "total_trades": int(s["total_trades"]),
        "avg_win": round(float(s["avg_win"]), 4),
        "avg_loss": round(float(s["avg_loss"]), 4),
        "sharpe_ratio": round(float(s["sharpe_ratio"]), 4),
        "final_balance": round(float(s["balance"]), 2),
        "action_dist": {
            "Hold(0)": action_dist.get(0, 0),
            "Buy(1)": action_dist.get(1, 0),
            "Sell(2)": action_dist.get(2, 0),
            "Close(3)": action_dist.get(3, 0),
        }
    }

results = {}
models_dir = PROJECT_ROOT / "models"

for name, fname in [("ichi_mtf", "rl_v2_ichi_mtf.zip"), ("best", "rl_v2_BEST.zip")]:
    p = models_dir / fname
    if p.exists():
        print(f"Testing {fname}...", flush=True)
        try:
            results[name] = test_model(p)
            print(f"  Done: return={results[name]['total_return']}%", flush=True)
        except Exception as e:
            results[name] = {"error": str(e)}
            print(f"  Error: {e}", flush=True)

out = PROJECT_ROOT / "compare_results.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"Saved to {out}", flush=True)
