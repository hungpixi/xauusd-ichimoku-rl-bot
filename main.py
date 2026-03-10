"""
🤖 XAUUSD Self-Improving Trading Bot (Ichimoku + PPO + RBI)
Main entry point.

Quy trình chính (ưu tiên):
    # 1. Progressive Validation: 1 tháng → 3 tháng → 1 năm
    python main.py --mode progressive --timesteps 200000

    # 2. RBI Self-Improving Loop (5 iterations)
    python main.py --mode rbi --iterations 5

Các mode khác:
    python main.py --mode smoke           # Quick test
    python main.py --mode train           # Full training
    python main.py --mode eval            # Evaluate model
    python main.py --mode walkforward     # Walk-forward test
"""

import sys
import argparse
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


def run_smoke_test(args):
    """Quick smoke test với Ichimoku features."""
    from src.data.data_loader import load_csv
    from src.data.resampler import resample_ohlcv
    from src.data.ichimoku_features import compute_ichimoku_features, normalize_features
    from src.env.trading_env import XAUUSDTradingEnv
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    logger.info("🧪 SMOKE TEST: Ichimoku PPO on Jan 2026...")

    # Load data
    csv_file = PROJECT_ROOT / "XAUUSD_2026_01.csv"
    if not csv_file.exists():
        csv_file = PROJECT_ROOT / "XAUUSD_2024_01.csv"
    df_m1 = load_csv(csv_file)
    df_m5 = resample_ohlcv(df_m1, "M5")
    df = compute_ichimoku_features(df_m5)
    df = normalize_features(df)

    feature_cols = [c for c in df.columns if c not in ["open", "high", "low", "close", "volume"]]

    # Train
    env = Monitor(XAUUSDTradingEnv(df=df, feature_columns=feature_cols, reward_mode="sharpe"))
    model = PPO("MlpPolicy", env, verbose=1, device="auto")
    model.learn(total_timesteps=10_000, progress_bar=True)

    # Evaluate
    test_env = XAUUSDTradingEnv(df=df, feature_columns=feature_cols, random_start=False)
    obs, _ = test_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = test_env.step(action)
        done = terminated or truncated

    summary = test_env.get_performance_summary()
    logger.info(f"\n{'='*50}")
    logger.info(f"📊 SMOKE TEST RESULTS:")
    for k, v in summary.items():
        logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    logger.info(f"{'='*50}")
    logger.info("✅ Smoke test complete!")


def run_progressive(args):
    """Progressive Validation: 1 tháng → 3 tháng → 1 năm."""
    from src.rbi.progressive_validation import progressive_validation

    results = progressive_validation(
        data_dir=PROJECT_ROOT,
        output_dir=PROJECT_ROOT / "models",
        timeframe=args.timeframe,
        timesteps_phase1=args.timesteps,
        timesteps_phase2=int(args.timesteps * 1.5),
        timesteps_phase3=int(args.timesteps * 2.5),
        config={
            "spread": 0.30,
            "reward_mode": args.reward,
            "learning_rate": args.lr,
        },
    )


def run_train(args):
    """Full Ichimoku training."""
    from src.rbi.progressive_validation import train_and_evaluate

    train_months = [args.month] if args.month else [f"{m:02d}" for m in range(1, 13)]
    test_months = [args.eval_month] if args.eval_month else ["01"]

    train_and_evaluate(
        data_dir=PROJECT_ROOT,
        output_dir=PROJECT_ROOT / "models",
        train_year=args.year, train_months=train_months,
        test_year=args.eval_year, test_months=test_months,
        timeframe=args.timeframe,
        timesteps=args.timesteps,
        config={"reward_mode": args.reward, "learning_rate": args.lr},
    )


def run_eval(args):
    """Evaluate model."""
    from src.models.evaluate import evaluate_model
    evaluate_model(
        model_path=args.model, data_dir=PROJECT_ROOT,
        year=args.eval_year, month=args.eval_month, timeframe=args.timeframe,
    )


def run_walkforward(args):
    """Walk-forward test."""
    from src.models.evaluate import walk_forward_test
    walk_forward_test(
        model_path=args.model, data_dir=PROJECT_ROOT,
        year=args.eval_year, timeframe=args.timeframe,
    )


def run_rbi(args):
    """RBI Self-Improving Loop."""
    from src.rbi.rbi_loop import RBILoop
    rbi = RBILoop(
        data_dir=PROJECT_ROOT, output_dir=PROJECT_ROOT / "models",
        base_config={"train_year": args.year, "timeframe": args.timeframe, "total_timesteps": args.timesteps},
        max_iterations=args.iterations,
    )
    results = rbi.run(eval_year=args.eval_year, eval_month=args.eval_month or "01")
    logger.info(f"\n🏆 Best model: {results['best_model']}")


def main():
    parser = argparse.ArgumentParser(
        description="🤖 XAUUSD Ichimoku Self-Improving Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", type=str, required=True,
                        choices=["smoke", "progressive", "train", "eval", "walkforward", "rbi"])
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--timeframe", type=str, default="M5")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--month", type=str, default=None)
    parser.add_argument("--eval-year", type=int, default=2025)
    parser.add_argument("--eval-month", type=str, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--reward", type=str, default="sharpe")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    mode_map = {
        "smoke": run_smoke_test,
        "progressive": run_progressive,
        "train": run_train,
        "eval": run_eval,
        "walkforward": run_walkforward,
        "rbi": run_rbi,
    }
    mode_map[args.mode](args)


if __name__ == "__main__":
    main()
