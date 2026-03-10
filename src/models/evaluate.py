"""
Model Evaluation & Backtesting cho XAUUSD Bot.
Chạy model đã train trên unseen data, output metrics.
"""

import sys
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_loader import load_year_data, load_csv
from src.data.resampler import resample_ohlcv
from src.data.feature_engine import compute_all_features, normalize_features
from src.env.trading_env import XAUUSDTradingEnv

logger = logging.getLogger(__name__)


def evaluate_model(
    model_path: str,
    data_dir: Path,
    year: int = 2025,
    month: str = None,
    timeframe: str = "M5",
    initial_balance: float = 10000,
    lot_size: float = 0.01,
    spread: float = 0.30,
) -> dict:
    """
    Evaluate model trên data cụ thể.
    
    Returns:
        Dict chứa performance metrics + trade log
    """
    # Load model
    model = PPO.load(model_path)
    logger.info(f"📦 Loaded model: {model_path}")

    # Prepare data
    if month:
        csv_file = data_dir / f"XAUUSD_{year}_{month}.csv"
        df_m1 = load_csv(csv_file)
    else:
        df_m1 = load_year_data(data_dir, year)

    if timeframe != "M1":
        df = resample_ohlcv(df_m1, timeframe)
    else:
        df = df_m1

    df = compute_all_features(df)
    df = normalize_features(df)

    feature_columns = [
        c for c in df.columns
        if c not in ["open", "high", "low", "close", "volume"]
    ]

    # Create env (no random start for evaluation)
    env = XAUUSDTradingEnv(
        df=df,
        feature_columns=feature_columns,
        initial_balance=initial_balance,
        lot_size=lot_size,
        spread=spread,
        reward_mode="sharpe",
        random_start=False,
    )

    # Run evaluation
    obs, info = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

    # Get results
    summary = env.get_performance_summary()
    trade_log = env.get_trade_log()
    equity_curve = env.equity_history

    results = {
        "model_path": str(model_path),
        "eval_year": year,
        "eval_month": month,
        "timeframe": timeframe,
        "metrics": summary,
        "total_reward": total_reward,
        "equity_curve": equity_curve,
    }

    logger.info(f"\n{'='*50}")
    logger.info(f"📊 EVALUATION RESULTS ({year}-{month or 'full'})")
    logger.info(f"{'='*50}")
    logger.info(f"  Total Return:   {summary['total_return']:.2f}%")
    logger.info(f"  Sharpe Ratio:   {summary['sharpe_ratio']:.4f}")
    logger.info(f"  Max Drawdown:   {summary['max_drawdown']:.2f}%")
    logger.info(f"  Win Rate:       {summary['win_rate']:.1f}%")
    logger.info(f"  Total Trades:   {summary['total_trades']}")
    logger.info(f"  Profit Factor:  {summary['profit_factor']:.2f}")
    logger.info(f"  Avg Win:        ${summary['avg_win']:.2f}")
    logger.info(f"  Avg Loss:       ${summary['avg_loss']:.2f}")
    logger.info(f"  Total PnL:      ${summary['total_pnl']:.2f}")
    logger.info(f"{'='*50}")

    return results, trade_log


def walk_forward_test(
    model_path: str,
    data_dir: Path,
    test_months: list = None,
    year: int = 2025,
    timeframe: str = "M5",
    **kwargs,
) -> pd.DataFrame:
    """
    Walk-forward test: evaluate từng tháng riêng biệt.
    Cho thấy model hoạt động ra sao qua từng giai đoạn.
    """
    if test_months is None:
        test_months = [f"{m:02d}" for m in range(1, 13)]

    results = []
    for month in test_months:
        csv_file = data_dir / f"XAUUSD_{year}_{month}.csv"
        if not csv_file.exists():
            logger.warning(f"⚠️ Bỏ qua {csv_file.name} (không tồn tại)")
            continue

        try:
            res, _ = evaluate_model(
                model_path=model_path,
                data_dir=data_dir,
                year=year,
                month=month,
                timeframe=timeframe,
                **kwargs,
            )
            metrics = res["metrics"]
            metrics["month"] = f"{year}-{month}"
            results.append(metrics)
        except Exception as e:
            logger.error(f"❌ Lỗi evaluate {year}-{month}: {e}")

    df_results = pd.DataFrame(results)
    if not df_results.empty:
        logger.info(f"\n📊 Walk-Forward Summary ({year}):")
        logger.info(f"\n{df_results[['month', 'total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'total_trades']].to_string()}")

        # Tổng kết
        logger.info(f"\n  Avg Monthly Return: {df_results['total_return'].mean():.2f}%")
        logger.info(f"  Avg Sharpe:         {df_results['sharpe_ratio'].mean():.4f}")
        logger.info(f"  Worst DD Month:     {df_results['max_drawdown'].max():.2f}%")
        logger.info(f"  Avg Win Rate:       {df_results['win_rate'].mean():.1f}%")

    return df_results


def compare_with_baseline(
    model_path: str,
    data_dir: Path,
    year: int = 2025,
    month: str = None,
    timeframe: str = "M5",
) -> dict:
    """So sánh model với Buy-and-Hold baseline."""
    # Model performance
    model_results, _ = evaluate_model(
        model_path=model_path,
        data_dir=data_dir,
        year=year,
        month=month,
        timeframe=timeframe,
    )

    # Buy & Hold performance
    if month:
        csv_file = data_dir / f"XAUUSD_{year}_{month}.csv"
        df = load_csv(csv_file)
    else:
        df = load_year_data(data_dir, year)

    start_price = df["close"].iloc[0]
    end_price = df["close"].iloc[-1]
    bh_return = (end_price - start_price) / start_price * 100

    comparison = {
        "model_return": model_results["metrics"]["total_return"],
        "buy_hold_return": bh_return,
        "model_sharpe": model_results["metrics"]["sharpe_ratio"],
        "model_max_dd": model_results["metrics"]["max_drawdown"],
        "alpha": model_results["metrics"]["total_return"] - bh_return,
    }

    logger.info(f"\n📊 Model vs Buy-and-Hold:")
    logger.info(f"  Model Return:     {comparison['model_return']:.2f}%")
    logger.info(f"  Buy&Hold Return:  {comparison['buy_hold_return']:.2f}%")
    logger.info(f"  Alpha:            {comparison['alpha']:.2f}%")

    return comparison


def main():
    parser = argparse.ArgumentParser(description="Evaluate XAUUSD Trading Bot")
    parser.add_argument("--model", type=str, required=True, help="Path to model .zip file")
    parser.add_argument("--data", type=str, default=str(PROJECT_ROOT), help="Data directory")
    parser.add_argument("--year", type=int, default=2025, help="Test year")
    parser.add_argument("--month", type=str, default=None, help="Test month (01-12)")
    parser.add_argument("--timeframe", type=str, default="M5", help="Timeframe")
    parser.add_argument("--walk-forward", action="store_true", help="Walk-forward test all months")
    parser.add_argument("--compare", action="store_true", help="Compare with Buy&Hold")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    data_dir = Path(args.data)

    if args.walk_forward:
        df_results = walk_forward_test(
            model_path=args.model,
            data_dir=data_dir,
            year=args.year,
            timeframe=args.timeframe,
        )
        if args.output:
            df_results.to_csv(args.output, index=False)
    elif args.compare:
        compare_with_baseline(
            model_path=args.model,
            data_dir=data_dir,
            year=args.year,
            month=args.month,
            timeframe=args.timeframe,
        )
    else:
        results, trade_log = evaluate_model(
            model_path=args.model,
            data_dir=data_dir,
            year=args.year,
            month=args.month,
            timeframe=args.timeframe,
        )

        if args.output:
            # Save results
            output_data = {
                "metrics": results["metrics"],
                "total_reward": results["total_reward"],
            }
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2, default=str)

            # Save trade log
            if not trade_log.empty:
                trade_log.to_csv(Path(args.output).with_suffix(".trades.csv"), index=False)


if __name__ == "__main__":
    main()
