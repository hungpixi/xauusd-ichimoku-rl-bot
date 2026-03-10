"""
Progressive Validation: 1 tháng → 3 tháng → 1 năm.
Logic: Train trên subset, test trên target. Nếu lời thì mở rộng.
Ưu tiên Ichimoku strategy theo EA IchiDCA_CCBSN.
"""

import sys
import json
import logging
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_loader import load_csv, load_year_data
from src.data.resampler import resample_ohlcv
from src.data.ichimoku_features import compute_ichimoku_features, normalize_features
from src.env.trading_env import XAUUSDTradingEnv
from src.models.mt5_report import compute_mt5_report, print_mt5_report

logger = logging.getLogger(__name__)


class LiveMetricsCallback(BaseCallback):
    """In live metrics mỗi N steps: PnL, Win Rate, Trades, Profit Factor, Speed."""

    def __init__(self, print_freq: int = 10000, verbose: int = 0):
        super().__init__(verbose)
        self.print_freq = print_freq
        self._start_time = None
        self._best_return = -999
        self._best_step = 0

    def _on_training_start(self):
        self._start_time = time.time()

    def _on_step(self) -> bool:
        if self.n_calls % self.print_freq != 0:
            return True

        elapsed = time.time() - self._start_time
        speed = self.n_calls / (elapsed + 0.01)

        infos = self.locals.get("infos", [])
        if not infos:
            return True

        info = infos[0]
        ret = info.get("total_return", 0)
        wr = info.get("win_rate", 0)
        trades = info.get("total_trades", 0)
        dd = info.get("max_drawdown", 0)
        pnl = info.get("total_pnl", 0)

        # Track best
        is_best = ""
        if ret > self._best_return:
            self._best_return = ret
            self._best_step = self.n_calls
            is_best = " 🏆 NEW BEST"

        print(
            f"  [{self.n_calls:>7,}/{self.num_timesteps:,}] "
            f"{speed:>5.0f} it/s | "
            f"PnL=${pnl:>8.2f} | Ret={ret:>6.2f}% | "
            f"WR={wr:>5.1f}% | DD={dd:>5.1f}% | "
            f"Trades={trades:>4}{is_best}"
        )
        return True

    def _on_training_end(self):
        elapsed = time.time() - self._start_time
        speed = self.num_timesteps / (elapsed + 0.01)
        print(f"\n  ⏱️  Done: {elapsed:.0f}s ({speed:.0f} it/s) | Best return: {self._best_return:.2f}% at step {self._best_step:,}")


def prepare_ichimoku_data(data_dir: Path, year: int, month: str = None, timeframe: str = "M5"):
    """Load data và tính Ichimoku features."""
    if month:
        csv_file = data_dir / f"XAUUSD_{year}_{month}.csv"
        df_m1 = load_csv(csv_file)
    else:
        df_m1 = load_year_data(data_dir, year)

    if timeframe != "M1":
        df = resample_ohlcv(df_m1, timeframe)
    else:
        df = df_m1

    df = compute_ichimoku_features(df)
    df = normalize_features(df)

    feature_cols = [c for c in df.columns if c not in ["open", "high", "low", "close", "volume"]]
    return df, feature_cols


def train_and_evaluate(
    data_dir: Path,
    output_dir: Path,
    train_year: int,
    train_months: list,
    test_year: int,
    test_months: list,
    timeframe: str = "M5",
    timesteps: int = 200_000,
    config: dict = None,
) -> dict:
    """
    Train trên train_months, evaluate trên test_months.
    Returns performance metrics.
    """
    if config is None:
        config = {}

    # === Prepare train data ===
    train_dfs = []
    for m in train_months:
        csv_file = data_dir / f"XAUUSD_{train_year}_{m}.csv"
        if csv_file.exists():
            df_m1 = load_csv(csv_file)
            train_dfs.append(df_m1)

    if not train_dfs:
        raise FileNotFoundError(f"No train data for {train_year} months {train_months}")

    train_m1 = pd.concat(train_dfs).sort_index()
    train_m1 = train_m1[~train_m1.index.duplicated(keep="first")]

    if timeframe != "M1":
        train_df = resample_ohlcv(train_m1, timeframe)
    else:
        train_df = train_m1

    train_df = compute_ichimoku_features(train_df)
    train_df = normalize_features(train_df)
    feature_cols = [c for c in train_df.columns if c not in ["open", "high", "low", "close", "volume"]]

    # === Train (optimized: n_steps=1024, smaller network) ===
    train_env = Monitor(XAUUSDTradingEnv(
        df=train_df,
        feature_columns=feature_cols,
        initial_balance=config.get("initial_balance", 10000),
        lot_size=config.get("lot_size", 0.01),
        spread=config.get("spread", 0.30),
        reward_mode=config.get("reward_mode", "sharpe"),
        random_start=True,
    ))

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=config.get("learning_rate", 3e-4),
        n_steps=config.get("n_steps", 1024),
        batch_size=config.get("batch_size", 64),
        n_epochs=config.get("n_epochs", 5),
        gamma=config.get("gamma", 0.99),
        clip_range=config.get("clip_range", 0.2),
        ent_coef=config.get("ent_coef", 0.01),
        policy_kwargs={"net_arch": config.get("policy_layers", [128, 128])},
        verbose=0,
        device="auto",
    )

    live_cb = LiveMetricsCallback(print_freq=5000)

    logger.info(f"🏋️ Training {timesteps:,} steps (Ichimoku PPO)...")
    model.learn(total_timesteps=timesteps, callback=live_cb, progress_bar=True)

    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = output_dir / f"ichi_ppo_{timestamp}.zip"
    model.save(str(model_path))
    logger.info(f"💾 Saved: {model_path}")

    # === Evaluate trên từng test month ===
    all_results = []
    for m in test_months:
        csv_file = data_dir / f"XAUUSD_{test_year}_{m}.csv"
        if not csv_file.exists():
            logger.warning(f"⚠️ Skip {csv_file.name}")
            continue

        try:
            test_m1 = load_csv(csv_file)
            if timeframe != "M1":
                test_df = resample_ohlcv(test_m1, timeframe)
            else:
                test_df = test_m1

            test_df = compute_ichimoku_features(test_df)
            test_df = normalize_features(test_df)

            test_feature_cols = [c for c in test_df.columns if c not in ["open", "high", "low", "close", "volume"]]

            test_env = XAUUSDTradingEnv(
                df=test_df,
                feature_columns=test_feature_cols,
                initial_balance=config.get("initial_balance", 10000),
                lot_size=config.get("lot_size", 0.01),
                spread=config.get("spread", 0.30),
                reward_mode="sharpe",
                random_start=False,
            )

            obs, _ = test_env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = test_env.step(action)
                done = terminated or truncated

            summary = test_env.get_performance_summary()
            summary["month"] = f"{test_year}-{m}"
            all_results.append(summary)

            # === MT5 Style Report ===
            trade_log = test_env.get_trade_log()
            mt5 = compute_mt5_report(
                trade_log=trade_log,
                equity_history=test_env.equity_history,
                initial_balance=config.get("initial_balance", 10000),
                total_bars=len(test_df),
            )
            print_mt5_report(mt5, title=f"Strategy Tester Report - XAUUSD {test_year}-{m} ({timeframe})")

        except Exception as e:
            logger.error(f"❌ {test_year}-{m}: {e}")
            import traceback
            traceback.print_exc()

    return {
        "model_path": str(model_path),
        "train_months": train_months,
        "test_months": test_months,
        "results": all_results,
    }


def progressive_validation(
    data_dir: Path,
    output_dir: Path,
    timeframe: str = "M5",
    timesteps_phase1: int = 200_000,
    timesteps_phase2: int = 300_000,
    timesteps_phase3: int = 500_000,
    config: dict = None,
):
    """
    Progressive Validation:
    Phase 1: Train 2025 Q4 → Test Jan 2026 (1 tháng)
    Phase 2: Nếu lời → Train 2025 H2 → Test Q1 2026 (3 tháng)
    Phase 3: Nếu lời → Train 2024 full → Test 2025 full (1 năm)
    """
    if config is None:
        config = {}

    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    # ============================================================
    # PHASE 1: 1 THÁNG - Train 2025 Q4 → Test Jan 2026
    # ============================================================
    logger.info(f"\n{'='*60}")
    logger.info(f"🔵 PHASE 1: 1 THÁNG")
    logger.info(f"   Train: 2025 Oct-Dec → Test: 2026 Jan")
    logger.info(f"{'='*60}")

    phase1 = train_and_evaluate(
        data_dir=data_dir,
        output_dir=output_dir,
        train_year=2025, train_months=["10", "11", "12"],
        test_year=2026, test_months=["01"],
        timeframe=timeframe,
        timesteps=timesteps_phase1,
        config=config,
    )
    results["phase1"] = phase1

    # Check profit
    if phase1["results"]:
        p1_return = phase1["results"][0]["total_return"]
        p1_sharpe = phase1["results"][0]["sharpe_ratio"]
        logger.info(f"\n🔵 Phase 1 Result: Return={p1_return:.2f}%, Sharpe={p1_sharpe:.3f}")

        if p1_return <= 0:
            logger.warning(f"⚠️ Phase 1 LỖ ({p1_return:.1f}%). Cần điều chỉnh trước khi mở rộng.")
            logger.info(f"💡 Gợi ý: thử đổi reward_mode, tăng timesteps, hoặc điều chỉnh spread.")

            # Lưu kết quả và dừng
            _save_results(output_dir, results)
            return results

    # ============================================================
    # PHASE 2: 3 THÁNG - Train 2025 H1 → Test 2025 Q3+Q4
    # ============================================================
    logger.info(f"\n{'='*60}")
    logger.info(f"🟢 PHASE 2: 3 THÁNG (Phase 1 lời!)")
    logger.info(f"   Train: 2025 Jan-Jun → Test: 2025 Jul-Sep")
    logger.info(f"{'='*60}")

    phase2 = train_and_evaluate(
        data_dir=data_dir,
        output_dir=output_dir,
        train_year=2025, train_months=["01", "02", "03", "04", "05", "06"],
        test_year=2025, test_months=["07", "08", "09"],
        timeframe=timeframe,
        timesteps=timesteps_phase2,
        config=config,
    )
    results["phase2"] = phase2

    if phase2["results"]:
        p2_returns = [r["total_return"] for r in phase2["results"]]
        p2_avg = np.mean(p2_returns)
        p2_profitable = sum(1 for r in p2_returns if r > 0)

        logger.info(f"\n🟢 Phase 2 Result: Avg={p2_avg:.2f}%, Profitable months={p2_profitable}/{len(p2_returns)}")

        if p2_avg <= 0 or p2_profitable < 2:
            logger.warning(f"⚠️ Phase 2 không đủ ổn. Dừng ở phase 2.")
            _save_results(output_dir, results)
            return results

    # ============================================================
    # PHASE 3: 1 NĂM - Train 2024 → Test 2025 full
    # ============================================================
    logger.info(f"\n{'='*60}")
    logger.info(f"🟡 PHASE 3: 1 NĂM (Phase 2 lời!)")
    logger.info(f"   Train: 2024 full → Test: 2025 full (12 tháng)")
    logger.info(f"{'='*60}")

    all_months = [f"{m:02d}" for m in range(1, 13)]

    phase3 = train_and_evaluate(
        data_dir=data_dir,
        output_dir=output_dir,
        train_year=2024, train_months=all_months,
        test_year=2025, test_months=all_months,
        timeframe=timeframe,
        timesteps=timesteps_phase3,
        config=config,
    )
    results["phase3"] = phase3

    if phase3["results"]:
        p3_returns = [r["total_return"] for r in phase3["results"]]
        p3_avg = np.mean(p3_returns)
        p3_total = sum(p3_returns)
        p3_profitable = sum(1 for r in p3_returns if r > 0)
        p3_worst_dd = max(r["max_drawdown"] for r in phase3["results"])

        logger.info(f"\n{'='*60}")
        logger.info(f"🏆 PHASE 3 FINAL RESULT:")
        logger.info(f"   Total Return (2025): {p3_total:.2f}%")
        logger.info(f"   Avg Monthly:         {p3_avg:.2f}%")
        logger.info(f"   Profitable months:   {p3_profitable}/12")
        logger.info(f"   Worst DD:            {p3_worst_dd:.2f}%")
        logger.info(f"   Best model:          {phase3['model_path']}")
        logger.info(f"{'='*60}")

    _save_results(output_dir, results)
    return results


def _save_results(output_dir: Path, results: dict):
    """Lưu kết quả progressive validation."""
    with open(output_dir / "progressive_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"💾 Results saved to {output_dir / 'progressive_results.json'}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Progressive Validation - Ichimoku XAUUSD Bot")
    parser.add_argument("--data", type=str, default=str(PROJECT_ROOT))
    parser.add_argument("--output", type=str, default=str(PROJECT_ROOT / "models"))
    parser.add_argument("--timeframe", type=str, default="M5")
    parser.add_argument("--timesteps", type=int, default=200_000, help="Phase 1 timesteps")
    parser.add_argument("--spread", type=float, default=0.30)
    parser.add_argument("--reward", type=str, default="sharpe")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    config = {
        "spread": args.spread,
        "reward_mode": args.reward,
    }

    results = progressive_validation(
        data_dir=Path(args.data),
        output_dir=Path(args.output),
        timeframe=args.timeframe,
        timesteps_phase1=args.timesteps,
        timesteps_phase2=int(args.timesteps * 1.5),
        timesteps_phase3=int(args.timesteps * 2.5),
        config=config,
    )


if __name__ == "__main__":
    main()
