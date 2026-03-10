"""
PPO Training Script cho XAUUSD Trading Bot.
Dùng Stable-Baselines3 PPO, lấy cảm hứng từ forbbiden403/tradingbot.
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
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_loader import load_year_data, load_csv
from src.data.resampler import resample_ohlcv
from src.data.feature_engine import compute_all_features, normalize_features
from src.env.trading_env import XAUUSDTradingEnv

logger = logging.getLogger(__name__)


class TradingMetricsCallback(BaseCallback):
    """Callback log trading metrics vào TensorBoard."""

    def __init__(self, eval_freq: int = 5000, verbose: int = 0):
        super().__init__(verbose)
        self.eval_freq = eval_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Lấy info từ env
            infos = self.locals.get("infos", [])
            if infos:
                info = infos[0]
                for key in ["total_return", "win_rate", "max_drawdown", "total_trades"]:
                    if key in info:
                        self.logger.record(f"trading/{key}", info[key])

        return True


def prepare_data(
    data_dir: Path,
    year: int,
    timeframe: str = "M5",
    use_monthly: str = None,
) -> tuple:
    """
    Load và prepare data cho training.
    
    Args:
        data_dir: Thư mục chứa CSV files
        year: Năm data
        timeframe: Timeframe base cho trading
        use_monthly: Nếu set, chỉ dùng 1 tháng (vd "01" cho tháng 1)
    
    Returns:
        (df_features, feature_columns)
    """
    if use_monthly:
        csv_file = data_dir / f"XAUUSD_{year}_{use_monthly}.csv"
        df_m1 = load_csv(csv_file)
    else:
        df_m1 = load_year_data(data_dir, year)

    # Resample sang timeframe mong muốn
    if timeframe != "M1":
        df = resample_ohlcv(df_m1, timeframe)
    else:
        df = df_m1

    # Tính features
    df = compute_all_features(df)
    df = normalize_features(df)

    feature_columns = [
        c for c in df.columns
        if c not in ["open", "high", "low", "close", "volume"]
    ]

    logger.info(f"📊 Prepared: {len(df)} candles, {len(feature_columns)} features")
    return df, feature_columns


def create_env(
    df: pd.DataFrame,
    feature_columns: list,
    config: dict = None,
) -> XAUUSDTradingEnv:
    """Tạo trading environment."""
    if config is None:
        config = {}

    env = XAUUSDTradingEnv(
        df=df,
        feature_columns=feature_columns,
        initial_balance=config.get("initial_balance", 10000),
        lot_size=config.get("lot_size", 0.01),
        spread=config.get("spread", 0.30),
        reward_mode=config.get("reward_mode", "sharpe"),
        max_steps=config.get("max_steps", None),
        random_start=config.get("random_start", True),
    )
    return Monitor(env)


def train(
    data_dir: Path,
    output_dir: Path,
    config: dict = None,
):
    """
    Main training function.
    
    Args:
        data_dir: Thư mục chứa CSV
        output_dir: Thư mục lưu model
        config: Training config dict
    """
    if config is None:
        config = {}

    # Default config
    cfg = {
        "train_year": 2024,
        "train_month": None,        # None = cả năm, "01" = tháng 1
        "timeframe": "M5",
        "total_timesteps": 500_000,
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "policy_layers": [256, 256],
        "initial_balance": 10000,
        "lot_size": 0.01,
        "spread": 0.30,
        "reward_mode": "sharpe",
        "max_steps": 50000,
        "random_start": True,
        **config,
    }

    logger.info(f"🚀 Training config:\n{json.dumps(cfg, indent=2, default=str)}")

    # Prepare data
    df, feature_cols = prepare_data(
        data_dir, cfg["train_year"], cfg["timeframe"], cfg["train_month"]
    )

    # Create env
    env = create_env(df, feature_cols, cfg)

    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=cfg["learning_rate"],
        n_steps=cfg["n_steps"],
        batch_size=cfg["batch_size"],
        n_epochs=cfg["n_epochs"],
        gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"],
        clip_range=cfg["clip_range"],
        ent_coef=cfg["ent_coef"],
        policy_kwargs={"net_arch": cfg["policy_layers"]},
        verbose=1,
        tensorboard_log=str(output_dir / "tb_logs"),
        device="auto",
    )

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=max(cfg["total_timesteps"] // 10, 10000),
        save_path=str(output_dir / "checkpoints"),
        name_prefix="ppo_xauusd",
    )

    metrics_cb = TradingMetricsCallback(eval_freq=5000)

    # Train!
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"🏋️ Training {cfg['total_timesteps']:,} timesteps...")

    model.learn(
        total_timesteps=cfg["total_timesteps"],
        callback=[checkpoint_cb, metrics_cb],
        progress_bar=True,
    )

    # Save final model
    model_path = output_dir / f"ppo_xauusd_{timestamp}.zip"
    model.save(str(model_path))
    logger.info(f"💾 Model saved: {model_path}")

    # Save config
    config_path = output_dir / f"config_{timestamp}.json"
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2, default=str)

    return model, model_path


def main():
    parser = argparse.ArgumentParser(description="Train PPO XAUUSD Bot")
    parser.add_argument("--data", type=str, default=str(PROJECT_ROOT), help="Data directory")
    parser.add_argument("--output", type=str, default=str(PROJECT_ROOT / "models"), help="Output directory")
    parser.add_argument("--timesteps", type=int, default=500_000, help="Total training timesteps")
    parser.add_argument("--timeframe", type=str, default="M5", help="Trading timeframe")
    parser.add_argument("--year", type=int, default=2024, help="Training year")
    parser.add_argument("--month", type=str, default=None, help="Month (01-12) or None for full year")
    parser.add_argument("--reward", type=str, default="sharpe", choices=["simple", "sharpe", "sortino"])
    parser.add_argument("--balance", type=float, default=10000, help="Initial balance")
    parser.add_argument("--lot", type=float, default=0.01, help="Lot size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    config = {
        "total_timesteps": args.timesteps,
        "timeframe": args.timeframe,
        "train_year": args.year,
        "train_month": args.month,
        "reward_mode": args.reward,
        "initial_balance": args.balance,
        "lot_size": args.lot,
        "learning_rate": args.lr,
    }

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    train(
        data_dir=Path(args.data),
        output_dir=output_dir,
        config=config,
    )


if __name__ == "__main__":
    main()
