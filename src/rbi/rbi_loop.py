"""
RBI Self-Improving Loop: Research → Backtest → Implement.
Logic MoonDev nhưng dùng rule-based analysis thay vì LLM API.
Bot tự phân tích kết quả backtest, đề xuất điều chỉnh, re-train, lặp lại.
"""

import sys
import json
import logging
import copy
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.train_ppo import prepare_data, create_env, train
from src.models.evaluate import evaluate_model, walk_forward_test

logger = logging.getLogger(__name__)


# ============================================================
# RESEARCH AGENT (Rule-based)
# ============================================================

def analyze_performance(metrics: dict) -> dict:
    """
    Phân tích performance metrics và đề xuất cải tiến.
    Đây là "Research Agent" rule-based, thay thế LLM.
    """
    diagnosis = {
        "issues": [],
        "strengths": [],
        "suggestions": [],
    }

    total_return = metrics.get("total_return", 0)
    sharpe = metrics.get("sharpe_ratio", 0)
    max_dd = metrics.get("max_drawdown", 0)
    win_rate = metrics.get("win_rate", 0)
    total_trades = metrics.get("total_trades", 0)
    profit_factor = metrics.get("profit_factor", 0)

    # === Phân tích strengths ===
    if total_return > 5:
        diagnosis["strengths"].append(f"Sinh lời tốt: {total_return:.1f}%")
    if sharpe > 1.0:
        diagnosis["strengths"].append(f"Risk-adjusted return tốt (Sharpe={sharpe:.2f})")
    if win_rate > 55:
        diagnosis["strengths"].append(f"Win rate cao: {win_rate:.0f}%")
    if max_dd < 10:
        diagnosis["strengths"].append(f"Drawdown thấp: {max_dd:.1f}%")

    # === Phân tích issues ===
    if total_return < 0:
        diagnosis["issues"].append("LOSS: Model đang lỗ tổng thể")
    if max_dd > 20:
        diagnosis["issues"].append(f"RISK: Drawdown quá cao ({max_dd:.1f}%)")
    if win_rate < 40:
        diagnosis["issues"].append(f"LOW_WINRATE: Win rate thấp ({win_rate:.0f}%)")
    if total_trades < 10:
        diagnosis["issues"].append("TOO_FEW_TRADES: Giao dịch quá ít")
    if total_trades > 5000:
        diagnosis["issues"].append("OVERTRADING: Giao dịch quá nhiều")
    if profit_factor < 1.0 and total_trades > 0:
        diagnosis["issues"].append(f"UNPROFITABLE: Profit factor < 1 ({profit_factor:.2f})")
    if sharpe < 0.5:
        diagnosis["issues"].append(f"LOW_SHARPE: Risk-adjusted return kém ({sharpe:.2f})")

    # === Đề xuất cải tiến ===
    suggestions = diagnosis["suggestions"]

    if "LOSS" in str(diagnosis["issues"]):
        suggestions.append({"type": "reward_mode", "value": "sortino", "reason": "Chuyển sang Sortino reward để penalize downside mạnh hơn"})
        suggestions.append({"type": "learning_rate", "value": 1e-4, "reason": "Giảm LR để ổn định training"})

    if "RISK" in str(diagnosis["issues"]):
        suggestions.append({"type": "spread", "value": 0.50, "reason": "Tăng spread giả lập để bot học cẩn thận hơn"})
        suggestions.append({"type": "gamma", "value": 0.95, "reason": "Giảm gamma → bot ưu tiên reward ngắn hạn, giảm risk"})

    if "LOW_WINRATE" in str(diagnosis["issues"]):
        suggestions.append({"type": "n_epochs", "value": 15, "reason": "Tăng epochs per update để học kỹ hơn"})
        suggestions.append({"type": "ent_coef", "value": 0.005, "reason": "Giảm entropy coefficient → ít random hơn"})

    if "OVERTRADING" in str(diagnosis["issues"]):
        suggestions.append({"type": "ent_coef", "value": 0.02, "reason": "Tăng entropy → giảm overtrading"})
        suggestions.append({"type": "spread", "value": 0.40, "reason": "Tăng transaction cost → bot trade ít hơn"})

    if "TOO_FEW_TRADES" in str(diagnosis["issues"]):
        suggestions.append({"type": "ent_coef", "value": 0.02, "reason": "Tăng entropy → encourage exploration"})
        suggestions.append({"type": "reward_mode", "value": "simple", "reason": "Đổi sang simple reward để dễ học"})

    if not diagnosis["issues"]:
        # Model đang tốt → tinh chỉnh nhẹ
        suggestions.append({"type": "total_timesteps", "value": 1_000_000, "reason": "Model tốt, train thêm steps"})
        suggestions.append({"type": "clip_range", "value": 0.15, "reason": "Giảm clip range → cập nhật policy nhẹ nhàng hơn"})

    return diagnosis


def generate_strategy_proposal(
    current_config: dict,
    diagnosis: dict,
    iteration: int,
) -> dict:
    """
    Tạo config mới dựa trên diagnosis.
    Mỗi iteration chỉ thay đổi 1-2 params để track được impact.
    """
    new_config = copy.deepcopy(current_config)
    changes = []

    suggestions = diagnosis.get("suggestions", [])
    if not suggestions:
        return new_config

    # Chỉ apply tối đa 2 suggestions per iteration
    for suggestion in suggestions[:2]:
        param = suggestion["type"]
        value = suggestion["value"]
        reason = suggestion["reason"]

        old_value = new_config.get(param)
        if old_value != value:
            new_config[param] = value
            changes.append({
                "param": param,
                "old": old_value,
                "new": value,
                "reason": reason,
            })

    logger.info(f"\n🔬 Iteration {iteration} - Strategy Proposal:")
    for c in changes:
        logger.info(f"  📝 {c['param']}: {c['old']} → {c['new']} ({c['reason']})")

    new_config["_changes"] = changes
    return new_config


# ============================================================
# RBI LOOP ORCHESTRATOR
# ============================================================

class RBILoop:
    """
    Research → Backtest → Implement loop.
    Tự cải tiến bot qua mỗi iteration.
    """

    def __init__(
        self,
        data_dir: Path,
        output_dir: Path,
        base_config: dict = None,
        max_iterations: int = 5,
        convergence_threshold: float = 2.0,  # Dừng nếu improvement < X%
    ):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

        # Default base config
        self.base_config = base_config or {
            "train_year": 2024,
            "train_month": None,
            "timeframe": "M5",
            "total_timesteps": 200_000,
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
        }

        self.history = []
        self.best_sharpe = -np.inf
        self.best_model_path = None
        self.best_config = None

    def run(self, eval_year: int = 2025, eval_month: str = "01") -> dict:
        """
        Chạy full RBI loop.
        
        Returns:
            Dict chứa history của tất cả iterations
        """
        current_config = copy.deepcopy(self.base_config)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.output_dir / f"rbi_run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\n{'='*60}")
        logger.info(f"🔄 RBI SELF-IMPROVING LOOP")
        logger.info(f"   Max iterations: {self.max_iterations}")
        logger.info(f"   Train: {current_config['train_year']}")
        logger.info(f"   Eval:  {eval_year}-{eval_month}")
        logger.info(f"{'='*60}\n")

        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"\n{'─'*40}")
            logger.info(f"📍 ITERATION {iteration}/{self.max_iterations}")
            logger.info(f"{'─'*40}")

            iter_dir = run_dir / f"iter_{iteration:02d}"
            iter_dir.mkdir(exist_ok=True)

            # === IMPLEMENT: Train model ===
            logger.info(f"\n🏋️ [IMPLEMENT] Training model...")
            try:
                model, model_path = train(
                    data_dir=self.data_dir,
                    output_dir=iter_dir,
                    config=current_config,
                )
            except Exception as e:
                logger.error(f"❌ Training failed: {e}")
                break

            # === BACKTEST: Evaluate on unseen data ===
            logger.info(f"\n📊 [BACKTEST] Evaluating on {eval_year}-{eval_month}...")
            try:
                results, trade_log = evaluate_model(
                    model_path=str(model_path),
                    data_dir=self.data_dir,
                    year=eval_year,
                    month=eval_month,
                    timeframe=current_config["timeframe"],
                    initial_balance=current_config["initial_balance"],
                    lot_size=current_config["lot_size"],
                    spread=current_config["spread"],
                )
                metrics = results["metrics"]
            except Exception as e:
                logger.error(f"❌ Evaluation failed: {e}")
                break

            # Track best model
            current_sharpe = metrics.get("sharpe_ratio", -np.inf)
            if current_sharpe > self.best_sharpe:
                self.best_sharpe = current_sharpe
                self.best_model_path = str(model_path)
                self.best_config = copy.deepcopy(current_config)
                logger.info(f"🏆 New best model! Sharpe: {current_sharpe:.4f}")

            # Save iteration results
            iter_result = {
                "iteration": iteration,
                "config": {k: v for k, v in current_config.items() if k != "_changes"},
                "changes": current_config.get("_changes", []),
                "metrics": metrics,
                "model_path": str(model_path),
            }
            self.history.append(iter_result)

            # Save to file
            with open(iter_dir / "results.json", "w") as f:
                json.dump(iter_result, f, indent=2, default=str)

            if not trade_log.empty:
                trade_log.to_csv(iter_dir / "trades.csv", index=False)

            # === RESEARCH: Analyze and propose improvements ===
            logger.info(f"\n🔬 [RESEARCH] Analyzing results...")
            diagnosis = analyze_performance(metrics)

            logger.info(f"  Strengths: {diagnosis['strengths']}")
            logger.info(f"  Issues:    {diagnosis['issues']}")

            # Save diagnosis
            with open(iter_dir / "diagnosis.json", "w") as f:
                json.dump(diagnosis, f, indent=2)

            # Check convergence
            if iteration >= 2:
                prev_sharpe = self.history[-2]["metrics"].get("sharpe_ratio", 0)
                improvement = abs(current_sharpe - prev_sharpe)
                if improvement < self.convergence_threshold * 0.01 and current_sharpe > 0.5:
                    logger.info(f"\n✅ Converged! Improvement ({improvement:.4f}) < threshold")
                    break

            # Generate new config
            if iteration < self.max_iterations:
                current_config = generate_strategy_proposal(
                    current_config, diagnosis, iteration
                )

        # === Summary ===
        self._print_summary()
        self._save_summary(run_dir)

        return {
            "history": self.history,
            "best_model": self.best_model_path,
            "best_sharpe": self.best_sharpe,
            "best_config": self.best_config,
        }

    def _print_summary(self):
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 RBI LOOP SUMMARY")
        logger.info(f"{'='*60}")

        for h in self.history:
            m = h["metrics"]
            logger.info(
                f"  Iter {h['iteration']}: "
                f"Return={m['total_return']:.1f}% | "
                f"Sharpe={m['sharpe_ratio']:.3f} | "
                f"DD={m['max_drawdown']:.1f}% | "
                f"WR={m['win_rate']:.0f}% | "
                f"Trades={m['total_trades']}"
            )

        logger.info(f"\n🏆 Best model: {self.best_model_path}")
        logger.info(f"🏆 Best Sharpe: {self.best_sharpe:.4f}")
        logger.info(f"{'='*60}")

    def _save_summary(self, run_dir: Path):
        summary = {
            "total_iterations": len(self.history),
            "best_model": self.best_model_path,
            "best_sharpe": self.best_sharpe,
            "best_config": self.best_config,
            "history": self.history,
        }
        with open(run_dir / "rbi_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="RBI Self-Improving Loop")
    parser.add_argument("--data", type=str, default=str(PROJECT_ROOT))
    parser.add_argument("--output", type=str, default=str(PROJECT_ROOT / "models"))
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--train-year", type=int, default=2024)
    parser.add_argument("--eval-year", type=int, default=2025)
    parser.add_argument("--eval-month", type=str, default="01")
    parser.add_argument("--timeframe", type=str, default="M5")
    parser.add_argument("--timesteps", type=int, default=200_000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    rbi = RBILoop(
        data_dir=Path(args.data),
        output_dir=Path(args.output),
        base_config={
            "train_year": args.train_year,
            "timeframe": args.timeframe,
            "total_timesteps": args.timesteps,
        },
        max_iterations=args.iterations,
    )

    results = rbi.run(eval_year=args.eval_year, eval_month=args.eval_month)

    print(f"\n✅ Done! Best model: {results['best_model']}")
    print(f"✅ Best Sharpe: {results['best_sharpe']:.4f}")


if __name__ == "__main__":
    main()
