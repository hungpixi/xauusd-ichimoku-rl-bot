"""
_runner.py — Thin wrapper cho run_rl_v2.run_v2()
Dùng khi import package: from xauusd_ichi import run_v2
"""

from pathlib import Path
import sys


def run_v2(
    data_dir: Path = None,
    timesteps: int = 500_000,
    sl: float = 5.0,
    tp: float = 3.0,
    initial_balance: float = 500.0,
    lot_size: float = 0.05,
):
    """
    Train PPO RL Bot và test trên Jan 2026.

    Args:
        data_dir: Thư mục chứa XAUUSD_YYYY_MM.csv files
        timesteps: Số training steps (default: 500_000)
        sl: Stop Loss in $ (default: 5.0)
        tp: Take Profit in $ (default: 3.0)
        initial_balance: Account balance (default: 500.0)
        lot_size: Lot size (default: 0.05)

    Returns:
        dict: Performance summary (return%, win_rate, max_dd, trades)

    Example:
        from xauusd_ichi import run_v2
        result = run_v2(data_dir=Path("."), timesteps=500_000, sl=5.0, tp=3.0)
    """
    if data_dir is None:
        data_dir = Path.cwd()

    # Try to import from parent project (khi dùng từ trong repo)
    sys.path.insert(0, str(data_dir))
    try:
        from run_rl_v2 import run_v2 as _run_v2
        return _run_v2(
            data_dir=data_dir,
            timesteps=timesteps,
            sl=sl,
            tp=tp,
            initial_balance=initial_balance,
            lot_size=lot_size,
        )
    except ImportError as e:
        raise ImportError(
            "Không tìm thấy run_rl_v2.py. "
            "Clone repo và chạy từ trong thư mục dự án:\n"
            "  git clone https://github.com/hungpixi/xauusd-ichimoku-rl-bot\n"
            f"Original error: {e}"
        )
