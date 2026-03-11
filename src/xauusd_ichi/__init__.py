"""
xauusd-ichi-rl: XAUUSD Ichimoku Multi-Timeframe RL Trading Bot
==============================================================
Train PPO bot, backtest rule-based, và generate MQL5 EA files.

Author: Phạm Phú Nguyễn Hưng — comarai.com
GitHub: https://github.com/hungpixi/xauusd-ichimoku-rl-bot
"""

__version__ = "2.1.0"
__author__ = "Phạm Phú Nguyễn Hưng"
__email__ = "hungphamphunguyen@gmail.com"
__url__ = "https://comarai.com"

# Public API
from xauusd_ichi._runner import run_v2
from xauusd_ichi._generator import generate_all

__all__ = [
    "run_v2",
    "generate_all",
    "__version__",
]
