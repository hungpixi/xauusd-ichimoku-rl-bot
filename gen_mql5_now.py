"""
Gen MQL5 ngay từ best known configs (không cần đợi optimizer).
Dựa trên:
 - rl_v2_ichi_mtf: +7.12% Jan2026, WR=86.7%, PF=3.76, DD=0.12%
 - old optimization_results.json top results
 - Ichimoku Cloud Break strategy principles
"""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path = [str(PROJECT_ROOT)] + __import__('sys').path

# Best known configs từ testing + domain knowledge
TOP3_MANUAL = [
    {
        "rank": 1,
        "params": {
            "stop_loss": 5.0,
            "take_profit": 3.0,
            "use_trailing_stop": True,
            "trailing_start": 1.5,
            "trailing_step": 0.5,
            "cooldown_bars": 5,
            "use_tk_cross": False,
            "use_ema_filter": True,
        },
        "lot_size": 0.05,
        "initial_balance": 500,
        "train_metrics": {
            "net_profit": 35.6,
            "return_pct": 7.12,
            "profit_factor": 3.76,
            "win_rate": 86.7,
            "max_dd_pct": 0.12,
            "total_trades": 406,
            "recovery_factor": 59.3,
            "composite_score": 0.45,
        },
        "multi_month": {
            "summary": {
                "avg_monthly_return": 7.12,
                "profitable_months": 1,
                "total_months_tested": 1,
                "avg_max_dd": 0.12,
                "blow_up_months": [],
            }
        },
        "_note": "Strategy: Cloud Break + EMA34/89 filter | SL=5 TP=3 trailing | From rl_v2_ichi_mtf backtest"
    },
    {
        "rank": 2,
        "params": {
            "stop_loss": 8.0,
            "take_profit": 5.0,
            "use_trailing_stop": True,
            "trailing_start": 2.0,
            "trailing_step": 0.5,
            "cooldown_bars": 5,
            "use_tk_cross": True,
            "use_ema_filter": True,
        },
        "lot_size": 0.05,
        "initial_balance": 500,
        "train_metrics": {
            "net_profit": 0,
            "return_pct": 0,
            "profit_factor": 0,
            "win_rate": 0,
            "max_dd_pct": 0,
            "total_trades": 0,
            "recovery_factor": 0,
            "composite_score": 0,
        },
        "multi_month": {
            "summary": {
                "avg_monthly_return": 0,
                "profitable_months": 0,
                "total_months_tested": 0,
                "avg_max_dd": 0,
                "blow_up_months": [],
            }
        },
        "_note": "Strategy: Cloud Break + TK Cross + EMA filter | Higher SL/TP for bigger moves"
    },
    {
        "rank": 3,
        "params": {
            "stop_loss": 5.0,
            "take_profit": 5.0,
            "use_trailing_stop": False,
            "trailing_start": 1.5,
            "trailing_step": 0.5,
            "cooldown_bars": 3,
            "use_tk_cross": True,
            "use_ema_filter": False,
        },
        "lot_size": 0.05,
        "initial_balance": 500,
        "train_metrics": {
            "net_profit": 0,
            "return_pct": 0,
            "profit_factor": 0,
            "win_rate": 0,
            "max_dd_pct": 0,
            "total_trades": 0,
            "recovery_factor": 0,
            "composite_score": 0,
        },
        "multi_month": {
            "summary": {
                "avg_monthly_return": 0,
                "profitable_months": 0,
                "total_months_tested": 0,
                "avg_max_dd": 0,
                "blow_up_months": [],
            }
        },
        "_note": "Strategy: Cloud Break + TK Cross | No EMA filter, aggressive entry | Symmetric SL=TP"
    },
]

import sys
sys.path.insert(0, str(PROJECT_ROOT))
from generate_mql5 import generate_all

generate_all(TOP3_MANUAL)
print("\nDone! Check mql5_output/ folder")
