import json
from pathlib import Path

top3 = Path("models/top3_configs.json")
print("top3_configs.json exists:", top3.exists())

if top3.exists():
    d = json.loads(top3.read_text(encoding="utf-8"))
    print(f"Loaded {len(d)} configs")
    for cfg in d:
        t = cfg["train_metrics"]
        p = cfg["params"]
        r = cfg["rank"]
        print(f"\n  Config #{r}: return={t['return_pct']}% | PF={t['profit_factor']} | WR={t['win_rate']}% | DD={t['max_dd_pct']}% | trades={t['total_trades']}")
        print(f"  SL={p['stop_loss']} TP={p['take_profit']} trailing={p['use_trailing_stop']} ema={p['use_ema_filter']} tk={p['use_tk_cross']} cooldown={p['cooldown_bars']}")
        if "multi_month" in cfg and "summary" in cfg["multi_month"]:
            s = cfg["multi_month"]["summary"]
            blow = s.get("blow_up_months", [])
            print(f"  Multi-month: avg={s['avg_monthly_return']}% | win={s['profitable_months']}/{s['total_months_tested']} | blow={blow}")
else:
    print("Optimizer still running or not done yet")
    opt_out = Path("optimizer_out.txt")
    if opt_out.exists():
        content = opt_out.read_text(encoding="utf-8", errors="replace")
        print("Last 500 chars of optimizer output:")
        print(content[-500:])
