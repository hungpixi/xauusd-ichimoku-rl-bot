import json
from pathlib import Path

models = Path("models")

tp = models / "training_passes.json"
bm = models / "rl_v2_BEST_meta.json"

print("=== RL v2 TRAINING PROGRESS ===")

if bm.exists():
    d = json.loads(bm.read_text())
    print("BEST checkpoint so far:")
    for k, v in d.items():
        print(f"  {k}: {v}")
else:
    print("No BEST checkpoint yet")

print()

if tp.exists():
    d = json.loads(tp.read_text())
    passes = d.get("all_passes", [])
    total = d.get("total_passes", 0)
    print(f"Training passes: {total}")
    print(f"Best return: {d.get('best_return', 0):.2f}%")
    print(f"Best step: {d.get('best_step', 0):,}")
    print()
    if passes:
        print(f"{'Pass':>5} | {'Step':>8} | {'Speed':>7} | {'PnL':>10} | {'Ret%':>7} | {'WR':>6} | {'DD':>6} | {'Score':>8}")
        print("-" * 75)
        for p in passes[-15:]:
            print(f"{p['pass']:>5} | {p['step']:>8,} | {p['speed']:>6.0f}/s | ${p['pnl']:>9.2f} | {p['return']:>6.2f}% | {p['win_rate']:>5.1f}% | {p['max_dd']:>5.1f}% | {p.get('composite',0):>8.5f}")
else:
    print("training_passes.json not found - training still initializing or not started yet")
