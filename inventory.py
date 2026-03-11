import json
from pathlib import Path

models_dir = Path("models")
mql5_dir = Path("mql5_output")
src_dir = Path("src")

print("=== MODELS ===")
for f in sorted(models_dir.iterdir()):
    size = f.stat().st_size
    print(f"  {f.name}: {size//1024}KB")

print()
print("=== MQL5 OUTPUT ===")
for f in sorted(mql5_dir.iterdir()):
    print(f"  {f.name}: {f.stat().st_size//1024}KB")

print()
print("=== SRC STRUCTURE ===")
for f in src_dir.rglob("*.py"):
    print(f"  {str(f.relative_to(src_dir))}: {f.stat().st_size//1024}KB")

bm = models_dir / "rl_v2_BEST_meta.json"
if bm.exists():
    d = json.loads(bm.read_text())
    print()
    print("=== CURRENT TRAINING BEST ===")
    for k, v in d.items():
        print(f"  {k}: {v}")

tp = models_dir / "training_passes.json"
if tp.exists():
    d = json.loads(tp.read_text())
    print()
    print(f"Training passes: {d.get('total_passes', 0)}")
    print(f"Best step: {d.get('best_step', 0)}")
    print(f"Best score: {d.get('best_composite_score', 0)}")
    print(f"Training time: {d.get('training_time_sec', 0)}s")
