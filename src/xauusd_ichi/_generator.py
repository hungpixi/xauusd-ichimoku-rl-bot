"""
_generator.py — Thin wrapper cho generate_mql5.generate_all()
Dùng khi import package: from xauusd_ichi import generate_all
"""

from pathlib import Path
import sys


def generate_all(top3_configs: list, output_dir: Path = None):
    """
    Generate MQL5 EA files từ top3 configs.

    Args:
        top3_configs: List of config dicts (từ models/top3_configs.json)
        output_dir: Output directory (default: mql5_output/)

    Example:
        import json
        from pathlib import Path
        from xauusd_ichi import generate_all

        with open("models/top3_configs.json") as f:
            configs = json.load(f)

        generate_all(configs, output_dir=Path("./my_eas"))
    """
    if output_dir is None:
        output_dir = Path.cwd() / "mql5_output"

    output_dir.mkdir(exist_ok=True)

    # Try project-local generate_mql5
    sys.path.insert(0, str(Path.cwd()))
    try:
        from generate_mql5 import generate_ea, generate_readme, OUTPUT_DIR
        for cfg in top3_configs:
            generate_ea(cfg, output_dir)
        generate_readme(top3_configs, output_dir)
        print(f"✅ {len(top3_configs)} EAs generated in: {output_dir}/")
    except ImportError:
        # Fallback: inline minimal generator
        _generate_all_inline(top3_configs, output_dir)


def _generate_all_inline(top3_configs: list, output_dir: Path):
    """Minimal inline generator khi không có generate_mql5.py."""
    print(f"\n{'='*60}")
    print(f"  Generating MQL5 EAs → {output_dir}")
    print(f"{'='*60}")

    for cfg in top3_configs:
        rank = cfg.get("rank", 1)
        params = cfg.get("params", {})
        ea_name = f"IchiRule_Top{rank}"
        out = output_dir / f"{ea_name}.mq5"
        print(f"  ✅ NOTE: Run from project directory to get full {ea_name}.mq5")
        print(f"     Or clone: git clone https://github.com/hungpixi/xauusd-ichimoku-rl-bot")

    print(f"\n📌 IchiMTF_RL_Strategy.mq5 (RL-based v2.1) is already in mql5_output/")
    print(f"   Copy it directly to MT5 Experts/ folder")
