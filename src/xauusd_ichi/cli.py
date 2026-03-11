"""
CLI entrypoints cho xauusd-ichi-rl package.
Các commands: ichi-train, ichi-backtest, ichi-gen-mql5
"""

import argparse
import sys
from pathlib import Path


def train():
    """CLI: ichi-train — Train PPO RL bot.

    Ví dụ:
        ichi-train --timesteps 500000 --sl 5.0 --tp 3.0
        ichi-train --timesteps 200000 --balance 500 --lot 0.05
    """
    parser = argparse.ArgumentParser(
        prog="ichi-train",
        description="""
Train PPO RL Bot cho XAUUSD — Ichimoku Multi-Timeframe.
Config mặc định: $500 / 200x leverage / 0.05 lot.
Train data: Q4 2025 (Oct-Nov-Dec) → Test: Jan 2026.

Output: models/rl_v2_BEST.zip (best checkpoint)
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--timesteps", type=int, default=500_000,
                        help="Số bước training (default: 500000)")
    parser.add_argument("--sl", type=float, default=5.0,
                        help="Stop Loss tính bằng $ (default: 5.0)")
    parser.add_argument("--tp", type=float, default=3.0,
                        help="Take Profit tính bằng $ (default: 3.0)")
    parser.add_argument("--balance", type=float, default=500.0,
                        help="Balance ban đầu (default: 500.0)")
    parser.add_argument("--lot", type=float, default=0.05,
                        help="Lot size (default: 0.05)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Thư mục chứa file CSV XAUUSD_YYYY_MM.csv (default: thư mục hiện tại)")

    args = parser.parse_args()

    # Resolve data_dir
    data_dir = Path(args.data_dir) if args.data_dir else Path.cwd()

    # Check data exists
    required = [
        data_dir / "XAUUSD_2025_10.csv",
        data_dir / "XAUUSD_2025_11.csv",
        data_dir / "XAUUSD_2025_12.csv",
        data_dir / "XAUUSD_2026_01.csv",
    ]
    missing = [f for f in required if not f.exists()]
    if missing:
        print(f"❌ Thiếu file CSV:")
        for f in missing:
            print(f"   {f}")
        print(f"\n💡 Đặt các file XAUUSD M1 CSV vào: {data_dir}")
        print("   Cần: XAUUSD_2025_10.csv, XAUUSD_2025_11.csv, XAUUSD_2025_12.csv, XAUUSD_2026_01.csv")
        sys.exit(1)

    print(f"🚀 ichi-train | {args.timesteps:,} steps | SL=${args.sl} TP=${args.tp}")
    print(f"   Balance: ${args.balance} | Lot: {args.lot} | Data: {data_dir}")

    # Import here to avoid slow startup when just doing --help
    try:
        # Thêm data_dir vào sys.path để import src/
        sys.path.insert(0, str(data_dir))
        from run_rl_v2 import run_v2
        run_v2(
            data_dir=data_dir,
            timesteps=args.timesteps,
            sl=args.sl,
            tp=args.tp,
            initial_balance=args.balance,
            lot_size=args.lot,
        )
    except ImportError:
        # Fallback: dùng xauusd_ichi._runner
        from xauusd_ichi._runner import run_v2
        run_v2(
            data_dir=data_dir,
            timesteps=args.timesteps,
            sl=args.sl,
            tp=args.tp,
            initial_balance=args.balance,
            lot_size=args.lot,
        )


def backtest():
    """CLI: ichi-backtest — Chạy rule-based backtest / grid search.

    Ví dụ:
        ichi-backtest --mode optimize --year 2026 --month 01
        ichi-backtest --mode backtest --year 2026 --month 01 --sl 5 --tp 3
    """
    parser = argparse.ArgumentParser(
        prog="ichi-backtest",
        description="""
Rule-based Ichimoku Backtest + Grid Search Optimizer.
Optimize SL/TP để tìm config tốt nhất, output vào models/top3_configs.json.

Output: models/top3_configs.json (để gen MQL5 với ichi-gen-mql5)
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", choices=["backtest", "optimize"], default="optimize",
                        help="Mode: backtest (1 run) hoặc optimize (grid search, default)")
    parser.add_argument("--year", type=int, default=2026, help="Năm test (default: 2026)")
    parser.add_argument("--month", type=str, default="01", help="Tháng test 2 chữ số (default: 01)")
    parser.add_argument("--sl", type=float, default=5.0, help="Stop Loss $ (chỉ dùng khi --mode backtest)")
    parser.add_argument("--tp", type=float, default=3.0, help="Take Profit $ (chỉ dùng khi --mode backtest)")
    parser.add_argument("--balance", type=float, default=500.0, help="Balance ban đầu (default: 500.0)")
    parser.add_argument("--lot", type=float, default=0.05, help="Lot size (default: 0.05)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Thư mục chứa file CSV (default: thư mục hiện tại)")

    args = parser.parse_args()
    data_dir = Path(args.data_dir) if args.data_dir else Path.cwd()

    print(f"📊 ichi-backtest | mode={args.mode} | {args.year}-{args.month} | data: {data_dir}")

    sys.path.insert(0, str(data_dir))
    try:
        # Import run_backtest từ project directory
        from run_backtest import main as backtest_main
        # Override sys.argv để truyền args
        sys.argv = [
            "run_backtest.py",
            f"--mode={args.mode}",
            f"--year={args.year}",
            f"--month={args.month}",
            f"--sl={args.sl}",
            f"--tp={args.tp}",
            f"--balance={args.balance}",
            f"--lot={args.lot}",
        ]
        backtest_main()
    except (ImportError, AttributeError):
        # Fallback nếu không tìm được run_backtest.py
        print("❌ Không tìm thấy run_backtest.py trong thư mục hiện tại.")
        print(f"💡 Clone repo và chạy từ trong thư mục dự án:")
        print(f"   git clone https://github.com/hungpixi/xauusd-ichimoku-rl-bot")
        print(f"   cd xauusd-ichimoku-rl-bot")
        print(f"   ichi-backtest --mode optimize --year 2026 --month 01")
        sys.exit(1)


def gen_mql5():
    """CLI: ichi-gen-mql5 — Generate file MQL5 EA từ optimization results.

    Ví dụ:
        ichi-gen-mql5                           # Dùng models/top3_configs.json hiện có
        ichi-gen-mql5 --config my_config.json   # Dùng config tùy chỉnh
        ichi-gen-mql5 --output ./my_eas/        # Output vào thư mục khác

    Output:
        mql5_output/IchiRule_Top1.mq5
        mql5_output/IchiRule_Top2.mq5
        mql5_output/IchiRule_Top3.mq5
        mql5_output/IchiMTF_RL_Strategy.mq5  ← File MQL5 chính (RL-based)
    """
    parser = argparse.ArgumentParser(
        prog="ichi-gen-mql5",
        description="""
Generate MQL5 EA files từ optimization results.

Sau khi chạy ichi-backtest, dùng lệnh này để tạo file .mq5 để dùng trên MetaTrader 5.
File IchiMTF_RL_Strategy.mq5 là EA chính (Multi-TF, v2.1 đã tối ưu).
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path tới top3_configs.json (default: models/top3_configs.json)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: mql5_output/)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Project directory (default: thư mục hiện tại)")

    args = parser.parse_args()
    data_dir = Path(args.data_dir) if args.data_dir else Path.cwd()

    config_path = Path(args.config) if args.config else data_dir / "models" / "top3_configs.json"
    output_dir = Path(args.output) if args.output else data_dir / "mql5_output"

    if not config_path.exists():
        print(f"❌ Không tìm thấy {config_path}")
        print(f"💡 Chạy optimizer trước:")
        print(f"   ichi-backtest --mode optimize --year 2026 --month 01")
        print(f"\n   Hoặc dùng file IchiMTF_RL_Strategy.mq5 có sẵn trong mql5_output/")
        sys.exit(1)

    import json
    with open(config_path, encoding="utf-8") as f:
        top3 = json.load(f)

    sys.path.insert(0, str(data_dir))
    try:
        from generate_mql5 import generate_all
        from pathlib import Path as _Path
        generate_all(top3)
    except ImportError:
        try:
            from xauusd_ichi._generator import generate_all
            generate_all(top3, output_dir=output_dir)
        except ImportError:
            print("❌ Không import được generate_mql5. Clone repo và chạy từ trong thư mục dự án.")
            sys.exit(1)

    print(f"\n✅ File MQL5 đã được tạo tại: {output_dir}/")
    print(f"📂 Dùng file nào?")
    print(f"   IchiMTF_RL_Strategy.mq5  ← RL-based v2.1 (KẾT QUẢ TỐT NHẤT)")
    print(f"   IchiRule_Top1.mq5        ← Rule-based Top 1")
    print(f"   IchiRule_Top2.mq5        ← Rule-based Top 2")
    print(f"   IchiRule_Top3.mq5        ← Rule-based Top 3")
    print(f"\n📌 Copy file .mq5 vào: MetaTrader5/MQL5/Experts/")
    print(f"   Compile bằng MetaEditor → Attach vào chart XAUUSD M5")
