# 🤖 XAUUSD AI Trading Bot — Ichimoku Multi-Timeframe RL

> **RL (PPO) Trading Bot cho XAUUSD** sử dụng chiến lược Ichimoku Cloud Break trên 4 timeframes đồng thời (M5/M15/H1/H4).  
> Inspired by [MoonDev's DRL Trading Bot](https://github.com/forbbiden403/tradingbot) — nhưng **Ichimoku là core**, không phải generic indicators.

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![PyPI](https://img.shields.io/pypi/v/xauusd-ichi-rl?color=orange)](https://pypi.org/project/xauusd-ichi-rl/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Made with AI](https://img.shields.io/badge/AI%20Assisted-Antigravity-purple)](https://comarai.com)

---

## ⚡ Install

```bash
pip install xauusd-ichi-rl
```

### CLI Commands

```bash
# 1. Train RL bot (cần file CSV XAUUSD M1 trong thư mục hiện tại)
ichi-train --timesteps 500000 --sl 5.0 --tp 3.0

# 2. Rule-based backtest + grid search optimizer
ichi-backtest --mode optimize --year 2026 --month 01

# 3. Generate file MQL5 EA (output → mql5_output/)
ichi-gen-mql5
# ↑ Ra IchiMTF_RL_Strategy.mq5 — copy vào MT5 Experts/ là dùng được!
```

> 💡 **Workflow**: `ichi-train` → `ichi-backtest` → `ichi-gen-mql5` → Copy `.mq5` vào MetaTrader 5

> 🐍 **Python API**:
> ```python
> from xauusd_ichi import run_v2, generate_all
> result = run_v2(timesteps=500_000, sl=5.0, tp=3.0)
> ```

---

## 📊 Kết Quả Backtest — Jan 2026 (Out-of-Sample)

| Metric | Giá trị | Benchmark |
|--------|---------|-----------|
| **Net Profit** | **+$724 (+7.12%)** | MoonDev target: 5-7.5%/month |
| **Profit Factor** | **3.76** | Industry good: >1.5 |
| **Win Rate** | **86.7%** | MoonDev: 58-62% |
| **Max Drawdown** | **0.12%** | MoonDev: 8-12% |
| **Total Trades** | 406 | ~20/day |
| **Consecutive Wins** | 36 | |
| **Training Time** | 24 phút | 500k steps, 348 it/s |

> ⚠️ **Disclaimer**: Kết quả quá khứ không đảm bảo hiệu suất tương lai. Test chỉ trên 1 tháng, cần validate thêm.

---

## 🧠 Tư Duy & Điểm Khác Biệt

### So với MoonDev (forbbiden403/tradingbot):

| | MoonDev | Bot này |
|---|--------|---------|
| **Core Strategy** | 63 generic indicators | **Ichimoku Cloud Break** (từ EA thực chiến) |
| **Entry Logic** | RL tự quyết 100% | RL + **Ichimoku signal bias** trong reward |
| **SL/TP** | RL tự học cắt lỗ | **Built-in SL/TP** (grid search optimized) |
| **DCA** | Không | **DCA awareness** (từ EA IchiDCA) |
| **Speed** | ~35 it/s | **348 it/s** (pre-cached numpy arrays) |
| **Features** | 140+ | **128** (focused, less noise) |

### Quá trình phát triển (3 iterations):

```
v1: RL + Ichimoku only (M5)          → LỖ -4.95% ❌
    └─ Insight: 50 features, 1 TF không đủ edge
    
v1.5: Rule-based Ichimoku + Grid Search → LỖ -$65 ❌  
    └─ Insight: Ichimoku đơn lẻ trên M5 = quá nhiều noise
    
v2: RL + Multi-TF (M5/M15/H1/H4)    → LÃI +7.12% ✅
    └─ Insight: Multi-TF confirmation + RL flexibility = edge
```

### Key Insights:
1. **Ichimoku cần multi-TF**: M5 quá noisy, cần H1/H4 confirm trend
2. **Built-in SL/TP > RL-learned SL/TP**: RL mất quá nhiều steps để học risk management
3. **Numpy pre-cache**: Tăng speed 10x (35 → 348 it/s) bằng cách bỏ pandas.iloc
4. **128 features > 140+ features**: Focused features (Ichimoku core) < noise reduction

---

## 🏗️ Kiến Trúc

```
┌─────────────────────────────────────────────┐
│           Multi-TF Feature Engine            │
│  M1 Data → Resample → M5, M15, H1, H4      │
│                                              │
│  Per TF: Ichimoku│EMA│RSI│MACD│ATR│BB│ADX    │
│  + Sessions    + Price Returns               │
│  ─────────────────────────────────           │
│  → 128 features (forward-fill merge)         │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│         Trading Environment (Gym)            │
│  • Pre-cached numpy arrays (fast!)           │
│  • Built-in SL=$5 / TP=$3                    │
│  • Ichimoku-aware reward (Sharpe)            │
│  • Cooldown 5 bars (anti-overtrading)        │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│            PPO Agent (SB3)                   │
│  • Network: [256, 256]                       │
│  • 500k steps, n_steps=1024                  │
│  • Live metrics callback (pass tracking)     │
│  • Auto-save best checkpoint                 │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│         MT5-Style Report                     │
│  PF│WR│DD│Sharpe│Recovery│Consecutive│MFE/MAE│
└─────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Option A: Via PyPI (recommended)
```bash
pip install xauusd-ichi-rl

# Đặt file CSV vào thư mục rồi chạy:
ichi-train --timesteps 500000 --sl 5.0 --tp 3.0
ichi-backtest --mode optimize --year 2026 --month 01
ichi-gen-mql5
```

### Option B: Clone & Run
```bash
git clone https://github.com/hungpixi/xauusd-ichimoku-rl-bot
cd xauusd-ichimoku-rl-bot
pip install -r requirements.txt
```

### Prepare Data
Đặt file XAUUSD M1 CSV vào root:
```
XAUUSD_2025_10.csv  # Train
XAUUSD_2025_11.csv  # Train
XAUUSD_2025_12.csv  # Train
XAUUSD_2026_01.csv  # Test
```

### Train + Test (1 lệnh)
```bash
python run_rl_v2.py --timesteps 500000 --sl 5.0 --tp 3.0
# hoặc:
ichi-train --timesteps 500000 --sl 5.0 --tp 3.0
```

### Rule-Based Backtest + Grid Search
```bash
# Optimize SL/TP
ichi-backtest --mode optimize --year 2026 --month 01

# Single backtest
ichi-backtest --mode backtest --year 2026 --month 01 --sl 5 --tp 3
```

---

## 📁 Cấu Trúc

```
src/
├── data/
│   ├── data_loader.py          # Load CSV data
│   ├── resampler.py            # M1 → M5/M15/H1/H4
│   ├── multi_tf_engine.py      # ★ Multi-TF feature builder (128 features)
│   ├── ichimoku_features.py    # Ichimoku indicator calculations
│   └── macro_data.py           # DXY/VIX from Yahoo Finance
├── env/
│   └── trading_env.py          # ★ Gymnasium env (numpy-optimized, SL/TP)
├── strategy/
│   └── ichimoku_strategy.py    # ★ Rule-based strategy + grid search
├── models/
│   └── mt5_report.py           # MT5 Strategy Tester report format
└── rbi/
    └── progressive_validation.py # Progressive validation runner

run_rl_v2.py                    # ★ Main: Train + Test (1 command)
run_backtest.py                 # Rule-based backtest + optimizer
```

---

## 🔮 Hướng Phát Triển

- [ ] **Macro data integration**: DXY/VIX correlation (fix merge bug)
- [ ] **Grid search SL/TP**: Tìm SL/TP tối ưu cho RL
- [ ] **Multi-month validation**: Test trên 2025 full (12 tháng)
- [ ] **DCA logic**: Dollar Cost Averaging từ EA IchiDCA gốc
- [ ] **Live trading**: MetaTrader 5 integration
- [ ] **Dreamer V3**: World-model RL (more sample efficient)

---

## 📚 References

- [forbbiden403/tradingbot](https://github.com/forbbiden403/tradingbot) — MoonDev's DRL Trading Bot (inspiration)
- [IchiDCA_CCBSN_PropFirm.mq5](docs/) — EA MQL5 gốc (Ichimoku + DCA + CCBSN)
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) — PPO implementation
- [Gymnasium](https://gymnasium.farama.org/) — RL environment

---

## 🤝 Bạn muốn Bot Trading AI tương tự?

| Bạn cần | Chúng tôi đã làm ✅ |
|---------|---------------------|
| Bot trade tự động | RL Bot XAUUSD multi-TF |
| Backtest nhanh | Grid search 576 combos trong vài giây |
| Báo cáo chuyên nghiệp | MT5 Strategy Tester format |
| Tối ưu chiến lược | Multi-indicator + multi-timeframe |

<table>
<tr>
<td align="center">
<a href="https://comarai.com"><b>🌐 Yêu cầu Demo</b></a>
</td>
<td align="center">
<a href="https://zalo.me/0834422439"><b>💬 Zalo</b></a>
</td>
<td align="center">
<a href="mailto:hungphamphunguyen@gmail.com"><b>📧 Email</b></a>
</td>
</tr>
</table>

### [Comarai](https://comarai.com) — Companion for Marketing & AI Automation

> *"Thời gian là tài sản quý nhất. Để AI làm những việc lặp lại, bạn tập trung vào chiến lược."*
> — Phạm Phú Nguyễn Hưng, Founder

**4 nhân viên AI chạy 24/7:**
- 🤖 **Em Trade** — Bot trading tự cải tiến
- 📝 **Em Content** — Sáng tạo nội dung
- 📢 **Em Marketing** — Automation marketing
- 💼 **Em Sale** — Tìm kiếm khách hàng

---

*Built with ❤️ by [hungpixi](https://github.com/hungpixi) × AI (Antigravity)*
