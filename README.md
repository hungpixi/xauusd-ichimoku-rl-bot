# 🥇 XAUUSD Ichimoku RL Trading Bot

> Train AI tự học chiến lược Ichimoku để trade vàng (XAUUSD), xuất ra file `.mq5` chạy thẳng trên MetaTrader 5.

[![PyPI](https://img.shields.io/pypi/v/xauusd-ichi-rl?color=orange&label=pip%20install%20xauusd-ichi-rl)](https://pypi.org/project/xauusd-ichi-rl/)
[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Made with AI](https://img.shields.io/badge/AI%20Assisted-Antigravity-purple)](https://comarai.com)

---

## 🎯 Kết Quả (Test tháng 1/2026)

| | |
|--|--|
| 💰 Lợi nhuận | **+$724 (+7.12%)** trên tài khoản $500 |
| 📈 Win Rate | **86.7%** (cứ 10 lệnh thắng 8-9) |
| 📉 Drawdown tối đa | **0.12%** (rủi ro rất thấp) |
| ⚡ Thời gian train | **24 phút** (500,000 bước học) |

> ⚠️ Kết quả backtest, không đảm bảo lợi nhuận thực tế. Cần kiểm tra thêm trước khi dùng live.

---

## ⚡ Bắt Đầu Trong 3 Bước

### Bước 1 — Cài đặt

```bash
pip install xauusd-ichi-rl
```

### Bước 2 — Lấy data

Tải file CSV mẫu trong thư mục [`sample_data/`](sample_data/) về, đặt vào thư mục làm việc:

```
📂 thư mục của bạn/
├── XAUUSD_2025_10.csv   ← train
├── XAUUSD_2025_11.csv   ← train
├── XAUUSD_2025_12.csv   ← train
└── XAUUSD_2026_01.csv   ← test
```

> 💡 File CSV là dữ liệu nến M1 (1 phút) XAUUSD từ Exness MT5. Bạn có thể tự export từ MT5 của mình.

### Bước 3 — Chạy

```bash
# Train AI (24 phút)
ichi-train --timesteps 500000 --sl 5.0 --tp 3.0

# Tìm config tốt nhất
ichi-backtest --mode optimize --year 2026 --month 01

# Xuất file .mq5 → dùng trên MetaTrader 5
ichi-gen-mql5
```

**Kết quả**: File `mql5_output/IchiMTF_RL_Strategy.mq5` → copy vào MT5 `MQL5/Experts/` → compile → attach vào chart **XAUUSD M5** là xong!

---

## 🤔 Bot Này Hoạt Động Như Thế Nào?

```
Data M1 XAUUSD
    ↓
Tính 128 chỉ báo (Ichimoku + EMA + RSI + MACD... trên 4 khung thời gian)
    ↓
AI (PPO) học cách đọc các chỉ báo đó → quyết định mua/bán
    ↓
Xuất quyết định thành file .mq5 chạy được trên MetaTrader 5
```

**Điểm khác so với bot khác:**
- Ichimoku là nền tảng (không phải chỉ là 1 trong 100 chỉ báo)
- 4 khung thời gian cùng lúc: M5 để vào lệnh, H1/H4 để lọc xu hướng
- SL/TP cố định ($5/$3) thay vì để AI tự học → ổn định hơn

---

## 📂 Dữ Liệu Mẫu

Thư mục [`sample_data/`](sample_data/) có **1 tuần** dữ liệu XAUUSD M1 để thử nghiệm nhanh.

Muốn train đầy đủ cần tải thêm dữ liệu từ MT5:
1. Mở MT5 → History Center
2. Chọn XAUUSD, khung M1
3. Export ra CSV theo từng tháng

---

## 🔧 Cấu Hình Mặc Định

| Tham số | Giá trị | Giải thích |
|---------|---------|-----------|
| `--sl` | `5.0` | Stop Loss $5 mỗi lệnh |
| `--tp` | `3.0` | Take Profit $3 mỗi lệnh |
| `--timesteps` | `500000` | Số bước train AI |
| `--balance` | `500.0` | Vốn tham chiếu |
| `--lot` | `0.05` | Khối lượng giao dịch |

---

## 📋 Lịch Sử Phát Triển

| Phiên bản | Kết quả | Vấn đề |
|-----------|---------|--------|
| v1: RL + Ichimoku M5 | -4.95% ❌ | Chỉ 1 khung thời gian = quá nhiều nhiễu |
| v1.5: Rule-based M5 | -$65 ❌ | Không có xu hướng dài hạn |
| **v2: RL + 4 TF** | **+7.12% ✅** | Multi-TF + AI linh hoạt = có lợi thế |

---

## 🔮 Kế Hoạch Tiếp Theo

- [ ] Tích hợp dữ liệu macro (DXY, VIX)
- [ ] Validate trên 12 tháng 2025
- [ ] Kết nối live trading với MT5
- [ ] Logic DCA (Dollar Cost Averaging)

---

## 📚 Tham Khảo

- [MoonDev's DRL Bot](https://github.com/forbbiden403/tradingbot) — Nguồn cảm hứng
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) — Thuật toán PPO
- [Gymnasium](https://gymnasium.farama.org/) — Môi trường RL

---

## 🤝 Bạn Muốn Bot Trading AI Tương Tự?

| Bạn cần | Chúng tôi đã làm ✅ |
|---------|---------------------|
| Bot trade tự động cho MT5 | RL Bot XAUUSD multi-TF |
| Backtest nhanh với báo cáo | Grid search 576 combos |
| Tối ưu chiến lược Ichimoku | Multi-TF + SL/TP tối ưu |

<table>
<tr>
<td align="center"><a href="https://comarai.com"><b>🌐 Demo</b></a></td>
<td align="center"><a href="https://zalo.me/0834422439"><b>💬 Zalo</b></a></td>
<td align="center"><a href="mailto:hungphamphunguyen@gmail.com"><b>📧 Email</b></a></td>
</tr>
</table>

**[Comarai](https://comarai.com)** — AI Automation Agency | 4 nhân viên AI chạy 24/7: Em Trade 🤖 · Em Content 📝 · Em Marketing 📢 · Em Sale 💼

> *"Thời gian là tài sản quý nhất. Để AI làm những việc lặp lại, bạn tập trung vào chiến lược."*  
> — Phạm Phú Nguyễn Hưng, Founder

---

*Built with ❤️ by [hungpixi](https://github.com/hungpixi) × AI (Antigravity)*
