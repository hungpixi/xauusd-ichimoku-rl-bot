# 📂 Sample Data

Dữ liệu mẫu XAUUSD M1 (~1 tuần giao dịch) để thử nghiệm bot.

## File

| File | Nội dung | Dòng |
|------|---------|------|
| `XAUUSD_2026_01_sample.csv` | Tuần đầu tháng 1/2026 | 7,000 |

## Format

```
Date, Time, Open, High, Low, Close, Volume
2026.01.02, 01:00, 4331.24, 4332.19, 4325.26, 4328.66, 204
...
```

## Cách dùng với bot

```bash
# Đặt file vào thư mục làm việc với đúng tên:
cp sample_data/XAUUSD_2026_01_sample.csv ./XAUUSD_2026_01.csv

# Chạy backtest nhanh (chỉ cần 1 file):
ichi-backtest --mode backtest --year 2026 --month 01 --sl 5 --tp 3
```

> 💡 Để train đầy đủ, cần dữ liệu 3-4 tháng. Export từ MT5 History Center → XAUUSD → 1 phút → CSV.

---
*Source: Exness MT5 XAUUSD M1 | [hungpixi/xauusd-ichimoku-rl-bot](https://github.com/hungpixi/xauusd-ichimoku-rl-bot)*
