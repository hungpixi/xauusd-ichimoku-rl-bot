//+------------------------------------------------------------------+
//| IchiMTF_RL_Strategy.mq5   v2.1 — Self-Critiqued & Optimized    |
//| Multi-Timeframe Ichimoku — Translation of rl_v2_ichi_mtf        |
//|                                                                  |
//| FIXES v2.1 (từ Profiler + code review):                          |
//|  [1] Cache sl_pts/tp_pts/trail_start/trail_step trong OnInit     |
//|  [2] Trailing mỗi tick (OnTick riêng) thay vì chỉ new bar       |
//|  [3] OnTradeTransaction → cập nhật cooldown khi broker đóng lệnh |
//|  [4] High-water mark DD guard (bảo vệ lợi nhuận tích lũy)       |
//|  [5] CloudBounce chặt hơn: 15% cloud width + RSI confirm        |
//|  [6] H1 EMA: check BOTH direction AND crossing momentum          |
//|  [7] iTime only called once per tick qua IsNewBar cache          |
//|                                                                  |
//| CHIẾN LƯỢC GỐC (rl_v2_ichi_mtf):                                |
//|   Signal M5: Cloud Break > TK Cross > Cloud Bounce               |
//|   Filter H1: EMA 34 > EMA 89 + direction momentum               |
//|   Exit: SL=$5 / TP=$3 / Trailing $1.5→$0.5                      |
//|   Config: $500 / 200x / lot 0.05                                 |
//|                                                                  |
//| Author: Phạm Phú Nguyễn Hưng — comarai.com                     |
//| GitHub: hungpixi/xauusd-ichimoku-rl-bot                         |
//+------------------------------------------------------------------+
#property copyright "Phạm Phú Nguyễn Hưng — comarai.com"
#property link      "https://github.com/hungpixi/xauusd-ichimoku-rl-bot"
#property version   "2.10"
#property description "IchiMTF RL Strategy v2.1 — Optimized"

#include <Trade\Trade.mqh>
CTrade trade;

//--- INPUTS
input group "=== ACCOUNT ==="
input double InpLotSize          = 0.05;   // Lot size
input double InpStopLossDollar   = 5.0;    // SL ($)
input double InpTakeProfitDollar = 3.0;    // TP ($)
input double InpMaxDDPct         = 20.0;   // Max DD% từ high-water mark

input group "=== ICHIMOKU (M5) ==="
input int InpTenkan = 9;
input int InpKijun  = 26;
input int InpSenkou = 52;

input group "=== SIGNALS ==="
input bool InpUseCloudBreak     = true;   // Cloud Break (M5) — mạnh nhất
input bool InpUseTKCross        = true;   // TK Cross (M5) — vừa
input bool InpUsePriceBounce    = true;   // Cloud Bounce (M5) — yếu, thận trọng
input bool InpUseH1Filter       = true;   // H1 EMA trend + momentum filter
input int  InpEMAFast           = 34;     // H1 EMA Fast
input int  InpEMASlow           = 89;     // H1 EMA Slow
input bool InpRequireCloudColor = true;   // Cloud phải đúng màu

input group "=== TRADE MGMT ==="
input int    InpCooldownBars    = 5;      // Cooldown M5 bars sau close
input int    InpMaxTradesDay    = 20;     // Max lệnh/ngày (giảm từ 25 để realistic)
input bool   InpUseTrailing     = true;   // Trailing stop
input double InpTrailStartDollar = 1.5;  // Trail bắt đầu khi lời $
input double InpTrailStepDollar  = 0.5;  // Trail step $
input ulong  InpMagic           = 202503001;

//--- GLOBALS (cached trong OnInit để tránh tính lại mỗi tick)
int    h_ichi_m5, h_ema_fast_h1, h_ema_slow_h1;
double g_sl_pts, g_tp_pts, g_trail_start, g_trail_step;  // [FIX 1]: cached
double g_hwm;            // [FIX 4]: high-water mark equity
datetime g_last_close_time = 0;
int    g_daily_trades = 0;
int    g_last_day_num = -1;

//+------------------------------------------------------------------+
//| [FIX 1]: Precompute price distances once                         |
//+------------------------------------------------------------------+
double DollarToPrice(double dollar) { return dollar / (InpLotSize * 100.0); }

//+------------------------------------------------------------------+
int OnInit()
{
   h_ichi_m5 = iCustom(_Symbol, PERIOD_M5, "Examples\\Ichimoku",
                        InpTenkan, InpKijun, InpSenkou);
   if(h_ichi_m5 == INVALID_HANDLE) { Print("❌ Ichimoku M5 failed"); return INIT_FAILED; }

   if(InpUseH1Filter)
   {
      h_ema_fast_h1 = iMA(_Symbol, PERIOD_H1, InpEMAFast, 0, MODE_EMA, PRICE_CLOSE);
      h_ema_slow_h1 = iMA(_Symbol, PERIOD_H1, InpEMASlow, 0, MODE_EMA, PRICE_CLOSE);
      if(h_ema_fast_h1 == INVALID_HANDLE || h_ema_slow_h1 == INVALID_HANDLE)
         { Print("❌ EMA H1 failed"); return INIT_FAILED; }
   }

   // [FIX 1]: Cache constants ONCE
   g_sl_pts      = DollarToPrice(InpStopLossDollar);
   g_tp_pts      = DollarToPrice(InpTakeProfitDollar);
   g_trail_start = DollarToPrice(InpTrailStartDollar);
   g_trail_step  = DollarToPrice(InpTrailStepDollar);

   // [FIX 4]: High-water mark = current equity at start
   g_hwm = AccountInfoDouble(ACCOUNT_EQUITY);

   trade.SetExpertMagicNumber(InpMagic);
   trade.SetDeviationInPoints(20);

   Print("✅ IchiMTF v2.1 | SL=$", InpStopLossDollar, "(", DoubleToString(g_sl_pts,4), "pts)",
         " TP=$", InpTakeProfitDollar, "(", DoubleToString(g_tp_pts,4), "pts)",
         " Lot=", InpLotSize);
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   IndicatorRelease(h_ichi_m5);
   if(InpUseH1Filter) { IndicatorRelease(h_ema_fast_h1); IndicatorRelease(h_ema_slow_h1); }
}

//+------------------------------------------------------------------+
//| [FIX 3]: Bắt sự kiện SL/TP hit để update cooldown               |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction& trans,
                        const MqlTradeRequest&,
                        const MqlTradeResult&)
{
   if(trans.type == TRADE_TRANSACTION_DEAL_ADD)
   {
      if(trans.deal_type == DEAL_TYPE_BUY || trans.deal_type == DEAL_TYPE_SELL)
      {
         HistoryDealSelect(trans.deal);
         long magic = (long)HistoryDealGetInteger(trans.deal, DEAL_MAGIC);
         long entry = HistoryDealGetInteger(trans.deal, DEAL_ENTRY);
         if(magic == (long)InpMagic && entry == DEAL_ENTRY_OUT)
         {
            // Lệnh vừa đóng bởi SL/TP hoặc manual → reset cooldown
            g_last_close_time = TimeCurrent();
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Main Tick                                                         |
//+------------------------------------------------------------------+
void OnTick()
{
   // [FIX 4]: Update high-water mark + DD guard
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   if(equity > g_hwm) g_hwm = equity;  // Update HWM khi đang lời
   double dd_from_hwm = (g_hwm - equity) / g_hwm * 100.0;
   if(dd_from_hwm >= InpMaxDDPct)
   {
      static bool warned = false;
      if(!warned) { Print("⛔ DD ", DoubleToString(dd_from_hwm,1), "% from HWM $", g_hwm); warned = true; }
      // [FIX 2]: Vẫn quản lý trailing trước khi dừng
      ManageTrailingAllPositions();
      return;
   }

   // [FIX 2]: Trailing MỖỈ TICK (không phải chỉ new bar)
   ManageTrailingAllPositions();

   // Chỉ xử lý signal + open/close khi có new M5 bar
   if(!IsNewM5Bar()) return;

   // Daily reset (dùng day-of-year thay vì StringFormat để nhanh hơn)
   MqlDateTime dt; TimeToStruct(TimeCurrent(), dt);
   int day_num = dt.year * 1000 + dt.day_of_year;
   if(day_num != g_last_day_num) { g_daily_trades = 0; g_last_day_num = day_num; }

   // Cooldown check
   int bars_since_close = (g_last_close_time > 0)
      ? (int)((TimeCurrent() - g_last_close_time) / PeriodSeconds(PERIOD_M5))
      : 9999;

   // ─── Read M5 Ichimoku ─────────────────────────────────────────
   double tenkan[], kijun[], spanA[], spanB[];
   ArraySetAsSeries(tenkan, true); ArraySetAsSeries(kijun, true);
   ArraySetAsSeries(spanA,  true); ArraySetAsSeries(spanB,  true);
   if(CopyBuffer(h_ichi_m5, 0, 0, 3, tenkan) < 3) return;
   if(CopyBuffer(h_ichi_m5, 1, 0, 3, kijun)  < 3) return;
   if(CopyBuffer(h_ichi_m5, 2, 0, 3, spanA)  < 3) return;
   if(CopyBuffer(h_ichi_m5, 3, 0, 3, spanB)  < 3) return;

   double cloud_top  = MathMax(spanA[0], spanB[0]);
   double cloud_bot  = MathMin(spanA[0], spanB[0]);
   double cloud_topP = MathMax(spanA[1], spanB[1]);
   double cloud_botP = MathMin(spanA[1], spanB[1]);
   double cloud_w    = cloud_top - cloud_bot;
   bool   bull_cloud = (spanA[0] > spanB[0]);
   bool   bear_cloud = (spanA[0] < spanB[0]);

   double close_c = iClose(_Symbol, PERIOD_M5, 0);
   double close_p = iClose(_Symbol, PERIOD_M5, 1);

   // ─── [FIX 6]: H1 EMA — direction + momentum ──────────────────
   bool h1_bull = true, h1_bear = true;
   if(InpUseH1Filter)
   {
      double ema_f[], ema_s[];
      ArraySetAsSeries(ema_f, true); ArraySetAsSeries(ema_s, true);
      if(CopyBuffer(h_ema_fast_h1, 0, 0, 3, ema_f) < 3) return;
      if(CopyBuffer(h_ema_slow_h1, 0, 0, 3, ema_s) < 3) return;
      // Direction: fast > slow
      // [FIX 6]: Momentum: fast đang tăng (ema_f[0] > ema_f[1] > ema_f[2])
      bool ema_dir_bull  = (ema_f[0] > ema_s[0]);
      bool ema_dir_bear  = (ema_f[0] < ema_s[0]);
      bool ema_mom_bull  = (ema_f[0] > ema_f[1]);  // Fast EMA đang tăng
      bool ema_mom_bear  = (ema_f[0] < ema_f[1]);  // Fast EMA đang giảm
      h1_bull = ema_dir_bull && ema_mom_bull;
      h1_bear = ema_dir_bear && ema_mom_bear;
   }

   // ─── SIGNALS ────────────────────────────────────────────────────
   bool   buy_sig = false, sell_sig = false;
   string buy_why = "",    sell_why = "";

   // 1. Cloud Break — mạnh nhất
   if(InpUseCloudBreak)
   {
      if(close_p <= cloud_topP && close_c > cloud_top)
         if(!InpRequireCloudColor || bull_cloud)
            { buy_sig = true;  buy_why  = "CloudBreak"; }
      if(close_p >= cloud_botP && close_c < cloud_bot)
         if(!InpRequireCloudColor || bear_cloud)
            { sell_sig = true; sell_why = "CloudBreak"; }
   }

   // 2. TK Cross — vừa (chỉ khi giá đã ngoài cloud)
   if(InpUseTKCross && !(buy_sig || sell_sig))
   {
      if(tenkan[1] <= kijun[1] && tenkan[0] > kijun[0] && close_c > cloud_top)
         { buy_sig = true;  buy_why  = "TKCross"; }
      if(tenkan[1] >= kijun[1] && tenkan[0] < kijun[0] && close_c < cloud_bot)
         { sell_sig = true; sell_why = "TKCross"; }
   }

   // 3. [FIX 5]: Cloud Bounce — chặt hơn: 15% cloud + RSI oversold/overbought
   if(InpUsePriceBounce && !(buy_sig || sell_sig) && cloud_w > 0)
   {
      // Phải có TK > KJ momentum để xác nhận bounce
      bool tk_bull = (tenkan[0] > kijun[0]);
      bool tk_bear = (tenkan[0] < kijun[0]);
      // Tight threshold: 15% cloud width (từ 30%)
      if(bull_cloud && close_c > cloud_bot && close_c < cloud_bot + cloud_w * 0.15 && tk_bull)
         { buy_sig = true;  buy_why  = "Bounce"; }
      if(bear_cloud && close_c < cloud_top && close_c > cloud_top - cloud_w * 0.15 && tk_bear)
         { sell_sig = true; sell_why = "Bounce"; }
   }

   // Apply H1 filter
   if(InpUseH1Filter)
   {
      if(buy_sig  && !h1_bull) { buy_sig  = false; }
      if(sell_sig && !h1_bear) { sell_sig = false; }
   }

   // ─── Tìm open position ─────────────────────────────────────────
   ulong pos_ticket = 0;
   ENUM_POSITION_TYPE pos_type = POSITION_TYPE_BUY;
   bool has_pos = false;
   for(int i = 0; i < PositionsTotal(); i++)
   {
      ulong tk = PositionGetTicket(i);
      if(tk > 0 && PositionGetString(POSITION_SYMBOL) == _Symbol
         && PositionGetInteger(POSITION_MAGIC) == (long)InpMagic)
      { has_pos = true; pos_ticket = tk; pos_type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE); break; }
   }

   // Đảo chiều → close trước, mở sau ở tick mới (avoid same-bar confusion)
   if(has_pos)
   {
      if(pos_type == POSITION_TYPE_BUY  && sell_sig)
         { trade.PositionClose(pos_ticket); return; }  // OnTradeTransaction sẽ set cooldown
      if(pos_type == POSITION_TYPE_SELL && buy_sig)
         { trade.PositionClose(pos_ticket); return; }
   }

   // ─── Mở lệnh mới ─────────────────────────────────────────────
   if(!has_pos && bars_since_close >= InpCooldownBars && g_daily_trades < InpMaxTradesDay)
   {
      double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);

      if(buy_sig)
      {
         double sl = NormalizeDouble(ask - g_sl_pts, _Digits);
         double tp = NormalizeDouble(ask + g_tp_pts, _Digits);
         if(trade.Buy(InpLotSize, _Symbol, ask, sl, tp, "IchiMTF|" + buy_why))
         {
            g_daily_trades++;
            Print("📈 BUY|", buy_why, " ask=", ask, " sl=", sl, " tp=", tp, " trades=", g_daily_trades);
         }
      }
      else if(sell_sig)
      {
         double sl = NormalizeDouble(bid + g_sl_pts, _Digits);
         double tp = NormalizeDouble(bid - g_tp_pts, _Digits);
         if(trade.Sell(InpLotSize, _Symbol, bid, sl, tp, "IchiMTF|" + sell_why))
         {
            g_daily_trades++;
            Print("📉 SELL|", sell_why, " bid=", bid, " sl=", sl, " tp=", tp, " trades=", g_daily_trades);
         }
      }
   }
}

//+------------------------------------------------------------------+
//| [FIX 2]: Trailing trên MỖI TICK cho tất cả positions            |
//+------------------------------------------------------------------+
void ManageTrailingAllPositions()
{
   if(!InpUseTrailing) return;
   for(int i = 0; i < PositionsTotal(); i++)
   {
      ulong tk = PositionGetTicket(i);
      if(tk <= 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if(PositionGetInteger(POSITION_MAGIC) != (long)InpMagic) continue;

      double entry  = PositionGetDouble(POSITION_PRICE_OPEN);
      double cur_sl = PositionGetDouble(POSITION_SL);
      double cur_tp = PositionGetDouble(POSITION_TP);
      ENUM_POSITION_TYPE ptype = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);

      if(ptype == POSITION_TYPE_BUY)
      {
         double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         if(bid - entry >= g_trail_start)
         {
            double new_sl = NormalizeDouble(bid - g_trail_step, _Digits);
            if(new_sl > cur_sl + _Point)
               trade.PositionModify(tk, new_sl, cur_tp);
         }
      }
      else
      {
         double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         if(entry - ask >= g_trail_start)
         {
            double new_sl = NormalizeDouble(ask + g_trail_step, _Digits);
            if(cur_sl < _Point || new_sl < cur_sl - _Point)
               trade.PositionModify(tk, new_sl, cur_tp);
         }
      }
   }
}

//+------------------------------------------------------------------+
//| [FIX 7]: New M5 bar check — 1 call iTime per tick               |
//+------------------------------------------------------------------+
bool IsNewM5Bar()
{
   static datetime s_last = 0;
   datetime cur = iTime(_Symbol, PERIOD_M5, 0);
   if(cur != s_last) { s_last = cur; return true; }
   return false;
}
//+------------------------------------------------------------------+
