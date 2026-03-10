"""
FAST Custom Gymnasium Environment cho XAUUSD Trading.
Tối ưu: pre-convert sang numpy arrays, tránh pandas iloc mỗi step.
Fixed: spread tính đúng, SL/TP giống EA, không reverse trade cùng step.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class XAUUSDTradingEnv(gym.Env):
    """
    Fast Trading Environment cho XAUUSD.
    Pre-cache tất cả data sang numpy arrays → >200 it/s.
    
    Fixes v2:
    - Spread: chỉ tính 1 lần = ask-bid (entry at ask/bid, exit at bid/ask)
    - SL/TP: giống EA (configurable, default SL=5.0 TP=3.0 cho XAUUSD)  
    - Không reverse cùng step (close xong phải đợi step sau mới mở)
    - Cooldown: bar tiếp mới được trade lại (tránh overtrading)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[list] = None,
        initial_balance: float = 10000.0,
        lot_size: float = 0.01,
        spread: float = 0.36,       # Exness XAUUSD actual spread
        slippage: float = 0.05,
        window_size: int = 1,
        reward_mode: str = "sharpe",
        max_steps: Optional[int] = None,
        random_start: bool = True,
        # SL/TP giống EA IchiDCA
        stop_loss: float = 5.0,     # $5 SL (500 pips XAUUSD = $5 move)
        take_profit: float = 3.0,   # $3 TP (300 pips = $3 move)
        use_sl_tp: bool = True,     # Enable SL/TP (giống EA)
        trade_cooldown: int = 3,    # Chờ 3 bars trước khi trade tiếp
        **kwargs,
    ):
        super().__init__()

        self.initial_balance = initial_balance
        self.lot_size = lot_size
        self.spread = spread
        self.slippage = slippage
        self.window_size = window_size
        self.reward_mode = reward_mode
        self.random_start = random_start
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.use_sl_tp = use_sl_tp
        self.trade_cooldown = trade_cooldown

        # === PRE-CACHE: Convert pandas → numpy ===
        ohlcv = ["open", "high", "low", "close", "volume"]
        if feature_columns is not None:
            feat_cols = feature_columns
        else:
            feat_cols = [c for c in df.columns if c not in ohlcv]

        self.n_features = len(feat_cols)
        self._features = df[feat_cols].values.astype(np.float32)
        self._close = df["close"].values.astype(np.float64)
        self._high = df["high"].values.astype(np.float64)
        self._low = df["low"].values.astype(np.float64)
        self._data_len = len(df)

        # Pre-clean NaN/Inf ONCE
        np.nan_to_num(self._features, copy=False, nan=0.0, posinf=5.0, neginf=-5.0)

        # Observation: features + position(1) + unrealized_pnl_norm(1) + balance_change(1) + bars_since_trade(1)
        obs_size = self.n_features + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)  # 0=Hold, 1=Buy, 2=Sell, 3=Close

        self.max_steps = max_steps or self._data_len - self.window_size - 1
        self._obs_buffer = np.zeros(obs_size, dtype=np.float32)
        self._returns_buf = np.zeros(20, dtype=np.float64)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        max_start = self._data_len - self.max_steps - self.window_size
        if self.random_start and max_start > 0:
            self.start_idx = self.np_random.integers(0, max(1, max_start))
        else:
            self.start_idx = 0

        self.current_step = 0
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.position = 0       # 1=long, -1=short, 0=flat
        self.entry_price = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.max_equity = self.initial_balance
        self.max_drawdown = 0.0
        self._prev_equity = self.initial_balance
        self._bars_since_trade = 99  # Cho phép trade ngay từ đầu

        # Tracking
        self.equity_history = [self.initial_balance]
        self.trade_log = []

        self._returns_buf[:] = 0
        self._returns_idx = 0
        self._returns_count = 0

        return self._get_obs(), {}

    def _idx(self) -> int:
        return self.start_idx + self.window_size + self.current_step

    def _price(self) -> float:
        return self._close[self._idx()]

    def _get_obs(self) -> np.ndarray:
        idx = self._idx()
        self._obs_buffer[:self.n_features] = self._features[idx]

        price = self._close[idx]
        unrealized = 0.0
        if self.position != 0:
            # Unrealized PnL: (current_bid/ask - entry) * direction
            # Long: mua ở ask, bán ở bid = price - spread/2
            # Short: bán ở bid, mua ở ask = price + spread/2
            if self.position == 1:
                exit_price = price - self.spread * 0.5  # bid
            else:
                exit_price = price + self.spread * 0.5  # ask
            unrealized = (exit_price - self.entry_price) * self.position * self.lot_size * 100

        self._obs_buffer[self.n_features] = float(self.position)
        self._obs_buffer[self.n_features + 1] = unrealized / (self.initial_balance + 1e-10)
        self._obs_buffer[self.n_features + 2] = (self.equity - self.initial_balance) / (self.initial_balance + 1e-10)
        self._obs_buffer[self.n_features + 3] = min(self._bars_since_trade / 10.0, 1.0)

        return self._obs_buffer

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        idx = self._idx()
        price = self._close[idx]
        high = self._high[idx]
        low = self._low[idx]
        realized_pnl = 0.0

        self._bars_since_trade += 1

        # === Check SL/TP hit FIRST (giống EA: check mỗi bar) ===
        if self.position != 0 and self.use_sl_tp:
            sl_tp_pnl = self._check_sl_tp(high, low, price)
            if sl_tp_pnl != 0:
                realized_pnl = sl_tp_pnl

        # === Execute agent action (chỉ khi flat hoặc close) ===
        if action == 1 and self.position == 0 and self._bars_since_trade >= self.trade_cooldown:
            # OPEN LONG: Buy at ask
            self.position = 1
            self.entry_price = price + self.spread * 0.5
            self.total_trades += 1
            self._bars_since_trade = 0

        elif action == 2 and self.position == 0 and self._bars_since_trade >= self.trade_cooldown:
            # OPEN SHORT: Sell at bid
            self.position = -1
            self.entry_price = price - self.spread * 0.5
            self.total_trades += 1
            self._bars_since_trade = 0

        elif action == 3 and self.position != 0:
            # MANUAL CLOSE
            realized_pnl += self._close_pos(price)

        # Không cho phép reverse cùng step - phải close rồi đợi step sau

        # === Update equity ===
        unrealized = 0.0
        if self.position != 0:
            if self.position == 1:
                exit_price = price - self.spread * 0.5  # bid
            else:
                exit_price = price + self.spread * 0.5  # ask
            unrealized = (exit_price - self.entry_price) * self.position * self.lot_size * 100

        self.equity = self.balance + unrealized

        # Returns ring buffer
        ret = (self.equity - self._prev_equity) / (abs(self._prev_equity) + 1e-10)
        self._returns_buf[self._returns_idx % 20] = ret
        self._returns_idx += 1
        self._returns_count = min(self._returns_count + 1, 20)
        self._prev_equity = self.equity

        # Drawdown
        if self.equity > self.max_equity:
            self.max_equity = self.equity
        dd = (self.max_equity - self.equity) / (self.max_equity + 1e-10)
        if dd > self.max_drawdown:
            self.max_drawdown = dd

        # === Reward ===
        reward = self._calc_reward(realized_pnl, dd)

        self.equity_history.append(self.equity)
        self.current_step += 1

        # Termination
        terminated = self.equity <= self.initial_balance * 0.5
        truncated = (self._idx() >= self._data_len - 1) or (self.current_step >= self.max_steps)

        if terminated or truncated:
            if self.position != 0:
                self._close_pos(self._price())
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            obs = self._get_obs()

        return obs, reward, terminated, truncated, self._get_info()

    def _check_sl_tp(self, high: float, low: float, close: float) -> float:
        """Check SL/TP giống EA: dùng high/low của bar, không chỉ close."""
        if self.position == 1:  # LONG
            # TP: high >= entry + take_profit
            if high >= self.entry_price + self.take_profit:
                tp_price = self.entry_price + self.take_profit
                return self._close_pos(tp_price)
            # SL: low <= entry - stop_loss
            if low <= self.entry_price - self.stop_loss:
                sl_price = self.entry_price - self.stop_loss
                return self._close_pos(sl_price)

        elif self.position == -1:  # SHORT
            # TP: low <= entry - take_profit
            if low <= self.entry_price - self.take_profit:
                tp_price = self.entry_price - self.take_profit
                return self._close_pos(tp_price)
            # SL: high >= entry + stop_loss
            if high >= self.entry_price + self.stop_loss:
                sl_price = self.entry_price + self.stop_loss
                return self._close_pos(sl_price)

        return 0.0

    def _close_pos(self, exit_price: float) -> float:
        """Đóng vị thế. Spread đã tính sẵn trong entry/exit price."""
        if self.position == 0:
            return 0.0

        # PnL đơn giản: (exit - entry) * direction * lot * 100oz
        # Spread đã được tính ở entry (ask/bid) và exit (bid/ask)
        if self.position == 1:
            actual_exit = exit_price - self.spread * 0.5  # sell at bid
        else:
            actual_exit = exit_price + self.spread * 0.5  # buy at ask to cover

        pnl = (actual_exit - self.entry_price) * self.position * self.lot_size * 100

        self.total_pnl += pnl
        self.balance += pnl
        if pnl > 0:
            self.winning_trades += 1

        self.trade_log.append({
            "step": self.current_step,
            "type": "LONG" if self.position == 1 else "SHORT",
            "entry": self.entry_price,
            "exit": exit_price,
            "pnl": pnl,
            "balance": self.balance,
        })

        self.position = 0
        self.entry_price = 0.0
        self._bars_since_trade = 0
        return pnl

    def _calc_reward(self, realized_pnl: float, dd: float) -> float:
        """Reward function tối ưu cho Ichimoku strategy."""
        if self.reward_mode == "sharpe" and self._returns_count >= 20:
            m = self._returns_buf.mean()
            s = self._returns_buf.std() + 1e-10
            reward = m / s
            if dd > 0.10:
                reward -= (dd - 0.10) * 15
        elif self.reward_mode == "sortino" and self._returns_count >= 20:
            m = self._returns_buf.mean()
            neg = self._returns_buf[self._returns_buf < 0]
            ds = neg.std() + 1e-10 if len(neg) > 0 else 1e-10
            reward = m / ds
        else:
            reward = realized_pnl / (self.initial_balance + 1e-10) * 100
            if dd > 0.05:
                reward -= dd * 10

        # Bonus cho winning trade (khuyến khích cắt lời đúng)
        if realized_pnl > 0:
            reward += 0.5
        elif realized_pnl < 0:
            reward -= 0.2

        # Penalty blow-up
        if self.equity <= self.initial_balance * 0.5:
            reward -= 100

        return max(-10.0, min(10.0, reward))

    def _get_info(self) -> dict:
        wr = self.winning_trades / max(1, self.total_trades) * 100
        ret = (self.equity - self.initial_balance) / self.initial_balance * 100
        return {
            "balance": self.balance, "equity": self.equity,
            "position": self.position, "total_trades": self.total_trades,
            "winning_trades": self.winning_trades, "win_rate": wr,
            "total_pnl": self.total_pnl, "total_return": ret,
            "max_drawdown": self.max_drawdown * 100, "step": self.current_step,
        }

    def get_trade_log(self) -> pd.DataFrame:
        return pd.DataFrame(self.trade_log) if self.trade_log else pd.DataFrame()

    def get_performance_summary(self) -> dict:
        info = self._get_info()
        eq = np.array(self.equity_history)
        if len(eq) > 1:
            rets = np.diff(eq) / (np.abs(eq[:-1]) + 1e-10)
            sharpe = rets.mean() / (rets.std() + 1e-10) * np.sqrt(252 * 24 * 12)
        else:
            sharpe = 0.0

        tl = self.get_trade_log()
        if len(tl) > 0:
            pnls = tl["pnl"].values
            gp = pnls[pnls > 0].sum()
            gl = abs(pnls[pnls < 0].sum())
            pf = gp / (gl + 1e-10)
            aw = pnls[pnls > 0].mean() if (pnls > 0).any() else 0
            al = pnls[pnls < 0].mean() if (pnls < 0).any() else 0
        else:
            pf, aw, al = 0, 0, 0

        return {**info, "sharpe_ratio": sharpe, "profit_factor": pf,
                "avg_win": aw, "avg_loss": al, "total_steps": self.current_step}
