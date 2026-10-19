"""
自動化交易系統 — 策略引擎 (5 分 K 雙強結合 + 結算日日盤不交易保護)
==================================================================
從 5 分 K 行情數據中運算交易信號，串接 strategies.py 中已驗證的策略函數，
並自動執行「每月第三個週三結算日日盤完全不交易 (05:00~15:00 強制空手)」保護機制。
"""
import pandas as pd
from typing import Dict
from core.strategies import STRATEGIES
import config


def is_settlement_day_blackout(ts: pd.Timestamp, rollover_mode: str = None) -> bool:
    """
    判斷給定時間戳是否處於台指期月結算日禁制交易時段（每月第三個星期三）：
    - skip_settlement_day (預設)：結算日 05:00 (夜盤收盤) ~ 15:00 (新合約夜盤開盤前) 完全不交易且強制空手
    - settlement_close：結算日 13:25 ~ 15:00 強制空手
    """
    mode = rollover_mode or getattr(config, "ROLLOVER_MODE", "skip_settlement_day")
    if ts.weekday() == 2 and 15 <= ts.day <= 21:
        t = ts.time()
        start_t = pd.Timestamp("05:00").time() if mode == "skip_settlement_day" else pd.Timestamp("13:25").time()
        end_t = pd.Timestamp("15:00").time()
        if start_t <= t < end_t:
            return True
    return False


class StrategyEngine:
    """
    5 分 K 策略引擎
    """

    def __init__(self, strategy_name: str = None):
        self.strategy_name = strategy_name or config.PRIMARY_STRATEGY
        self.strategy_func = STRATEGIES.get(self.strategy_name)
        self.last_signal = 0

        if self.strategy_func is None:
            available = ", ".join(STRATEGIES.keys())
            raise ValueError(
                f"找不到策略 '{self.strategy_name}'。\n"
                f"可用策略：{available}"
            )

        print(f"[StrategyEngine] 🧠 已載入策略: {self.strategy_name}")

    def compute_signal(self, df: pd.DataFrame) -> Dict:
        """
        根據最新 5 分 K 行情計算交易信號，並套用結算日保護規則。
        """
        needs_warmup = "Session Edge Ensemble" not in self.strategy_name
        min_bars = getattr(config, "WARMUP_BARS", 120)

        if df.empty or (needs_warmup and len(df) < min(min_bars, 100)):
            return {
                "signal": 0,
                "prev_signal": self.last_signal,
                "action_needed": "HOLD",
                "price": 0.0,
                "date": "",
                "reason": f"數據不足 ({len(df)}/{min_bars})",
            }

        last_ts = pd.Timestamp(df.index[-1])
        current_price = float(df["Close"].iloc[-1])
        current_date = last_ts.strftime("%Y-%m-%d %H:%M")
        prev_signal = self.last_signal

        # 1. 檢查是否處於每月第三個週三「結算日日盤不交易」時段 (05:00 ~ 15:00)
        if is_settlement_day_blackout(last_ts):
            current_signal = 0
            action = self._determine_action(prev_signal, current_signal)
            reason = (
                "結算日日盤不交易 (05:00~15:00) → 強制平倉空手"
                if action == "CLOSE"
                else "結算日日盤不交易 (05:00~15:00) → 空手觀望至 15:00 新合約夜盤"
            )
            self.last_signal = current_signal
            return {
                "signal": current_signal,
                "prev_signal": prev_signal,
                "action_needed": action,
                "price": current_price,
                "date": current_date,
                "reason": reason,
            }

        # 2. 計算策略信號
        try:
            if "Swing + Keltner Hybrid" in self.strategy_name:
                signals = self.strategy_func(
                    df,
                    swing_len=getattr(config, "SWING_LEN", 5),
                    ema_len=getattr(config, "EMA_LEN", 60),
                    atr_mult=getattr(config, "ATR_MULT", 2.0),
                )
            elif "Trendlines" in self.strategy_name:
                signals = self.strategy_func(df, lookback=5)
            else:
                signals = self.strategy_func(df)
        except Exception as e:
            return {
                "signal": 0,
                "prev_signal": prev_signal,
                "action_needed": "HOLD",
                "price": current_price,
                "date": current_date,
                "reason": f"策略計算錯誤: {e}",
            }

        current_signal = int(signals.iloc[-1])
        action = self._determine_action(prev_signal, current_signal)

        result = {
            "signal": current_signal,
            "prev_signal": prev_signal,
            "action_needed": action,
            "price": current_price,
            "date": current_date,
            "reason": self._describe_action(action, current_signal),
        }

        self.last_signal = current_signal
        return result

    def _determine_action(self, prev: int, current: int) -> str:
        if prev == current:
            return "HOLD"
        if prev == 0:
            if current == 1:
                return "OPEN_LONG"
            elif current == -1:
                return "OPEN_SHORT"
        if prev == 1:
            if current == 0:
                return "CLOSE"
            elif current == -1:
                return "REVERSE_SHORT"
        if prev == -1:
            if current == 0:
                return "CLOSE"
            elif current == 1:
                return "REVERSE_LONG"
        return "HOLD"

    def _describe_action(self, action: str, signal: int) -> str:
        descriptions = {
            "HOLD": "持倉續抱 (跨日夜盤波段)" if signal != 0 else "繼續觀望",
            "OPEN_LONG": "雙強突破上軌 + Swing High → 開多單",
            "OPEN_SHORT": "雙強跌破下軌 + Swing Low → 開空單",
            "CLOSE": "策略轉中性 → 平倉",
            "REVERSE_LONG": "雙強由空翻多 → 反手做多",
            "REVERSE_SHORT": "雙強由多翻空 → 反手做空",
        }
        return descriptions.get(action, action)

    def set_last_signal(self, signal: int):
        self.last_signal = signal
