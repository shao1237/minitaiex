"""
自動化交易系統 — 策略引擎
============================
從行情數據中運算交易信號，串接 strategies.py 中已驗證的策略函數。
"""
import pandas as pd
from typing import Optional, Dict
from strategies import STRATEGIES
import config


class StrategyEngine:
    """
    策略引擎

    負責：
    1. 從歷史日線數據計算交易信號
    2. 比較「目前信號」與「前一個信號」，判斷是否需要下單
    3. 支援單策略或多策略投票機制

    Parameters
    ----------
    strategy_name : str  策略名稱（對應 STRATEGIES 字典的 key）
    """

    def __init__(self, strategy_name: str = None):
        self.strategy_name = strategy_name or config.PRIMARY_STRATEGY
        self.strategy_func = STRATEGIES.get(self.strategy_name)
        self.last_signal = 0  # 上一個信號

        if self.strategy_func is None:
            available = ", ".join(STRATEGIES.keys())
            raise ValueError(
                f"找不到策略 '{self.strategy_name}'。\n"
                f"可用策略：{available}"
            )

        print(f"[StrategyEngine] 🧠 已載入策略: {self.strategy_name}")

    def compute_signal(self, df: pd.DataFrame) -> Dict:
        """
        根據最新行情計算交易信號。

        Parameters
        ----------
        df : pd.DataFrame  歷史日線數據 (含 Open, High, Low, Close, Volume)

        Returns
        -------
        dict : {
            "signal": int (1=多, -1=空, 0=空手),
            "prev_signal": int,
            "action_needed": str ("OPEN_LONG", "OPEN_SHORT", "CLOSE", "REVERSE_LONG", "REVERSE_SHORT", "HOLD"),
            "price": float (最新收盤價),
            "date": str
        }
        """
        # 檢查是否需要暖機資料
        needs_warmup = "Session Edge Ensemble" not in self.strategy_name
        
        if df.empty or (needs_warmup and len(df) < config.WARMUP_BARS):
            return {
                "signal": 0, "prev_signal": 0,
                "action_needed": "HOLD",
                "price": 0, "date": "",
                "reason": f"數據不足 ({len(df)}/{config.WARMUP_BARS})"
            }

        # 計算策略信號
        try:
            # Trendlines 策略需要特殊參數
            if "Trendlines" in self.strategy_name:
                signals = self.strategy_func(df, lookback=5)
            else:
                signals = self.strategy_func(df)
        except Exception as e:
            return {
                "signal": 0, "prev_signal": self.last_signal,
                "action_needed": "HOLD",
                "price": float(df["Close"].iloc[-1]),
                "date": str(df.index[-1].date()),
                "reason": f"策略計算錯誤: {e}"
            }

        current_signal = int(signals.iloc[-1])
        prev_signal = self.last_signal
        current_price = float(df["Close"].iloc[-1])
        current_date = str(df.index[-1].date())

        # 判斷需要什麼動作
        action = self._determine_action(prev_signal, current_signal)

        result = {
            "signal": current_signal,
            "prev_signal": prev_signal,
            "action_needed": action,
            "price": current_price,
            "date": current_date,
            "reason": self._describe_action(action, current_signal),
        }

        # 更新上一個信號
        self.last_signal = current_signal

        return result

    def _determine_action(self, prev: int, current: int) -> str:
        """
        根據前後信號判斷需要執行的動作。

        Returns
        -------
        str : 動作類型
            "HOLD" - 持倉不動 / 繼續空手
            "OPEN_LONG" - 從空手開多
            "OPEN_SHORT" - 從空手開空
            "CLOSE" - 平倉回到空手
            "REVERSE_LONG" - 從空單反手做多（平空 + 開多）
            "REVERSE_SHORT" - 從多單反手做空（平多 + 開空）
        """
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
        """產生人類可讀的動作描述。"""
        descriptions = {
            "HOLD": "持倉不動" if signal != 0 else "繼續觀望",
            "OPEN_LONG": "策略翻多 → 開多單",
            "OPEN_SHORT": "策略翻空 → 開空單",
            "CLOSE": "策略轉中性 → 平倉",
            "REVERSE_LONG": "策略由空轉多 → 反手做多",
            "REVERSE_SHORT": "策略由多轉空 → 反手做空",
        }
        return descriptions.get(action, action)

    def set_last_signal(self, signal: int):
        """手動設定上一個信號（用於恢復狀態）。"""
        self.last_signal = signal
        print(f"[StrategyEngine] 上一個信號已設定為: {signal}")
