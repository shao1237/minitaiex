"""
自動化交易系統 — 風控模組
============================
保護帳戶免受極端虧損，是整個系統最重要的安全機制。
"""
import json
import os
from datetime import datetime, date
from typing import Dict, List
import config
import notifier


class RiskManager:
    """
    風控模組

    功能：
    1. 每日最大虧損限制
    2. 連續虧損次數限制
    3. 交易時段管控
    4. 部位上限控制

    所有風控觸發都會透過 Telegram 通知。
    """

    def __init__(self):
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.daily_wins = 0
        self.consecutive_losses = 0
        self.is_halted = False
        self.halt_reason = ""
        self._today = date.today()

        # 載入今日狀態（如果程式重啟）
        self._state_path = os.path.join(config.LOG_DIR, "risk_state.json")
        os.makedirs(config.LOG_DIR, exist_ok=True)
        self._load_state()

    # ═══════════════════════════════════════════════════════════
    # 核心檢查
    # ═══════════════════════════════════════════════════════════

    def can_trade(self) -> tuple:
        """
        綜合風控檢查：是否允許交易。

        Returns
        -------
        tuple : (allowed: bool, reason: str)
        """
        # 每日重置
        if date.today() != self._today:
            self._reset_daily()

        # 檢查 1：是否已被暫停
        if self.is_halted:
            return False, f"交易已暫停: {self.halt_reason}"

        # 檢查 2：交易時段
        now = datetime.now()
        market_open = now.replace(
            hour=config.TRADING_START_HOUR,
            minute=config.TRADING_START_MINUTE, second=0
        )
        market_close = now.replace(
            hour=config.TRADING_END_HOUR,
            minute=config.TRADING_END_MINUTE, second=0
        )
        if not (market_open <= now <= market_close):
            return False, f"非交易時段 ({now.strftime('%H:%M')})"

        # 檢查 3：每日最大虧損
        if self.daily_pnl <= -config.MAX_DAILY_LOSS:
            self._halt(f"當日虧損已達上限 ({self.daily_pnl:,.0f} TWD)")
            return False, self.halt_reason

        # 檢查 4：連續虧損
        if self.consecutive_losses >= config.MAX_CONSECUTIVE_LOSSES:
            self._halt(
                f"連續虧損 {self.consecutive_losses} 次，"
                f"已達上限 {config.MAX_CONSECUTIVE_LOSSES} 次"
            )
            return False, self.halt_reason

        return True, "OK"

    def check_position_limit(self, current_contracts: int) -> tuple:
        """
        檢查部位是否超過上限。

        Returns
        -------
        tuple : (allowed: bool, reason: str)
        """
        if current_contracts >= config.MAX_CONTRACTS:
            return False, f"持倉已達上限 ({current_contracts}/{config.MAX_CONTRACTS})"
        return True, "OK"

    # ═══════════════════════════════════════════════════════════
    # 交易結果回報
    # ═══════════════════════════════════════════════════════════

    def record_trade(self, pnl: float):
        """
        回報一筆交易的損益，更新風控狀態。

        Parameters
        ----------
        pnl : float  該筆交易的淨損益 (TWD)
        """
        self.daily_pnl += pnl
        self.daily_trades += 1

        if pnl > 0:
            self.daily_wins += 1
            self.consecutive_losses = 0
        elif pnl < 0:
            self.consecutive_losses += 1

        self._save_state()

        # 檢查是否觸發風控
        if self.daily_pnl <= -config.MAX_DAILY_LOSS:
            self._halt(f"當日虧損已達上限 ({self.daily_pnl:,.0f} TWD)")

        if self.consecutive_losses >= config.MAX_CONSECUTIVE_LOSSES:
            self._halt(
                f"連續虧損 {self.consecutive_losses} 次"
            )

    def get_daily_summary(self) -> Dict:
        """取得今日交易摘要。"""
        return {
            "date": self._today.isoformat(),
            "daily_pnl": self.daily_pnl,
            "daily_trades": self.daily_trades,
            "daily_wins": self.daily_wins,
            "consecutive_losses": self.consecutive_losses,
            "is_halted": self.is_halted,
            "halt_reason": self.halt_reason,
        }

    # ═══════════════════════════════════════════════════════════
    # 內部方法
    # ═══════════════════════════════════════════════════════════

    def _halt(self, reason: str):
        """暫停交易並發送通知。"""
        self.is_halted = True
        self.halt_reason = reason
        self._save_state()
        notifier.notify_risk_alert("交易暫停", reason)
        print(f"[RiskManager] 🚨 {reason}")

    def _reset_daily(self):
        """每日重置。"""
        self._today = date.today()
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.daily_wins = 0
        self.is_halted = False
        self.halt_reason = ""
        # 注意：consecutive_losses 不重置，跨日延續
        self._save_state()
        print(f"[RiskManager] 📅 每日重置完成 ({self._today})")

    def _save_state(self):
        """儲存狀態到磁碟（程式重啟後可恢復）。"""
        state = {
            "date": self._today.isoformat(),
            "daily_pnl": self.daily_pnl,
            "daily_trades": self.daily_trades,
            "daily_wins": self.daily_wins,
            "consecutive_losses": self.consecutive_losses,
            "is_halted": self.is_halted,
            "halt_reason": self.halt_reason,
        }
        try:
            with open(self._state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[RiskManager] 狀態儲存失敗: {e}")

    def _load_state(self):
        """從磁碟載入狀態。"""
        if not os.path.exists(self._state_path):
            return

        try:
            with open(self._state_path, "r", encoding="utf-8") as f:
                state = json.load(f)

            saved_date = date.fromisoformat(state.get("date", "2000-01-01"))
            if saved_date == date.today():
                # 同一天 → 恢復狀態
                self.daily_pnl = state.get("daily_pnl", 0)
                self.daily_trades = state.get("daily_trades", 0)
                self.daily_wins = state.get("daily_wins", 0)
                self.consecutive_losses = state.get("consecutive_losses", 0)
                self.is_halted = state.get("is_halted", False)
                self.halt_reason = state.get("halt_reason", "")
                print(f"[RiskManager] 已恢復今日狀態 (PnL: {self.daily_pnl:,.0f})")
            else:
                # 不同天 → 重置（但保留 consecutive_losses）
                self.consecutive_losses = state.get("consecutive_losses", 0)
                print(f"[RiskManager] 新的一天，連續虧損次數延續: {self.consecutive_losses}")

        except Exception as e:
            print(f"[RiskManager] 狀態載入失敗: {e}")
