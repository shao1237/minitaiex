"""
台指期自動化交易系統 — 主控程式
====================================
系統入口：初始化所有模組，排程在盤中定時檢查信號並執行交易。

使用方式：
    # Paper Trading（模擬盤）
    python auto_trader.py

    # 實盤交易（需要 Shioaji API）
    python auto_trader.py --live

    # 單次信號檢查（不持續運行）
    python auto_trader.py --once

    # 指定策略
    python auto_trader.py --strategy "Trendlines with Breaks (LuxAlgo)"
"""
import os
import sys
import time
import signal
import argparse
import traceback
import json
import pandas as pd
from datetime import datetime, date, timedelta

# 確保當前目錄在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import notifier
from core.broker_api import BrokerAPI
from core.risk_manager import RiskManager
from strategy_engine import StrategyEngine


class AutoTrader:
    """
    自動化交易主控程式

    負責：
    1. 初始化與協調所有模組
    2. 定時循環：抓行情 → 算信號 → 風控 → 下單 → 通知
    3. 部位管理與損益追蹤
    4. 優雅關閉與例外處理

    Parameters
    ----------
    paper_trading : bool  是否為模擬盤模式
    strategy_name : str  策略名稱（None=使用 config 預設）
    """

    def __init__(self, paper_trading: bool = True,
                 strategy_name: str = None):
        self.paper_trading = paper_trading
        self.running = False

        # 初始化模組
        print("=" * 60)
        print("🚀 台指期自動化交易系統 v1.0")
        print(f"   模式: {'🧪 Paper Trading' if paper_trading else '🔥 實盤交易'}")
        print("=" * 60)

        self.broker = BrokerAPI(paper_trading=paper_trading)
        self.risk_mgr = RiskManager()
        self.engine = StrategyEngine(strategy_name=strategy_name)

        # 持倉狀態
        self.current_position = 0  # 0=空手, 1=多, -1=空
        self.entry_price = 0.0

        # 註冊 Ctrl+C 平滑關閉
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    # ═══════════════════════════════════════════════════════════
    # 啟動 / 關閉
    # ═══════════════════════════════════════════════════════════

    def start(self, run_once: bool = False):
        """
        啟動交易系統。

        Parameters
        ----------
        run_once : bool  True=僅檢查一次信號後退出
        """
        # 連接券商
        if not self.broker.connect():
            print("❌ 無法連接券商 API，系統終止。")
            return

        # 恢復部位狀態
        self._restore_position()

        # 通知啟動
        mode = "Paper Trading" if self.paper_trading else "實盤交易"
        notifier.notify_system(
            "系統啟動",
            f"策略: {self.engine.strategy_name}\n模式: {mode}"
        )

        if run_once:
            self._check_and_trade()
            self._print_status()
            return

        # 持續運行循環
        self.running = True
        print(f"\n🔄 開始持續監控（每 {config.SIGNAL_CHECK_INTERVAL} 秒檢查一次）")
        print("   按 Ctrl+C 退出\n")

        while self.running:
            try:
                self._check_and_trade()
                self._print_status()

                # 檢查是否到收盤時間
                now = datetime.now()
                close_time = now.replace(
                    hour=config.TRADING_END_HOUR,
                    minute=config.TRADING_END_MINUTE,
                    second=0, microsecond=0
                ) + timedelta(minutes=5)
                if now > close_time:
                    self._on_market_close()
                    # 等到明天開盤
                    self._wait_until_tomorrow()

                time.sleep(config.SIGNAL_CHECK_INTERVAL)

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\n❌ 異常: {e}")
                traceback.print_exc()
                notifier.notify_system("系統異常", str(e))
                time.sleep(30)  # 出錯後等 30 秒再試

        self._shutdown()

    def _shutdown(self):
        """優雅關閉。"""
        print("\n🛑 正在關閉系統...")
        notifier.notify_system("系統關閉", "交易系統已停止運行。")
        self.broker.disconnect()
        self.running = False

    def _handle_shutdown(self, signum, frame):
        """處理 Ctrl+C。"""
        self.running = False

    # ═══════════════════════════════════════════════════════════
    # 核心交易邏輯
    # ═══════════════════════════════════════════════════════════

    def _check_and_trade(self):
        """一次完整的「檢查信號 → 風控 → 下單」流程。"""

        # 1. 風控預檢
        can_trade, reason = self.risk_mgr.can_trade()
        if not can_trade:
            print(f"   ⛔ 風控攔截: {reason}")
            return

        # 2. 取得歷史數據或即時狀態
        if "Session Edge Ensemble" in self.engine.strategy_name:
            current_price = self.broker.get_current_price()
            if current_price <= 0:
                print("   ⚠️ 無法取得有效即時報價，跳過本次檢查")
                return
            now = datetime.now()
            df = pd.DataFrame({"Close": [current_price]}, index=[now])
        else:
            df = self.broker.get_historical_data(days=config.WARMUP_BARS * 2)
            
        if df.empty:
            print("   ⚠️ 無法取得行情數據")
            return

        # 3. 設定引擎的上一次信號（與目前持倉同步）
        self.engine.set_last_signal(self.current_position)

        # 4. 計算信號
        result = self.engine.compute_signal(df)
        action = result["action_needed"]
        price = result["price"]

        print(f"   📊 信號: {result['signal']} | 動作: {action} | "
              f"價格: {price:.0f} | {result['reason']}")

        # 5. 若無需動作，直接返回
        if action == "HOLD":
            return

        # 6. 執行交易
        self._execute_action(action, result)

    def _execute_action(self, action: str, signal_result: dict):
        """
        根據策略引擎的指令執行交易。

        Parameters
        ----------
        action : str  動作類型
        signal_result : dict  策略引擎的完整結果
        """
        price = signal_result["price"]
        new_signal = signal_result["signal"]

        # ── 平倉 ──
        if action in ("CLOSE", "REVERSE_LONG", "REVERSE_SHORT"):
            if self.current_position != 0:
                pnl = self._close_position(price)
                self.risk_mgr.record_trade(pnl)

        # ── 開倉 ──
        if action in ("OPEN_LONG", "REVERSE_LONG"):
            self._open_position(1, price, signal_result["reason"])

        elif action in ("OPEN_SHORT", "REVERSE_SHORT"):
            self._open_position(-1, price, signal_result["reason"])

        elif action == "CLOSE":
            # 純平倉，不開新倉
            pass

    def _open_position(self, direction: int, price: float, reason: str):
        """開新倉。"""
        # 部位限制檢查
        allowed, msg = self.risk_mgr.check_position_limit(
            abs(self.current_position)
        )
        if not allowed:
            print(f"   ⛔ {msg}")
            return

        dir_text = "LONG" if direction == 1 else "SHORT"
        result = self.broker.place_order(direction, config.MAX_CONTRACTS, price)

        if result["success"]:
            self.current_position = direction
            self.entry_price = price
            self.broker.update_paper_position(direction, price)
            self._save_position()

            notifier.notify_trade("OPEN", dir_text, price, reason)
            print(f"   ✅ 開倉成功: {dir_text} @ {price:.0f}")
        else:
            print(f"   ❌ 開倉失敗: {result['msg']}")
            notifier.notify_system("下單失敗", result["msg"])

    def _close_position(self, price: float) -> float:
        """
        平倉並計算損益。

        Returns
        -------
        float : 淨損益 (TWD)
        """
        if self.current_position == 0:
            return 0.0

        dir_text = "LONG" if self.current_position == 1 else "SHORT"
        # 平倉方向與持倉相反
        close_direction = -self.current_position
        result = self.broker.place_order(close_direction, config.MAX_CONTRACTS, price)

        if result["success"]:
            # 計算損益
            pnl_points = (price - self.entry_price) * self.current_position
            pnl_cash = (pnl_points * config.CONTRACT_MULTIPLIER * config.MAX_CONTRACTS
                        - config.COMMISSION_PER_SIDE * 2 * config.MAX_CONTRACTS)

            pnl_emoji = "💰" if pnl_cash > 0 else "💸"
            print(f"   {pnl_emoji} 平倉: {dir_text} @ {price:.0f} | "
                  f"損益: {pnl_cash:+,.0f} TWD")

            notifier.notify_trade(
                "CLOSE", dir_text, price,
                f"損益: {pnl_cash:+,.0f} TWD (點數: {pnl_points:+.0f})"
            )

            self.current_position = 0
            self.entry_price = 0
            self.broker.update_paper_position(0, 0)
            self._save_position()

            return pnl_cash
        else:
            print(f"   ❌ 平倉失敗: {result['msg']}")
            return 0.0

    # ═══════════════════════════════════════════════════════════
    # 輔助方法
    # ═══════════════════════════════════════════════════════════

    def _save_position(self):
        """儲存模擬盤持倉狀態到檔案。"""
        if not self.paper_trading:
            return
        state = {
            "current_position": self.current_position,
            "entry_price": self.entry_price,
            "updated_at": datetime.now().isoformat()
        }
        state_path = os.path.join(config.LOG_DIR, "position_state.json")
        os.makedirs(config.LOG_DIR, exist_ok=True)
        try:
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"[警告] 無法儲存持倉狀態: {e}")

    def _restore_position(self):
        """從券商或模擬盤恢復持倉狀態。"""
        if self.paper_trading:
            state_path = os.path.join(config.LOG_DIR, "position_state.json")
            if os.path.exists(state_path):
                try:
                    with open(state_path, "r", encoding="utf-8") as f:
                        state = json.load(f)
                    self.current_position = state.get("current_position", 0)
                    self.entry_price = state.get("entry_price", 0.0)
                    self.broker.update_paper_position(self.current_position, self.entry_price)
                except Exception as e:
                    print(f"[警告] 無法讀取持倉狀態: {e}")
            else:
                self.current_position = 0
                self.entry_price = 0.0
        else:
            pos = self.broker.get_position()
            self.current_position = pos["direction"]
            self.entry_price = pos["entry_price"]

        if self.current_position != 0:
            dir_text = "多單" if self.current_position == 1 else "空單"
            print(f"📌 恢復持倉: {dir_text} @ {self.entry_price:.0f}")
        else:
            print("📌 目前空手")

    def _on_market_close(self):
        """收盤後處理：發送每日摘要。"""
        summary = self.risk_mgr.get_daily_summary()
        notifier.notify_daily_summary(
            date=summary["date"],
            pnl=summary["daily_pnl"],
            trades=summary["daily_trades"],
            win_trades=summary["daily_wins"],
            equity=config.INITIAL_CAPITAL + summary["daily_pnl"],
        )
        print(f"\n📊 每日摘要已發送 (PnL: {summary['daily_pnl']:+,.0f} TWD)")

    def _wait_until_tomorrow(self):
        """等到明天開盤前 5 分鐘。"""
        now = datetime.now()
        tomorrow_open = now.replace(
            hour=config.TRADING_START_HOUR,
            minute=config.TRADING_START_MINUTE,
            second=0, microsecond=0
        ) - timedelta(minutes=5)
        if now > tomorrow_open:
            # 加一天
            tomorrow_open += timedelta(days=1)

        wait_seconds = (tomorrow_open - now).total_seconds()
        if wait_seconds > 0:
            hours = wait_seconds / 3600
            print(f"😴 等待 {hours:.1f} 小時後開盤...")
            # 每 60 秒輪詢，以便可以 Ctrl+C 中斷
            while wait_seconds > 0 and self.running:
                time.sleep(min(60, wait_seconds))
                wait_seconds -= 60

    def _print_status(self):
        """印出目前狀態。"""
        now = datetime.now().strftime("%H:%M:%S")
        pos_text = {0: "空手", 1: "多單", -1: "空單"}
        summary = self.risk_mgr.get_daily_summary()

        print(f"   ⏰ {now} | 持倉: {pos_text[self.current_position]} | "
              f"今日 PnL: {summary['daily_pnl']:+,.0f} | "
              f"交易: {summary['daily_trades']} 筆")


# ═══════════════════════════════════════════════════════════════
# CLI 入口
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="台指期自動化交易系統 v1.0",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--live", action="store_true",
        help="實盤模式（預設為模擬盤）"
    )
    parser.add_argument(
        "--once", action="store_true",
        help="僅檢查一次信號後退出"
    )
    parser.add_argument(
        "--strategy", type=str, default=None,
        help="指定策略名稱（預設: ADX and DI）"
    )
    args = parser.parse_args()

    paper = not args.live
    trader = AutoTrader(
        paper_trading=paper,
        strategy_name=args.strategy,
    )
    trader.start(run_once=args.once)


if __name__ == "__main__":
    main()
