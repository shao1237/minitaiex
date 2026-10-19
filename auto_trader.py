"""
台指期自動化交易系統 — 5 分 K 雙強波段主控程式 (日夜盤連續 + 結算日保護)
==========================================================================
系統入口：預設執行通過 Walk-Forward 與 10,000 次蒙地卡羅檢定之
「💎 Swing + Keltner Hybrid (Causal, breakout_confirm: EMA=60, Mult=2.0, SW=5)」策略。

核心特性：
  1. 即時聚合 Shioaji 5 分鐘 K 線（涵蓋日盤 08:45~13:45 與夜盤 15:00~05:00）
  2. 倉位跨日盤與夜盤保留（取消每日 13:30 強制平倉）
  3. 每月第三個週三「結算日日盤完全不交易 (05:00~15:00 強制空手)」避開換月價差跳空
  4. K 棒防抖動 (Bar Debounce) 與本地狀態持久化 (logs/position_state.json)

使用方式：
    # Paper Trading（模擬盤，預設）
    python auto_trader.py

    # 實盤交易（需要 Shioaji API 憑證）
    python auto_trader.py --live

    # 單次信號檢查（不持續運行）
    python auto_trader.py --once
"""
import os
import sys
import time
import json
import signal
import argparse
import traceback
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import config
from core import notifier
from core.broker_api import BrokerAPI
from core.risk_manager import RiskManager
from core.strategy_engine import StrategyEngine


class AutoTrader:
    """
    5 分 K 自動化交易主控程式（支援日夜盤連續持倉與結算日不交易保護）
    """

    def __init__(self, paper_trading: bool = True, strategy_name: str = None):
        self.paper_trading = paper_trading
        self.running = False
        os.makedirs(config.LOG_DIR, exist_ok=True)
        self.state_path = os.path.join(config.LOG_DIR, "position_state.json")

        print("=" * 72)
        print("🚀 台指期 5 分 K 自動化交易系統 (日夜盤連續 + 結算日不交易保護)")
        print(f"   模式: {'🧪 Paper Trading (模擬盤)' if paper_trading else '🔥 Live Trading (實盤交易)'}")
        print(f"   結算規則: 每月第三個週三 05:00~15:00 日盤完全不交易 ({config.ROLLOVER_MODE})")
        print("=" * 72)

        self.broker = BrokerAPI(paper_trading=paper_trading)
        self.risk_mgr = RiskManager()
        self.engine = StrategyEngine(strategy_name=strategy_name)

        self.current_position = 0  # 0=空手, 1=多, -1=空
        self.entry_price = 0.0
        self._last_bar_ts = None

        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    # ═══════════════════════════════════════════════════════════
    # 啟動 / 關閉
    # ═══════════════════════════════════════════════════════════

    def start(self, run_once: bool = False):
        if not self.broker.connect():
            print("❌ 無法連接券商 API，系統終止。")
            return

        self._restore_position()

        mode = "Paper Trading" if self.paper_trading else "實盤交易"
        notifier.notify_system(
            "系統啟動",
            f"策略: {self.engine.strategy_name}\n週期: 5分K (含夜盤)\n模式: {mode}",
        )

        if run_once:
            self._check_and_trade()
            self._print_status()
            self.broker.disconnect()
            return

        self.running = True
        print(f"\n🔄 開始持續監控（每 {config.SIGNAL_CHECK_INTERVAL} 秒檢查最新已完成 5 分 K）")
        print("   按 Ctrl+C 退出\n")

        while self.running:
            try:
                self._check_and_trade()
                self._print_status()
                time.sleep(config.SIGNAL_CHECK_INTERVAL)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\n❌ 異常: {e}")
                traceback.print_exc()
                notifier.notify_system("系統異常", str(e))
                time.sleep(30)

        self._shutdown()

    def _shutdown(self):
        print("\n🛑 正在關閉系統...")
        notifier.notify_system("系統關閉", "交易系統已停止運行。")
        self.broker.disconnect()
        self.running = False

    def _handle_shutdown(self, signum, frame):
        self.running = False

    # ═══════════════════════════════════════════════════════════
    # 核心交易邏輯
    # ═══════════════════════════════════════════════════════════

    def _check_and_trade(self):
        can_trade, reason = self.risk_mgr.can_trade()
        if not can_trade:
            print(f"   ⛔ 風控攔截: {reason}")
            return

        df = self.broker.get_5min_kbars(bars=config.WARMUP_BARS)
        if df.empty:
            print("   ⚠️ 無法取得 5 分 K 行情數據")
            return

        latest_bar_ts = df.index[-1]

        # 設定引擎的上一次信號（與目前持倉同步）
        self.engine.set_last_signal(self.current_position)

        # 計算信號（內含結算日 05:00~15:00 自動強制空手邏輯）
        result = self.engine.compute_signal(df)
        action = result["action_needed"]
        price = result["price"]

        # 若最新完成的 5 分 K 時間戳未變且無需動作，僅略過重複觸發
        if self._last_bar_ts == latest_bar_ts and action == "HOLD":
            return

        self._last_bar_ts = latest_bar_ts

        print(
            f"   📊 [{result['date']}] 信號: {result['signal']:+d} | 動作: {action} | "
            f"價格: {price:.0f} | {result['reason']}"
        )

        if action == "HOLD":
            return

        self._execute_action(action, result)

    def _execute_action(self, action: str, signal_result: dict):
        price = signal_result["price"]

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

    def _open_position(self, direction: int, price: float, reason: str):
        allowed, msg = self.risk_mgr.check_position_limit(abs(self.current_position))
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
        if self.current_position == 0:
            return 0.0

        dir_text = "LONG" if self.current_position == 1 else "SHORT"
        close_direction = -self.current_position
        result = self.broker.place_order(close_direction, config.MAX_CONTRACTS, price)

        if result["success"]:
            pnl_points = (price - self.entry_price) * self.current_position
            pnl_cash = (
                pnl_points * config.CONTRACT_MULTIPLIER * config.MAX_CONTRACTS
                - config.COMMISSION_PER_SIDE * 2 * config.MAX_CONTRACTS
            )

            pnl_emoji = "💰" if pnl_cash > 0 else "💸"
            print(f"   {pnl_emoji} 平倉: {dir_text} @ {price:.0f} | 損益: {pnl_cash:+,.0f} TWD")

            notifier.notify_trade(
                "CLOSE", dir_text, price, f"損益: {pnl_cash:+,.0f} TWD (點數: {pnl_points:+.0f})"
            )

            self.current_position = 0
            self.entry_price = 0.0
            self.broker.update_paper_position(0, 0.0)
            self._save_position()

            return pnl_cash
        else:
            print(f"   ❌ 平倉失敗: {result['msg']}")
            return 0.0

    # ═══════════════════════════════════════════════════════════
    # 狀態持久化與恢復
    # ═══════════════════════════════════════════════════════════

    def _save_position(self):
        state = {
            "direction": self.current_position,
            "entry_price": self.entry_price,
            "strategy": self.engine.strategy_name,
            "updated_at": datetime.now().isoformat(),
        }
        tmp_path = self.state_path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self.state_path)
        except Exception as e:
            print(f"[AutoTrader] ⚠️ 儲存持倉狀態失敗: {e}")

    def _restore_position(self):
        if self.paper_trading and os.path.exists(self.state_path):
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    state = json.load(f)
                self.current_position = int(state.get("direction", 0))
                self.entry_price = float(state.get("entry_price", 0.0))
                self.broker.update_paper_position(self.current_position, self.entry_price)
            except Exception as e:
                print(f"[AutoTrader] ⚠️ 讀取本地持倉狀態失敗: {e}")
        else:
            pos = self.broker.get_position()
            self.current_position = pos["direction"]
            self.entry_price = pos["entry_price"]

        if self.current_position != 0:
            dir_text = "多單" if self.current_position == 1 else "空單"
            print(f"📌 恢復持倉: {dir_text} @ {self.entry_price:.0f}")
        else:
            print("📌 目前空手")

    def _print_status(self):
        now = datetime.now().strftime("%H:%M:%S")
        pos_text = {0: "空手", 1: "多單", -1: "空單"}
        summary = self.risk_mgr.get_daily_summary()
        print(
            f"   ⏰ {now} | 持倉: {pos_text.get(self.current_position, '未知')} | "
            f"今日 PnL: {summary['daily_pnl']:+,.0f} | 交易: {summary['daily_trades']} 筆"
        )


def main():
    parser = argparse.ArgumentParser(
        description="台指期 5 分 K 自動化交易系統 (Swing + Keltner Hybrid)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--live", action="store_true", help="實盤模式（預設為模擬盤）")
    parser.add_argument("--sim", action="store_true", help="強制使用模擬盤模式")
    parser.add_argument("--once", action="store_true", help="僅檢查一次信號後退出")
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help=f"指定策略名稱（預設: {config.PRIMARY_STRATEGY}）",
    )
    args = parser.parse_args()

    paper = True if args.sim else (not args.live)
    trader = AutoTrader(
        paper_trading=paper,
        strategy_name=args.strategy,
    )
    trader.start(run_once=args.once)


if __name__ == "__main__":
    main()
