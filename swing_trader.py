"""
台指期自動化交易系統 — Swing 策略專屬執行主程式
===================================================
專門用來執行「Swing High Low + Keltner Channel」順勢拉回策略的實體自動化下單程式。
支持分批平倉 50%、移動保本停損、以及 Telegram 即時交易卡片推送。

運行方式：
    # Paper Trading（模擬盤）
    python swing_trader.py

    # 實盤交易（需要 Shioaji API 憑證）
    python swing_trader.py --live

    # 單次信號檢查（不持續運行）
    python swing_trader.py --once
"""
import os
import sys
import time
import signal
import argparse
import traceback
import json
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta

# 確保當前目錄在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from core import notifier
from core.broker_api import BrokerAPI
from core.risk_manager import RiskManager
from core.swing_kc_backtest import compute_indicators


class SwingAutoTrader:
    """
    Swing High Low + Keltner Channel 實體自動化交易引擎
    """
    def __init__(self, paper_trading: bool = True):
        self.paper_trading = paper_trading
        self.running = False
        self.state_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "swing_state.json")

        print("=" * 60)
        print("🚀 台指期 Swing Keltner 自動交易系統 v1.0")
        print(f"   模式: {'🧪 Paper Trading (模擬盤)' if paper_trading else '🔥 Live Trading (實盤交易)'}")
        print("=" * 60)

        # 初始化核心模組
        self.broker = BrokerAPI(paper_trading=paper_trading)
        self.risk_mgr = RiskManager()

        # 策略核心參數 (與 15分K 黃金調教組合一致)
        self.ema_len = 10
        self.kc_mult = 2.5
        self.swing_len = 1
        self.buffer_pct = 0.0015
        self.slope_threshold = 0.02
        self.cooldown_bars = 12

        # 交易狀態變數 (預設)
        self.position = 0.0          # 0.0=空手, 1.0=多單滿倉, 0.5=多單半倉
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.cooldown_counter = 0

        # 註冊 Ctrl+C 平滑關閉
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    # ═══════════════════════════════════════════════════════════
    # 狀態持久化 (防止重啟丟失停損點)
    # ═══════════════════════════════════════════════════════════

    def _save_state(self):
        """將目前交易狀態原子寫入至 JSON (使用 tmp 檔與 os.replace 防止損毀)"""
        state = {
            "position": self.position,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "cooldown_counter": self.cooldown_counter,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        tmp_file = self.state_file + ".tmp"
        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=4)
            # 原子替換
            os.replace(tmp_file, self.state_file)
        except Exception as e:
            print(f"[Error] 原子寫入交易狀態失敗: {e}")
            if os.path.exists(tmp_file):
                try:
                    os.remove(tmp_file)
                except:
                    pass

    def _load_state(self):
        """從 JSON 載入交易狀態"""
        if not os.path.exists(self.state_file):
            print("📝 找不到舊有交易狀態，初始化為空手狀態。")
            return

        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            self.position = float(state.get("position", 0.0))
            self.entry_price = float(state.get("entry_price", 0.0))
            self.stop_loss = float(state.get("stop_loss", 0.0))
            self.cooldown_counter = int(state.get("cooldown_counter", 0))
            print(f"💾 成功載入歷史狀態：部位={self.position} | 進場價={self.entry_price} | 停損點={self.stop_loss} | 冷卻={self.cooldown_counter}")
        except Exception as e:
            print(f"[Error] 載入交易狀態失敗: {e}")

    def _reconcile_position(self) -> bool:
        """雙向對齊機制 (Reconciliation)：比對本地 JSON 狀態與券商實體帳戶持倉"""
        print("🔍 啟動雙向持倉對齊檢查 (Reconciliation)...")
        # 取得實體持倉
        real_pos = self.broker.get_position()
        real_dir = real_pos.get("direction", 0)
        real_contracts = real_pos.get("contracts", 0)

        # 實體持倉轉換為策略狀態部位 (Long-Only)
        expected_position = 0.0
        if real_dir == 1:
            if real_contracts == 2:
                expected_position = 1.0
            elif real_contracts == 1:
                expected_position = 0.5
            else:
                expected_position = -999.0  # 口數不匹配
        elif real_dir == -1:
            expected_position = -999.0  # 持有空單不匹配

        if expected_position == -999.0:
            err_msg = (
                f"🚨 <b>【嚴重對齊警告】券商實體持倉異常！</b>\n"
                f"券商持倉方向: {real_dir} | 口數: {real_contracts}\n"
                f"此持倉與策略 Long-Only 不相符，為保障帳戶安全，系統拒絕交易！請人工檢查部位。"
            )
            print(f"❌ {err_msg.replace('<b>', '').replace('</b>', '')}")
            notifier.send_message(err_msg)
            return False

        if self.position != expected_position:
            err_msg = (
                f"🚨 <b>【嚴重對齊警告】雙向持倉不一致！</b>\n"
                f"本地 JSON 記錄狀態: {self.position} (1.0=2口, 0.5=1口)\n"
                f"券商實體實際持倉: {expected_position} (實際口數: {real_contracts})\n"
                f"可能存在檔案毀損或人工手動平倉，為保障安全已自動終止程式！"
            )
            print(f"❌ {err_msg.replace('<b>', '').replace('</b>', '')}")
            notifier.send_message(err_msg)
            return False

        print("✅ 雙向持倉對齊一致，系統安全，准予交易。")
        return True

    # ═══════════════════════════════════════════════════════════
    # 啟動與循環
    # ═══════════════════════════════════════════════════════════

    def start(self, run_once: bool = False):
        """啟動交易系統"""
        if not self.broker.connect():
            print("❌ 無法連接券商 API，系統終止。")
            return

        # 載入歷史部位狀態
        self._load_state()

        # 執行雙向對齊檢查
        if not self._reconcile_position():
            print("❌ 雙向對齊失敗，系統自動終止運作。")
            return

        mode_text = "Paper Trading" if self.paper_trading else "實盤交易"
        notifier.send_message(
            f"<b>🔔 【交易系統啟動】</b>\n"
            f"策略: Swing High Low + Keltner Channel\n"
            f"模式: {mode_text}\n"
            f"目前狀態: 部位 {self.position} | 停損 {self.stop_loss:.0f}"
        )

        if run_once:
            self._check_and_trade()
            return

        self.running = True
        print(f"\n🔄 開始持續監控（每 {config.SIGNAL_CHECK_INTERVAL} 秒檢查一次）")
        print("   按 Ctrl+C 退出\n")

        while self.running:
            try:
                self._check_and_trade()
                
                # 檢查是否到收盤時間 (台指期日盤收盤 13:45)
                # 這裡不強迫自動關閉，可持續運行到夜盤
                time.sleep(config.SIGNAL_CHECK_INTERVAL)

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\n❌ 異常: {e}")
                traceback.print_exc()
                notifier.send_message(f"⚠️ <b>交易系統異常警告</b>\n{str(e)}")
                time.sleep(30)

        self._shutdown()

    def _shutdown(self):
        print("\n🛑 正在關閉系統...")
        notifier.send_message("<b>🛑 【交易系統已停止運行】</b>")
        self.broker.disconnect()
        self.running = False

    def _handle_shutdown(self, signum, frame):
        self.running = False

    # ═══════════════════════════════════════════════════════════
    # 核心交易邏輯 (與回測 100% 邏輯一致)
    # ═══════════════════════════════════════════════════════════

    def _check_and_trade(self):
        """檢查信號並執行下單與狀態轉移"""
        # 1. 風控檢查
        can_trade, reason = self.risk_mgr.can_trade()
        if not can_trade:
            print(f"   ⛔ 風控攔截: {reason}")
            return

        # 2. 同步並讀取 15分K Parquet 歷史數據
        if self.broker.api and self.broker.connected:
            try:
                from core.download_kbar import download_1m_kbars, aggregate_to_5min
                from core.aggregate_15min import aggregate_5m_to_15m
                
                # 同步最近 5 天的資料
                today_str = datetime.now().strftime("%Y-%m-%d")
                start_str = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
                
                download_1m_kbars(self.broker.api, start_str, today_str)
                aggregate_to_5min()
                aggregate_5m_to_15m()
                print("   🔄 已即時同步最新 K 線 Parquet 資料")
            except Exception as e:
                print(f"   ⚠️ 即時 K 線同步失敗 (將使用本地歷史): {e}")

        # 載入本地 Parquet 資料
        root_dir = os.path.dirname(os.path.abspath(__file__))
        parquet_path = os.path.join(root_dir, "data", "mxf_15min.parquet")
        if not os.path.exists(parquet_path):
            print(f"   ❌ 找不到 15分K 本地 Parquet 資料，路徑: {parquet_path}")
            return

        try:
            df = pd.read_parquet(parquet_path)
        except Exception as e:
            print(f"   ❌ 讀取 Parquet 資料失敗: {e}")
            return

        if len(df) < 50:
            print("   ⚠️ 行情數據不足以暖機指標")
            return

        # 3. 獲取即時價格 (跳動報價)
        curr_price = self.broker.get_current_price()
        if curr_price <= 0:
            # 降級方案：使用已收盤最後一根 K 棒 Close
            curr_price = float(df["Close"].iloc[-1])
        else:
            # 將最新即時跳動報價併入 DataFrame 最後一根 (模擬未完成的即時 K 棒)
            now = datetime.now()
            new_row = pd.DataFrame({
                "Open": [curr_price],
                "High": [curr_price],
                "Low": [curr_price],
                "Close": [curr_price],
                "Volume": [0]
            }, index=[now])
            df = pd.concat([df, new_row])

        # 計算指標
        df_indicators = compute_indicators(df, ema_len=self.ema_len, atr_len=self.ema_len, kc_mult=self.kc_mult)

        # 已收盤的最新 K 棒索引 (倒數第二根，因為最後一根是我們拼接的即時跳動價)
        t = len(df_indicators) - 2
        
        # 指標值 (已收盤)
        basis = df_indicators["Basis"].values
        upper = df_indicators["UpperBand"].values
        lower = df_indicators["LowerBand"].values
        atr = df_indicators["ATR"].values
        ema_slope = df_indicators["EMA_Slope"].values
        
        pct_above_ema = df_indicators["Pct_Above_EMA"].values
        touched_upper = df_indicators["Touched_Upper"].values
        pulled_back_long = df_indicators["Pulled_Back_Long"].values
        
        opens = df_indicators["Open"].values
        highs = df_indicators["High"].values
        lows = df_indicators["Low"].values
        closes = df_indicators["Close"].values

        # 3. 判定上一根 K 棒是否為 Swing Low
        # Swing Low 條件：L_t-1 < L_t 且 L_t-1 < L_t-2 (左右各1根)
        is_swing_low = False
        target_bar = t - 1
        if lows[target_bar] < lows[t] and lows[target_bar] < lows[t-2]:
            is_swing_low = True
            
        swing_low_near_support = is_swing_low and (lows[target_bar] <= basis[target_bar]) and (lows[target_bar] >= lower[target_bar] - 0.2 * atr[target_bar])

        # 更新冷卻計數器
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self._save_state()

        print(f"⏰ 檢查時間: {datetime.now().strftime('%H:%M:%S')} | 現價: {curr_price:.0f} | 部位: {self.position} | 停損點: {self.stop_loss:.0f} | 冷卻: {self.cooldown_counter}")

        # ─── A. 空手狀態 (Position == 0.0) ───
        if self.position == 0.0 and self.cooldown_counter == 0:
            # 大趨勢判定 (中軌斜率大於門檻，且大部分時間在中軌之上，且強勢觸碰過上軌)
            is_trend_long = (ema_slope[t] > self.slope_threshold) and (pct_above_ema[t] >= 0.6) and touched_upper[t]
            
            # 拉回判定 (最近曾拉回到價值區，且有 Swing Low 轉折)
            if is_trend_long and pulled_back_long[t] and swing_low_near_support:
                # 計算初始停損價
                init_sl = lows[target_bar] * (1 - self.buffer_pct)
                
                # 執行開倉：進場多單 2 口 (小台)
                # 1.0 部位代表 2 口，0.5 部位代表 1 口
                contracts = int(config.MAX_CONTRACTS * 2) if config.MAX_CONTRACTS > 0 else 2
                
                result = self.broker.place_order(direction=1, contracts=contracts, price=curr_price)
                if result["success"]:
                    self.position = 1.0
                    self.entry_price = curr_price
                    self.stop_loss = init_sl
                    self._save_state()
                    
                    msg = (
                        f"🚀 <b>【OPEN】LONG 多單進場 (2口)</b>\n"
                        f"成交價格: {curr_price:.0f}\n"
                        f"初始停損: {self.stop_loss:.0f}\n"
                        f"進場依據: 趨勢拉回且確認 Swing Low 支撐"
                    )
                    notifier.send_message(msg)
                    print(f"   ✅ 進場成功: LONG 2口 @ {curr_price:.0f}")

        # ─── B. 多單滿倉 (Position == 1.0) ───
        elif self.position == 1.0:
            # B1. 觸及止損
            if curr_price <= self.stop_loss:
                contracts = int(config.MAX_CONTRACTS * 2) if config.MAX_CONTRACTS > 0 else 2
                result = self.broker.place_order(direction=-1, contracts=contracts, price=curr_price)
                if result["success"]:
                    self.position = 0.0
                    self.cooldown_counter = self.cooldown_bars
                    self._save_state()
                    
                    msg = (
                        f"🚨 <b>【STOP_LOSS】多單觸及初始停損</b>\n"
                        f"平倉價格: {curr_price:.0f}\n"
                        f"虧損點數: {curr_price - self.entry_price:.0f}\n"
                        f"進入冷卻期: {self.cooldown_bars} 根 K 棒暫停交易"
                    )
                    notifier.send_message(msg)
                    print(f"   ❌ 觸及停損，多單全平 @ {curr_price:.0f}")
                    
            # B2. 觸及上軌，分批平倉 50% (平倉 1 口)
            elif curr_price >= upper[t+1]:
                contracts = int(config.MAX_CONTRACTS) if config.MAX_CONTRACTS > 0 else 1
                result = self.broker.place_order(direction=-1, contracts=contracts, price=curr_price)
                if result["success"]:
                    self.position = 0.5
                    self.stop_loss = self.entry_price  # 剩餘部位停損移至保本
                    self._save_state()
                    
                    msg = (
                        f"💰 <b>【TAKE_PROFIT_1】多單減碼平倉 50% (1口)</b>\n"
                        f"成交價格: {curr_price:.0f}\n"
                        f"獲利點數: {curr_price - self.entry_price:.0f}\n"
                        f"剩餘部位安全防守點: 移至保本價 {self.stop_loss:.0f}"
                    )
                    notifier.send_message(msg)
                    print(f"   💰 觸及上軌，減碼一半 @ {curr_price:.0f}")

        # ─── C. 多單半倉 (Position == 0.5) ───
        elif self.position == 0.5:
            # C1. 觸及移動防守點/保本點
            if curr_price <= self.stop_loss:
                contracts = int(config.MAX_CONTRACTS) if config.MAX_CONTRACTS > 0 else 1
                result = self.broker.place_order(direction=-1, contracts=contracts, price=curr_price)
                if result["success"]:
                    self.position = 0.0
                    self._save_state()
                    
                    pnl = curr_price - self.entry_price
                    pnl_text = f"獲利: +{pnl:.0f}" if pnl >= 0 else f"虧損: {pnl:.0f}"
                    msg = (
                        f"🛡️ <b>【EXIT】多單防守平倉離場 (1口)</b>\n"
                        f"平倉價格: {curr_price:.0f}\n"
                        f"最終損益點數: {pnl_text}"
                    )
                    notifier.send_message(msg)
                    print(f"   🛡️ 觸及防守點，平倉離場 @ {curr_price:.0f}")
                    
            # C2. 產生新的 Swing Low，上移防守點 (追蹤止盈)
            elif swing_low_near_support:
                new_sl = lows[target_bar] * (1 - self.buffer_pct)
                if new_sl > self.stop_loss:
                    old_sl = self.stop_loss
                    self.stop_loss = new_sl
                    self._save_state()
                    
                    msg = (
                        f"📈 <b>【STOP_LOSS_UPDATE】移動止盈上移</b>\n"
                        f"原防守點: {old_sl:.0f}\n"
                        f"新防守點: {self.stop_loss:.0f} (鎖定更多利潤)"
                    )
                    notifier.send_message(msg)
                    print(f"   📈 追蹤止盈上移: {old_sl:.0f} -> {self.stop_loss:.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="台指期 Swing 策略自動交易")
    parser.add_argument("--live", action="store_true", help="啟用實盤交易 (預設為 Paper Trading)")
    parser.add_argument("--sim", action="store_true", help="啟用模擬交易 (Paper Trading)")
    parser.add_argument("--once", action="store_true", help="單次信號檢查後退出")
    args = parser.parse_args()

    # 判定模式 (優先級: --live > --sim > env/config)
    paper_trading = config.SHIOAJI_SIMULATION
    
    if args.live:
        print("⚠️⚠️⚠️ 警告：您正試圖啟動【實盤交易】！ ⚠️⚠️⚠️")
        confirm = input("確認請輸入 'YES'：")
        if confirm.strip() == "YES":
            paper_trading = False
        else:
            print("❌ 實盤啟動被取消，降級為模擬交易。")
            paper_trading = True
    elif args.sim:
        paper_trading = True

    trader = SwingAutoTrader(paper_trading=paper_trading)
    trader.start(run_once=args.once)
