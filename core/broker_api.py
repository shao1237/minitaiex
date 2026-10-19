"""
自動化交易系統 — 券商 API 抽象層
===================================
封裝永豐金 Shioaji API，統一介面。
支援：
  1. 即時 1 分 K 抓取並聚合為已完成之 5 分 K（涵蓋日盤 08:45~13:45 與夜盤 15:00~05:00）
  2. Paper Trading（模擬環境連線取即時 K 棒與模擬下單）與 Live Trading（實盤下單）
"""
import os
import sys
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import config

try:
    import shioaji as sj
    HAS_SHIOAJI = True
except ImportError:
    HAS_SHIOAJI = False
    print("[BrokerAPI] ⚠️ shioaji 未安裝，僅支援本地歷史資料模擬模式。")


class BrokerAPI:
    """
    券商 API 抽象層
    """

    def __init__(self, paper_trading: bool = True):
        self.paper_trading = paper_trading
        self.api = None
        self.connected = False
        self.contract = None  # 期貨合約物件 (MXFR1)

        # 模擬盤用的假部位與損益追蹤
        self._paper_position = 0  # 0=空手, 1=多, -1=空
        self._paper_entry_price = 0.0
        self._paper_trades: List[Dict] = []
        self._paper_equity = config.INITIAL_CAPITAL

        # 交易日誌
        os.makedirs(config.LOG_DIR, exist_ok=True)
        self._trade_log_path = os.path.join(
            config.LOG_DIR,
            f"trades_{datetime.now().strftime('%Y%m%d')}.json"
        )

    # ═══════════════════════════════════════════════════════════
    # 連線 / 斷線
    # ═══════════════════════════════════════════════════════════

    def connect(self) -> bool:
        """連接券商 API（Paper 模式使用 simulation=True 取即時 K 棒；若無憑證則降級為本地資料）。"""
        if not HAS_SHIOAJI or not config.SHIOAJI_API_KEY or not config.SHIOAJI_SECRET_KEY:
            if self.paper_trading:
                print("[BrokerAPI] 🧪 Paper Trading 離線模式（使用本地 5 分 K 資料）。")
                self.connected = True
                return True
            print("[BrokerAPI] ❌ 實盤模式需要安裝 shioaji 並設定 API_KEY / SECRET_KEY。")
            return False

        try:
            use_sim = True if self.paper_trading else config.SHIOAJI_SIMULATION
            self.api = sj.Shioaji(simulation=use_sim)
            self.api.login(
                api_key=config.SHIOAJI_API_KEY,
                secret_key=config.SHIOAJI_SECRET_KEY,
                contracts_timeout=10000,
            )

            # 啟用憑證（僅實盤下單必要）
            if not self.paper_trading and config.SHIOAJI_CA_PATH:
                self.api.activate_ca(
                    ca_path=config.SHIOAJI_CA_PATH,
                    ca_passwd=config.SHIOAJI_CA_PASSWORD,
                    person_id=config.SHIOAJI_PERSON_ID,
                )

            # 取得小台近月連續合約（MXFR1）
            futures_group = self.api.Contracts.Futures[config.FUTURES_CODE]
            self.contract = getattr(futures_group, f"{config.FUTURES_CODE}R1")
            self.connected = True

            mode_text = "🧪 模擬環境 (Simulation)" if use_sim else "🔥 實盤環境 (Live)"
            print(f"[BrokerAPI] ✅ 已連接永豐金 API {mode_text}，合約: {self.contract.code}")
            return True

        except Exception as e:
            print(f"[BrokerAPI] ⚠️ Shioaji 連線失敗 ({e})")
            if self.paper_trading:
                print("[BrokerAPI] 🧪 降級為本地 Parquet 5 分 K 資料模式。")
                self.api = None
                self.connected = True
                return True
            return False

    def disconnect(self):
        """斷開連線。"""
        if self.api:
            try:
                self.api.logout()
            except Exception:
                pass
        self.connected = False
        print("[BrokerAPI] 已斷開連線。")

    # ═══════════════════════════════════════════════════════════
    # 行情數據（5 分 K 即時聚合 + 日夜盤全時段）
    # ═══════════════════════════════════════════════════════════

    def get_5min_kbars(self, bars: int = 300) -> pd.DataFrame:
        """
        取得最新已完成之 5 分鐘 K 線（包含日盤與夜盤）。
        1. 若已連接 Shioaji API，抓取近 7 天 1 分 K 即時聚合成 5 分 K，並剔除尚未收盤的當根 K 棒。
        2. 若離線或處於休市測試，自動讀取本地 data/mxf_5min.parquet 並與即時 K 棒拼接。
        """
        local_df = pd.DataFrame()
        try:
            from core.download_kbar import load_5min_data
            local_df = load_5min_data()
        except Exception:
            pass

        live_df = pd.DataFrame()
        if self.api and self.contract:
            try:
                now = datetime.now()
                start_str = (now - timedelta(days=7)).strftime("%Y-%m-%d")
                end_str = (now + timedelta(days=1)).strftime("%Y-%m-%d")
                kbars = self.api.kbars(
                    contract=self.contract,
                    start=start_str,
                    end=end_str,
                )
                df_1m = pd.DataFrame({**kbars})
                if not df_1m.empty and "ts" in df_1m.columns:
                    df_1m["datetime"] = pd.to_datetime(df_1m["ts"])
                    df_1m = df_1m.sort_values("datetime").reset_index(drop=True)
                    df_1m["date"] = df_1m["datetime"].dt.date

                    df_5m = (
                        df_1m.set_index("datetime")
                        .groupby("date")
                        .resample("5min")
                        .agg({
                            "Open": "first",
                            "High": "max",
                            "Low": "min",
                            "Close": "last",
                            "Volume": "sum",
                        })
                        .dropna()
                        .reset_index()
                    )
                    df_5m = df_5m.drop(columns=["date"]).rename(columns={"datetime": "Date"}).set_index("Date")

                    # 剔除當前尚未收完的 5 分鐘 K 棒（避免盤中未收線假突破）
                    current_bucket = pd.Timestamp(now).floor("5min")
                    df_5m = df_5m[df_5m.index < current_bucket]
                    live_df = df_5m
            except Exception as e:
                print(f"[BrokerAPI] ⚠️ 即時 5 分 K 抓取失敗，使用本地快取: {e}")

        if not local_df.empty and not live_df.empty:
            combined = pd.concat([local_df, live_df])
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
            return combined.iloc[-bars:] if len(combined) > bars else combined
        elif not live_df.empty:
            return live_df.iloc[-bars:] if len(live_df) > bars else live_df
        elif not local_df.empty:
            return local_df.iloc[-bars:] if len(local_df) > bars else local_df

        return pd.DataFrame()

    def get_historical_data(self, days: int = 120) -> pd.DataFrame:
        """相容舊介面：預設回傳 5 分 K 資料。"""
        if getattr(config, "TIMEFRAME", "5min") in ("5min", "5T"):
            return self.get_5min_kbars(bars=max(days * 5, config.WARMUP_BARS))
        try:
            from core.download_data import load_local_data
            df = load_local_data()
            return df.iloc[-days:] if len(df) > days else df
        except Exception as e:
            print(f"[BrokerAPI] 本地數據讀取失敗: {e}")
            return pd.DataFrame()

    def get_current_price(self) -> float:
        """取得目前最新價格。"""
        if self.api and self.contract:
            try:
                snapshot = self.api.snapshots([self.contract])
                if snapshot and snapshot[0].close > 0:
                    return float(snapshot[0].close)
            except Exception as e:
                print(f"[BrokerAPI] 取得即時報價失敗: {e}")

        df = self.get_5min_kbars(bars=5)
        if not df.empty:
            return float(df["Close"].iloc[-1])
        return 0.0

    # ═══════════════════════════════════════════════════════════
    # 下單
    # ═══════════════════════════════════════════════════════════

    def place_order(self, direction: int, contracts: int = 1,
                    price: Optional[float] = None) -> Dict:
        """下單（1=做多, -1=做空）。"""
        current_price = price or self.get_current_price()
        action = "Buy" if direction == 1 else "Sell"

        if self.paper_trading:
            return self._paper_order(direction, contracts, current_price)

        if not self.api or not self.contract:
            return {"success": False, "order_id": "", "price": 0, "msg": "API 未連接"}

        try:
            order = self.api.Order(
                price=0,
                quantity=contracts,
                action=sj.constant.Action.Buy if direction == 1 else sj.constant.Action.Sell,
                price_type=sj.constant.FuturesPriceType.MKT,
                order_type=sj.constant.OrderType.IOC,
            )
            trade = self.api.place_order(self.contract, order)
            return {
                "success": True,
                "order_id": str(trade.order.id),
                "price": current_price,
                "msg": f"實盤下單成功: {action} {contracts}口 @ ≈{current_price:.0f}",
            }
        except Exception as e:
            return {"success": False, "order_id": "", "price": 0, "msg": f"下單失敗: {e}"}

    def _paper_order(self, direction: int, contracts: int, price: float) -> Dict:
        """模擬下單（Paper Trading）。"""
        action = "Buy" if direction == 1 else "Sell"
        order_id = f"PAPER_{datetime.now().strftime('%H%M%S%f')}"

        trade_record = {
            "order_id": order_id,
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "direction": direction,
            "contracts": contracts,
            "price": price,
            "mode": "paper",
        }
        self._paper_trades.append(trade_record)
        self._save_trade_log(trade_record)

        return {
            "success": True,
            "order_id": order_id,
            "price": price,
            "msg": f"🧪 模擬下單: {action} {contracts}口 @ {price:.0f}",
        }

    # ═══════════════════════════════════════════════════════════
    # 部位查詢
    # ═══════════════════════════════════════════════════════════

    def get_position(self) -> Dict:
        """查詢目前持倉。保證 contracts==0 時 direction 一律為 0。"""
        _EMPTY = {"direction": 0, "contracts": 0, "entry_price": 0.0}

        if self.paper_trading:
            direction = self._paper_position or 0
            contracts = abs(direction)
            if contracts == 0:
                direction = 0
            return {
                "direction": 1 if direction > 0 else (-1 if direction < 0 else 0),
                "contracts": contracts,
                "entry_price": self._paper_entry_price if contracts > 0 else 0.0,
            }

        if not self.api:
            return _EMPTY.copy()

        try:
            positions = self.api.list_positions(self.api.futopt_account)
            if positions is None:
                return _EMPTY.copy()
            for pos in positions:
                if config.FUTURES_CODE in str(getattr(pos, "code", "")):
                    quantity = getattr(pos, "quantity", 0) or 0
                    if quantity == 0:
                        return _EMPTY.copy()
                    raw_dir = getattr(pos, "direction", None)
                    direction = 1 if raw_dir == "Buy" else (-1 if raw_dir == "Sell" else 0)
                    if direction == 0:
                        return _EMPTY.copy()
                    raw_price = getattr(pos, "price", 0)
                    entry_price = float(raw_price) if raw_price else 0.0
                    return {
                        "direction": direction,
                        "contracts": quantity,
                        "entry_price": entry_price,
                    }
        except Exception as e:
            print(f"[BrokerAPI] 部位查詢失敗: {e}")

        return _EMPTY.copy()

    def update_paper_position(self, direction: int, entry_price: float):
        """更新模擬盤部位。"""
        self._paper_position = direction
        self._paper_entry_price = entry_price if direction != 0 else 0.0

    def _save_trade_log(self, record: Dict):
        """將交易紀錄追加到日誌檔案。"""
        try:
            logs = []
            if os.path.exists(self._trade_log_path):
                with open(self._trade_log_path, "r", encoding="utf-8") as f:
                    logs = json.load(f)
            logs.append(record)
            with open(self._trade_log_path, "w", encoding="utf-8") as f:
                json.dump(logs, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[BrokerAPI] 日誌寫入失敗: {e}")
