"""
自動化交易系統 — 券商 API 抽象層
===================================
封裝永豐金 Shioaji API，統一介面。
支援 Paper Trading 模式（不實際下單）。
"""
import os
import sys
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import config

# 嘗試載入 Shioaji（若未安裝則降級為純模擬模式）
try:
    import shioaji as sj
    HAS_SHIOAJI = True
except ImportError:
    HAS_SHIOAJI = False
    print("[BrokerAPI] ⚠️ shioaji 未安裝，僅支援 Paper Trading 模式。")
    print("            安裝方式：pip install shioaji")


class BrokerAPI:
    """
    券商 API 抽象層

    支援兩種模式：
    1. Paper Trading（模擬盤）：不連接真實 API，模擬下單
    2. Live Trading（實盤）：透過 Shioaji 連接永豐金下單

    Parameters
    ----------
    paper_trading : bool  是否使用模擬盤模式
    """

    def __init__(self, paper_trading: bool = True):
        self.paper_trading = paper_trading
        self.api = None
        self.connected = False
        self.contract = None  # 期貨合約物件

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
        """連接券商 API。"""
        if self.paper_trading:
            print("[BrokerAPI] 🧪 Paper Trading 模式，無需連接券商。")
            self.connected = True
            return True

        if not HAS_SHIOAJI:
            print("[BrokerAPI] ❌ 實盤模式需要安裝 shioaji。")
            return False

        try:
            self.api = sj.Shioaji()
            self.api.login(
                api_key=config.SHIOAJI_API_KEY,
                secret_key=config.SHIOAJI_SECRET_KEY,
            )

            # 啟用憑證（下單必要）
            if config.SHIOAJI_CA_PATH:
                self.api.activate_ca(
                    ca_path=config.SHIOAJI_CA_PATH,
                    ca_passwd=config.SHIOAJI_CA_PASSWORD,
                    person_id=config.SHIOAJI_PERSON_ID,
                )

            # 取得小台近月合約
            self.contract = self.api.Contracts.Futures[config.FUTURES_CODE].MXF
            self.connected = True
            print(f"[BrokerAPI] ✅ 已連接永豐金 API，合約: {self.contract}")
            return True

        except Exception as e:
            print(f"[BrokerAPI] ❌ 連接失敗: {e}")
            return False

    def disconnect(self):
        """斷開連線。"""
        if self.api and not self.paper_trading:
            try:
                self.api.logout()
            except Exception:
                pass
        self.connected = False
        print("[BrokerAPI] 已斷開連線。")

    # ═══════════════════════════════════════════════════════════
    # 行情數據
    # ═══════════════════════════════════════════════════════════

    def get_historical_data(self, days: int = 120) -> pd.DataFrame:
        """
        取得歷史日線數據。

        Paper Trading 模式下從本地 CSV 讀取（與回測共用數據源）。
        實盤模式下從 Shioaji API 取得或從 yfinance 取得。

        Parameters
        ----------
        days : int  需要幾天的歷史數據

        Returns
        -------
        pd.DataFrame : 含 Open, High, Low, Close, Volume 的 DataFrame
        """
        # 不論模式，都優先從本地 CSV 讀取（確保與回測一致）
        try:
            from download_data import load_local_data
            df = load_local_data()
            # 取最近 N 天
            if len(df) > days:
                df = df.iloc[-days:]
            return df
        except Exception as e:
            print(f"[BrokerAPI] 本地數據讀取失敗: {e}")

        # 備用方案：從 yfinance 抓取
        try:
            import yfinance as yf
            ticker = yf.Ticker("^TWII")
            df = ticker.history(period=f"{days}d")
            df = df[["Open", "High", "Low", "Close", "Volume"]]
            return df
        except Exception as e:
            print(f"[BrokerAPI] yfinance 數據抓取失敗: {e}")
            return pd.DataFrame()

    def get_current_price(self) -> float:
        """
        取得目前最新價格。

        Paper Trading 模式下取本地數據的最後收盤價。
        實盤模式下取即時報價。
        """
        if self.paper_trading or not self.api:
            df = self.get_historical_data(days=5)
            if not df.empty:
                return float(df["Close"].iloc[-1])
            return 0.0

        # 實盤：取即時報價
        try:
            snapshot = self.api.snapshots([self.contract])
            if snapshot:
                return float(snapshot[0].close)
        except Exception as e:
            print(f"[BrokerAPI] 取得即時報價失敗: {e}")
        return 0.0

    # ═══════════════════════════════════════════════════════════
    # 下單
    # ═══════════════════════════════════════════════════════════

    def place_order(self, direction: int, contracts: int = 1,
                    price: Optional[float] = None) -> Dict:
        """
        下單。

        Parameters
        ----------
        direction : int  1=做多, -1=做空
        contracts : int  口數
        price : float  限價（None=市價）

        Returns
        -------
        dict : {"success": bool, "order_id": str, "price": float, "msg": str}
        """
        current_price = price or self.get_current_price()
        action = "Buy" if direction == 1 else "Sell"

        if self.paper_trading:
            return self._paper_order(direction, contracts, current_price)

        # 實盤下單
        if not self.api or not self.contract:
            return {"success": False, "order_id": "", "price": 0,
                    "msg": "API 未連接"}

        try:
            order = self.api.Order(
                price=0,  # 0 = 市價
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
            return {"success": False, "order_id": "", "price": 0,
                    "msg": f"下單失敗: {e}"}

    def _paper_order(self, direction: int, contracts: int,
                     price: float) -> Dict:
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
        """
        查詢目前持倉。

        Returns
        -------
        dict : {"direction": int, "contracts": int, "entry_price": float}
        """
        if self.paper_trading:
            return {
                "direction": self._paper_position,
                "contracts": abs(self._paper_position),
                "entry_price": self._paper_entry_price,
            }

        # 實盤查詢
        if not self.api:
            return {"direction": 0, "contracts": 0, "entry_price": 0}

        try:
            positions = self.api.list_positions(self.api.futopt_account)
            for pos in positions:
                if config.FUTURES_CODE in str(pos.code):
                    direction = 1 if pos.direction == "Buy" else -1
                    return {
                        "direction": direction,
                        "contracts": pos.quantity,
                        "entry_price": float(pos.price),
                    }
        except Exception as e:
            print(f"[BrokerAPI] 部位查詢失敗: {e}")

        return {"direction": 0, "contracts": 0, "entry_price": 0}

    def update_paper_position(self, direction: int, entry_price: float):
        """更新模擬盤部位。"""
        self._paper_position = direction
        self._paper_entry_price = entry_price if direction != 0 else 0.0

    # ═══════════════════════════════════════════════════════════
    # 工具
    # ═══════════════════════════════════════════════════════════

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
