"""
自動化交易系統 — 設定檔
========================
集中管理所有參數，從環境變數讀取敏感資訊。
"""
import os
from dotenv import load_dotenv

# 支援本地 .env 與 intraday-scanner 共用憑證
_env_paths = [
    os.path.join(os.path.dirname(__file__), ".env"),
    r"D:\Antigravity\intraday-scanner\.env",
]
for _p in _env_paths:
    if os.path.exists(_p):
        load_dotenv(_p)
        break

# ═══════════════════════════════════════════════════════════════
# 券商 API 設定（永豐金 Shioaji）
# ═══════════════════════════════════════════════════════════════
SHIOAJI_API_KEY = os.getenv("SHIOAJI_API_KEY", os.getenv("API_KEY", ""))
SHIOAJI_SECRET_KEY = os.getenv("SHIOAJI_SECRET_KEY", os.getenv("SECRET_KEY", ""))
SHIOAJI_PERSON_ID = os.getenv("SHIOAJI_PERSON_ID", "")  # 身分證字號
SHIOAJI_CA_PATH = os.getenv("SHIOAJI_CA_PATH", "")  # 憑證路徑
SHIOAJI_CA_PASSWORD = os.getenv("SHIOAJI_CA_PASSWORD", "")  # 憑證密碼
# 設為 true 使用模擬模式 (sj.Shioaji(simulation=True))
SHIOAJI_SIMULATION = os.getenv("SHIOAJI_SIMULATION", os.getenv("SIMULATION", "True")).lower() == "true"

# ═══════════════════════════════════════════════════════════════
# Telegram 通知
# ═══════════════════════════════════════════════════════════════
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ═══════════════════════════════════════════════════════════════
# 交易商品設定
# ═══════════════════════════════════════════════════════════════
FUTURES_CODE = "MXF"           # 小台指期貨代碼
CONTRACT_MULTIPLIER = 50       # 每點價值 (TWD)
COMMISSION_PER_SIDE = 20       # 單邊手續費 (TWD)

# ═══════════════════════════════════════════════════════════════
# 策略設定（5 分 K 日夜盤連續波段 + 結算日日盤不交易）
# ═══════════════════════════════════════════════════════════════
# 主策略：通過 Walk-Forward (2025->2026) 與 10,000 次 Bootstrap/MCPT 壓力檢定之雙強策略
PRIMARY_STRATEGY = "💎 Swing + Keltner Hybrid (Causal)"
SWING_LEN = 5                  # 波段高低點確認長度 (Causal)
EMA_LEN = 60                   # 肯特納通道 EMA 與 ATR 週期 (60根 5分K = 5小時跨日夜盤中軸)
ATR_MULT = 2.0                 # 肯特納通道 ATR 倍數

# 策略所需的歷史 5 分 K 棒數量（確保指標與狀態機有足夠暖機資料）
WARMUP_BARS = 300
TIMEFRAME = "5min"
# 結算日換倉規則："skip_settlement_day" (每月第三個週三 05:00~15:00 日盤完全不交易並空手)
ROLLOVER_MODE = "skip_settlement_day"

# ═══════════════════════════════════════════════════════════════
# 風控參數
# ═══════════════════════════════════════════════════════════════
MAX_CONTRACTS = 1              # 單策略最大持倉口數
MAX_TOTAL_CONTRACTS = 3        # 所有策略合計最大持倉口數上限
MAX_DAILY_LOSS = 10_000        # 當日最大虧損 (TWD)，觸發後停止交易
MAX_CONSECUTIVE_LOSSES = 5     # 連續虧損次數上限
INITIAL_CAPITAL = 500_000      # 初始基準資金
RISK_GUARD_INTERVAL = 3        # 停損高頻迴圈檢查間隔 (秒)

# ═══════════════════════════════════════════════════════════════
# 交易時段（台灣時間，涵蓋日盤 08:45~13:45 與夜盤 15:00~05:00）
# ═══════════════════════════════════════════════════════════════
TRADING_START_HOUR = 0
TRADING_START_MINUTE = 0
TRADING_END_HOUR = 23
TRADING_END_MINUTE = 59

# ═══════════════════════════════════════════════════════════════
# 系統設定
# ═══════════════════════════════════════════════════════════════
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
PAPER_TRADING = True  # True = 模擬盤（不實際下單）, False = 實盤
SIGNAL_CHECK_INTERVAL = 15  # 每幾秒檢查一次 5 分 K 信號 (秒)
