"""
自動化交易系統 — 設定檔
========================
集中管理所有參數，從環境變數讀取敏感資訊。
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ═══════════════════════════════════════════════════════════════
# 券商 API 設定（永豐金 Shioaji）
# ═══════════════════════════════════════════════════════════════
SHIOAJI_API_KEY = os.getenv("SHIOAJI_API_KEY", "")
SHIOAJI_SECRET_KEY = os.getenv("SHIOAJI_SECRET_KEY", "")
SHIOAJI_PERSON_ID = os.getenv("SHIOAJI_PERSON_ID", "")  # 身分證字號
SHIOAJI_CA_PATH = os.getenv("SHIOAJI_CA_PATH", "")  # 憑證路徑
SHIOAJI_CA_PASSWORD = os.getenv("SHIOAJI_CA_PASSWORD", "")  # 憑證密碼
# 設為 true 使用模擬模式 (sj.Shioaji(simulation=True))
SHIOAJI_SIMULATION = os.getenv("SHIOAJI_SIMULATION", "True").lower() == "true"

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
# 策略設定
# ═══════════════════════════════════════════════════════════════
# 主策略（回測中 Sharpe 最高、MDD 最低的策略）
PRIMARY_STRATEGY = "Session Edge Ensemble"
# 策略所需的歷史 K 棒數量（確保指標有足夠的暖機資料）
WARMUP_BARS = 60
# 使用日線 (D) 或分鐘線
TIMEFRAME = "15T"

# ═══════════════════════════════════════════════════════════════
# 風控參數
# ═══════════════════════════════════════════════════════════════
MAX_CONTRACTS = 1              # 單策略最大持倉口數
MAX_TOTAL_CONTRACTS = 3        # 所有策略合計最大持倉口數上限 (Swing 2口 + Edge 1口)
MAX_DAILY_LOSS = 10_000        # 當日最大虧損 (TWD)，觸發後停止交易
MAX_CONSECUTIVE_LOSSES = 5     # 連續虧損次數上限
INITIAL_CAPITAL = 200_000      # 帳戶建議最低資金
RISK_GUARD_INTERVAL = 3        # 停損高頻迴圈檢查間隔 (秒)

# ═══════════════════════════════════════════════════════════════
# 交易時段（台灣時間）
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
SIGNAL_CHECK_INTERVAL = 15  # 每幾秒檢查一次信號 (秒)
