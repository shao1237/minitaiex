# 📈 Mini TAIEX Futures Auto-Trader & Backtest System (小台指期量化回測與自動交易系統)

本專案為針對**台灣小型台指期貨 (MXF)** 設計的量化研究、嚴格統計檢定與全自動交易系統（串接永豐金 **Shioaji API** 與 **Telegram 即時推播**）。

系統核心採用通過 **Walk-Forward 跨年度推進檢驗 (2025 IS → 2026 OOS)**、**參數高原測試** 以及 **10,000 次蒙地卡羅排列 / Bootstrap 重抽樣壓力測試** 之無未來函數策略：**`💎 Swing + Keltner Hybrid (Causal)`**，並支援**日盤 + 夜盤跨時段連續持倉**與**每月第三個週三結算日日盤完全不交易 (`skip_settlement_day`)** 保護機制。

---

## ✨ 核心特色

1. **雙強結合策略 (`💎 Swing + Keltner Hybrid (Causal)`)**
   - **100% 無未來函數 (Causal Pivot Confirmation)**：嚴格在右方 `swing_len` 根 K 棒收線後才確認波段高低點，杜絕前視偏差 (Look-ahead Bias)。
   - **波動率與結構雙重突破確認 (`breakout_confirm`)**：
     - **多單進場 / 反手**：5 分 K 收盤價同時突破 Keltner Channel 上軌 (`EMA_60 + 2.0 * ATR_60`) 與最近已確認之 Causal Swing High (`SW=5`)。
     - **空單進場 / 反手**：5 分 K 收盤價同時跌破 Keltner Channel 下軌 (`EMA_60 - 2.0 * ATR_60`) 與最近已確認之 Causal Swing Low (`SW=5`)。
     - **狀態鎖定 (State Machine Latch)**：趨勢未反轉前跨日夜盤持續抱單，避免盤整區洗盤與日盤收盤強平造成的摩擦成本損耗。
2. **日夜盤連續持倉 + 結算日保護 (`skip_settlement_day`)**
   - 納入台指期日盤 (`08:45 ~ 13:45`) 與夜盤 (`15:00 ~ 次日 05:00`) 完整 5 分鐘 K 線。
   - **換月結算保護**：自動偵測每月第三個星期三（台指期結算日），於 **`05:00`（夜盤收盤）至 `15:00`（新合約夜盤開盤前）強制平倉並禁止新開倉**，避開結算日拉抬壓低噪音與新舊合約價差跳空。
3. **實盤級穩定性與風控保護**
   - **K 棒收線防抖動 (Bar Debounce)**：自動剔除尚未收完的當根 K 棒，且同一根 5 分 K 時間戳只處理一次。
   - **原子化狀態持久化 (`logs/position_state.json`)**：程式重啟或斷線重連時自動載入既有部位，並與券商實盤部位進行雙向對帳。
   - **多層風控 ([`RiskManager`](core/risk_manager.py))**：支援單筆固定點數停損、當日最大虧損熔斷、連續虧損次數限制與總權益回撤停機保護。

---

## 📊 策略量化檢定摘要 (`breakout_confirm`: `EMA=60, Mult=2.0, SW=5`)

> 回測設定：小台指 `MXF`（每點 50 元台幣）、初始資金 `500,000` 元、雙邊手續費 `20` 元/口、滑價 `1` 點/邊（單趟總摩擦成本 `140` 元/口），套用 `skip_settlement_day` 規則。

| 檢驗期間 / 項目 | 總報酬率 | 真實日年化 Sharpe | 最大回撤 (MDD) | 獲利因子 (PF) | 交易筆數 | Bootstrap 5% 下限 (獲利機率) | MCPT 脆弱尾端 $p$-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2025 In-Sample** (`2025/01 ~ 2025/12`) | **+63.71%** | **+1.102** | -27.21% | 1.13 | 561 筆 | +0.62% (95.14%) | $p = 0.7204$ |
| **2026 Out-of-Sample** (`2026/01 ~ 2026/10/16`) | **+151.78%** | **+1.617** | -38.76% | 1.23 | 466 筆 | +18.43% (96.95%) | $p = 0.6778$ |
| **2025–2026 全期合併** (`451` 交易日) | **+213.64%** | **+1.327** | **-27.24%** | **1.16** | **1,028 筆** | **+14.69% (96.21%)** | **$p = 0.7806$** |

- **參數高原穩健度**：在 `32` 組 `(swing_len, ema_len, atr_mult)` 網格中，2025 IS 獲利比例達 **93.8% (30/32)**，2026 OOS 盲測獲利比例達 **100.0% (32/32)**。

---

## 📂 專案目錄結構

```text
mini_taiex_backtest/
├── config.py                       # 全域交易設定、策略參數、結算日規則與風控門檻
├── auto_trader.py                  # 主控入口 (1)：5 分 K 雙強波段自動交易系統 (預設推薦)
├── swing_trader.py                 # 主控入口 (2)：15 分 K 雙向/多頭波段自動交易系統
├── core/                           # 核心模組庫
│   ├── strategies.py               # 策略庫 (含 💎 Swing + Keltner Hybrid 與 16 種技術指標策略)
│   ├── strategy_engine.py          # 5 分 K 即時訊號引擎與結算日禁制時段判斷
│   ├── intraday_backtester.py      # 5 分 K 回測引擎 (支援跨日夜盤持倉與結算日轉倉規則)
│   ├── swing_kc_backtest.py        # 15 分 K 波段回測與實盤訊號運算核心
│   ├── broker_api.py               # 永豐金 Shioaji API 封裝 (即時 5m/15m K 線聚合、下單、模擬盤)
│   ├── download_kbar.py            # Shioaji 歷史 1 分 K 逐日下載與 5 分 K Parquet 聚合工具
│   ├── aggregate_15min.py          # 5 分 K 轉 15 分 K 聚合工具
│   ├── download_data.py            # 加權指數 (^TWII) 日線下載工具
│   ├── risk_manager.py             # 風險管理模組 (單筆停損、單日虧損上限、連虧熔斷、MDD 保護)
│   └── notifier.py                 # Telegram 即時通知模組
├── tests/                          # 量化回測與統計檢定腳本
│   ├── run_backtest_5min.py        # 5 分 K 全策略回測排名
│   ├── advanced_validation.py      # Walk-Forward (2025 IS / 2026 OOS) 與 Bonferroni 多重比較校正
│   ├── test_swing_kc_5min.py       # Swing + Keltner 雙強策略參數網格掃描 (Grid Search)
│   ├── monte_carlo_test.py         # 10,000 次 Bootstrap 重抽樣與 MCPT 排列壓力測試
│   ├── full_spectrum_validation.py # 15 分 K 策略全光譜檢驗
│   ├── grid_search.py              # 15 分 K 參數網格搜尋
│   └── run_swing_strategy.py       # 15 分 K 策略單獨回測腳本
├── reports/                        # 回測輸出報表 (CSV / Markdown / 績效圖表)
├── data/                           # 本地歷史 K 線快取 (kbar_1m/*.csv, mxf_5min.parquet) [Git Ignored]
├── logs/                           # 部位狀態持久化與每日交易紀錄 JSON
└── Archived/                       # 歷史實驗與舊版腳本封存區
```

---

## 🛠️ 環境安裝與設定

### 1. 安裝相依套件

建議使用 Python 3.10+ 環境：

```bash
pip install shioaji pandas numpy scipy pyarrow python-dotenv requests yfinance
```

### 2. 設定環境變數 (`.env`)

在專案根目錄建立 `.env` 檔案（不會納入 Git 版控），填入永豐金 Shioaji API 金鑰與憑證設定，以及選填的 Telegram 通知設定：

```ini
# 永豐金 Shioaji API 憑證
SHIOAJI_API_KEY=your_api_key_here
SHIOAJI_SECRET_KEY=your_secret_key_here

# 實盤下單所需憑證路徑與密碼 (模擬盤可不填)
SHIOAJI_CA_PATH=C:/ekey/551/YOUR_ID/S/Sinopac.pfx
SHIOAJI_CA_PASSWORD=your_ca_password
SHIOAJI_PERSON_ID=your_person_id

# Telegram 通知設定 (選填)
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

---

## 🚀 常用指令說明

### 1. 下載與更新歷史 K 線資料

從 Shioaji 下載 1 分 K 並自動聚合成 `data/mxf_5min.parquet`（包含日盤與夜盤全時段）：

```bash
# 下載指定區間 1 分 K 並聚合為 5 分 K
python core/download_kbar.py --start 2025-01-01 --end 2026-10-16

# 若 1 分 K CSV 已存在，僅重新執行 5 分 K 聚合
python core/download_kbar.py --aggregate-only
```

### 2. 執行量化回測與統計檢定

```bash
# 1. 執行 2026 年全策略 (含雙強策略) 5 分 K 回測排名
python tests/run_backtest_5min.py

# 2. 執行 2025 IS -> 2026 OOS Walk-Forward 與 Bonferroni 校正檢定
python tests/advanced_validation.py

# 3. 執行 Swing + Keltner 參數高原網格檢驗 (64 組參數)
python tests/test_swing_kc_5min.py

# 4. 執行 10,000 次 Bootstrap 重抽樣與蒙地卡羅排列壓力測試 (MCPT)
python tests/monte_carlo_test.py
```

### 3. 啟動 5 分 K 自動交易系統 ([`auto_trader.py`](auto_trader.py))

```bash
# 單次信號檢查（抓取最新已完成之 5 分 K、印出訊號並結束，適合測試連線）
python auto_trader.py --once

# 啟動 Paper Trading 模擬盤（預設模式，每 60 秒輪詢最新收線之 5 分 K）
python auto_trader.py

# 啟動實盤自動交易（需於 .env 配置完整憑證，啟動前會有 5 秒安全確認倒數）
python auto_trader.py --live
```

---

## 🛡️ 風險管理與注意事項

- **資料同步**：實盤或模擬盤運行時，[`BrokerAPI.get_5min_kbars()`](core/broker_api.py) 會自動抓取近 7 日即時 1 分 K 聚合成 5 分 K（並剔除尚未收線的當根 K 棒），與本地 `data/mxf_5min.parquet` 無縫拼接以確保 `EMA=60` 暖機長度充足。
- **結算日空手時段**：每月第三個週三 `05:00 ~ 15:00` 系統會自動將目標部位設為 `0`（若持有部位則自動平倉），待 `15:00` 新合約夜盤開盤後才恢復訊號交易。
- **風控參數調整**：可於 [`config.py`](config.py) 中調整 `CONTRACTS`（口數）、`STOP_LOSS_POINTS`（單筆硬停損點數）、`MAX_DAILY_LOSS`（單日最大虧損金額）與 `MAX_DRAWDOWN_PCT`（最大回撤百分比）。
