# Session Edge Ensemble 策略檢測報告

產出日期：2026-05-18  
資料範圍：2025-01-01 至 2026-05-15  
商品與週期：小台指期 MXF，5 分 K  
回測成本：手續費 20 元/邊，滑價 1 點/邊，乘數 50，初始資金 500,000 元

---

## 1. 策略摘要

`Session Edge Ensemble` 是一組盤中時段型 ensemble 策略，不使用未來價格，也不依賴複雜指標參數。策略核心假設是：小台指期在特定星期與交易時段存在可重複的延續或反轉行為，因此用多個低相關的 session sleeve 組合成單一訊號。

策略訊號位置：

- 程式：`strategies.py`
- 函式：`session_edge_ensemble(df)`
- 註冊名稱：`Session Edge Ensemble`

訊號定義：

- `1`：做多
- `-1`：做空
- `0`：空手

---

## 2. 策略規則

策略採用 pandas weekday 定義：Monday=0，Tuesday=1，Wednesday=2，Thursday=3，Friday=4。

| 星期 | 時段 | 方向 | 設計意圖 |
|---|---:|:---:|---|
| Monday | 18:10-22:10 | 多 | 週一夜盤延續 |
| Tuesday | 00:00-09:40 | 多 | 週二夜盤至開盤延續 |
| Tuesday | 09:40-13:40 | 空 | 週二日盤反轉 |
| Tuesday | 15:00-17:00 | 多 | 週二下午盤延續 |
| Wednesday | 08:50-10:25 | 多 | 週三開盤延續 |
| Wednesday | 15:15-18:15 | 多 | 週三下午盤延續 |
| Thursday | 02:45-04:15 | 多 | 週四清晨盤延續 |
| Friday | 09:10-09:25 | 空 | 週五開盤短線 fade |
| Friday | 09:30-13:30 | 多 | 週五日盤延續 |

此策略屬於「盤中季節性/微結構」策略，而不是價格型技術指標策略。因此它的主要風險不是指標失效，而是交易時段結構改變，例如夜盤流動性、結算制度、主力交易節奏或市場波動型態改變。

---

## 3. 一般 5 分 K 回測結果

測試腳本：`run_backtest_5min.py`  
測試區間：2026-01-01 至 2026-05-15

| 策略 | Return | Sharpe | MDD | Win Rate | Trades | Profit Factor | Final Equity |
|---|---:|---:|---:|---:|---:|---:|---:|
| Session Edge Ensemble | 67.32% | 1.949 | -11.88% | 53.2% | 156 | 1.64 | 836,610 |
| Swing High Low | 50.74% | 0.754 | -25.18% | 38.5% | 371 | 1.15 | 753,710 |
| Keltner Channel | 46.08% | 0.685 | -26.54% | 38.6% | 249 | 1.12 | 730,420 |

結論：`Session Edge Ensemble` 在 2026 OOS 排名第一，報酬、Sharpe、MDD 與 Profit Factor 都優於原本表現較好的 `Swing High Low` 和 `Keltner Channel`。

---

## 4. Bonferroni 檢測

測試腳本：`advanced_validation.py`  
樣本內區間：2025-01-01 至 2025-12-31  
策略總數：17  
原始顯著水準：0.05  
Bonferroni 校正後顯著水準：0.002941  
最低 t-statistic 門檻：2.754

| 策略 | IS Return | IS Sharpe | p-value | 是否通過 |
|---|---:|---:|---:|:---:|
| Session Edge Ensemble | 131.88% | 3.004 | 0.001330 | 通過 |
| Swing High Low | 104.57% | 1.042 | 0.148711 | 未通過 |
| SuperTrend | 42.45% | 0.423 | 0.336105 | 未通過 |

結論：新策略在納入 17 組策略的多重比較校正後仍通過 Bonferroni 檢測，代表 2025 樣本內績效不是一般資料挖礦下容易出現的偶然結果。

---

## 5. Walk-Forward 檢測

測試方式：使用 2025 作為樣本內挑選期，2026 作為樣本外盲測期。

| 策略 | IS Return | IS Sharpe | OOS Return | OOS Sharpe | OOS MDD | Bonferroni |
|---|---:|---:|---:|---:|---:|:---:|
| Session Edge Ensemble | 131.88% | 3.004 | 67.32% | 1.949 | -11.88% | 通過 |
| Swing High Low | 104.57% | 1.042 | 50.74% | 0.754 | -25.18% | 未通過 |
| SuperTrend | 42.45% | 0.423 | -58.86% | -0.875 | -85.70% | 未通過 |

結論：新策略不只在 2025 樣本內有效，也在 2026 樣本外保留正報酬與正 Sharpe，通過 Walk-Forward 檢測。

---

## 6. Bootstrap CI 檢測

測試腳本：`monte_carlo_test.py`  
測試區間：2026-01-01 至 2026-05-15  
重抽樣次數：10,000 次  
交易次數：156 次

| 項目 | 數值 |
|---|---:|
| 原始總報酬率 | 67.32% |
| 最大單筆獲利 | 63,510 TWD |
| 最大單筆虧損 | -52,190 TWD |
| Bootstrap 95% CI | [5.07%, 131.42%] |
| 最差 5% 報酬率下界 | 14.36% |

結論：Bootstrap 95% 信心區間下界仍大於 0，且最差 5% 報酬率下界為正，表示策略不是只靠少數極端獲利交易撐起來。

---

## 7. MCPT 檢測

測試腳本：`monte_carlo_test.py`  
測試方式：將 2026 交易損益順序隨機排列 10,000 次，觀察最大回撤是否落在排列分布的脆弱尾端。

| 項目 | 數值 |
|---|---:|
| 原始 MDD | -11.88% |
| 隨機洗牌平均 MDD | -13.64% |
| 隨機排列比原始 MDD 更差的比例 | 61.3% |
| MDD fragile-tail p-value | 0.387 |

結論：原始 MDD 未落在排列分布最脆弱 5% 尾端，代表交易順序沒有呈現異常連虧集中風險，通過 MCPT 壓力測試。

---

## 8. 四項檢測總表

| 檢測方法 | 門檻/重點 | Session Edge Ensemble 結果 | 判定 |
|---|---|---:|:---:|
| Bonferroni | p-value < 0.002941 | 0.001330 | 通過 |
| Walk-Forward | 2026 OOS 維持正績效 | Return 67.32%，Sharpe 1.949 | 通過 |
| Bootstrap CI | 下界需大於 0 | 95% CI [5.07%, 131.42%] | 通過 |
| MCPT | 不落入 MDD 脆弱尾端 | fragile-tail p-value 0.387 | 通過 |

---

## 9. 風險與使用建議

1. 這是一個時段型策略，最需要監控的是「時段 edge 是否衰退」，建議每週或每月重新統計各 sleeve 的獨立績效。
2. 若未來夜盤流動性或台指期交易制度改變，需重新跑 Walk-Forward，不應直接沿用。
3. 目前回測成本已包含手續費與滑價，但實盤仍應限制最大口數與單日虧損。
4. 建議先用 paper trading 或模擬單觀察至少 20 至 30 筆交易，再考慮小部位實盤。
5. 若要再強化，可加入每個 sleeve 的 rolling 停用條件，例如最近 20 筆該 sleeve Profit Factor 低於 1 時暫停。

---

## 10. 結論

`Session Edge Ensemble` 是目前專案中第一組同時通過 Bonferroni、Walk-Forward、Bootstrap CI 與 MCPT 的策略。它在 2025 樣本內具有統計顯著性，並在 2026 樣本外維持高於既有策略的報酬與風險調整後績效。

此策略可以作為下一階段 paper trading 的候選，但不建議直接以滿部位上線。較穩健的做法是先監控每個 session sleeve 的持續性，再逐步加入風控與停用機制。
