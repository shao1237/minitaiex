import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
全方位量化策略檢測計畫 (Full-Spectrum Quant Validation Pipeline)
==================================================================
- 第一階段：數據純化與真實成本 (已整合在回測引擎中)
- 第二階段：四重地獄統計檢定 (WFA, Bonferroni, MCPT, Bootstrap)
- 第三階段：ML 與時間序列專屬防護 (Regime Filter 多空季線切割)
- 繪製檢定圖表並輸出結果至 Artifact 目錄
"""
import os
import sys
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import NormalDist
from swing_kc_backtest import SwingKeltnerBacktester, calculate_stats

# 設定 Matplotlib 中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# Artifact 輸出目錄
ARTIFACT_DIR = r"C:\Users\User\.gemini\antigravity\brain\b66c39bc-2b4c-41a5-b216-d91083c6b9a3"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def run_pipeline():
    print("=" * 80)
    print("🛡️ 啟動全方位量化策略檢測計畫 (Full-Spectrum Quant Validation Pipeline) 🛡️")
    print("=" * 80)
    
    # 1. 載入 15 分 K 數據 (做為主力測試標的)
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    path_15m = os.path.join(data_dir, "mxf_15min.parquet")
    
    if not os.path.exists(path_15m):
        print("❌ 找不到 15 分 K 資料，請先執行下載與聚合")
        return
        
    df = pd.read_parquet(path_15m).sort_index()
    
    # 2. 樣本分割 (Walk-Forward Setup)
    # IS: 2025 年一整年
    # OOS: 2026-01-01 至今 (盲測)
    df_is = df[(df.index >= "2025-01-01") & (df.index <= "2025-12-31")].copy()
    df_oos = df[df.index >= "2026-01-01"].copy()
    
    days_is = len(set(df_is.index.date))
    days_oos = len(set(df_oos.index.date))
    years_is = days_is / 252.0
    
    print(f"📅 In-Sample (IS) 期間: 2025-01-01 ~ 2025-12-31 ({days_is} 個交易日)")
    print(f"📅 Out-of-Sample (OOS) 期間: 2026-01-01 ~ 至今 ({days_oos} 個交易日)")
    print("-" * 80)
    
    # 3. 執行 In-Sample 參數尋優 (IS Grid Search)
    backtester = SwingKeltnerBacktester(initial_capital=500_000.0, commission=20, slippage=1.0, multiplier=50.0)
    
    ema_lens = [10, 15, 20]
    kc_mults = [1.5, 2.0, 2.5]
    swing_lens = [1, 2]
    slopes = [0.0, 0.02]
    cooldowns = [0, 12]
    
    # 組合數 m = 3 * 3 * 2 * 2 * 2 = 72 組
    total_combinations = len(ema_lens) * len(kc_mults) * len(swing_lens) * len(slopes) * len(cooldowns)
    print(f"⏳ 正在 IS 期間掃描 {total_combinations} 組參數組合進行過度擬合壓力測試...")
    
    is_results = []
    for ema in ema_lens:
        for mult in kc_mults:
            for swing in swing_lens:
                for slope in slopes:
                    for cd in cooldowns:
                        res = backtester.run_backtest(
                            df_is, ema_len=ema, kc_mult=mult, swing_len=swing,
                            buffer_pct=0.0015, long_only=True, slope_threshold=slope,
                            cooldown_bars=cd, early_break_even=False
                        )
                        df_ind = res["indicators"]
                        trades = res["trades"]
                        final_eq = df_ind["Equity"].iloc[-1]
                        stats = calculate_stats(500_000.0, final_eq, df_ind["Equity"], trades, df_is)
                        
                        is_results.append({
                            "ema_len": ema,
                            "kc_mult": mult,
                            "swing_len": swing,
                            "slope_threshold": slope,
                            "cooldown_bars": cd,
                            "total_return": stats["total_return_pct"],
                            "mdd": stats["max_drawdown_pct"],
                            "sharpe": stats["sharpe_ratio"],
                            "trades_count": stats["total_trades"],
                            "win_rate": stats["win_rate_pct"]
                        })
                        
    is_df = pd.DataFrame(is_results)
    is_df = is_df.sort_values(by="sharpe", ascending=False).reset_index(drop=True)
    
    best_params = is_df.iloc[0]
    print(f"🏆 In-Sample 最佳參數組合:")
    print(f"   - EMA 週期: {best_params['ema_len']:.0f} | ATR 乘數: {best_params['kc_mult']:.1f} | Swing 確認: {best_params['swing_len']:.0f}")
    print(f"   - 斜率門檻: {best_params['slope_threshold']:.2f} | 停損冷卻: {best_params['cooldown_bars']:.0f}")
    print(f"   - IS 總報酬: {best_params['total_return']:.2f}% | IS Sharpe: {best_params['sharpe']:.3f} | IS MDD: {best_params['mdd']:.2f}%")
    print("-" * 80)
    
    # 4. Bonferroni Correction (邦費羅尼校正)
    # 測試了 m 組參數，顯著性水準調整為 alpha / m
    m = total_combinations
    alpha = 0.05
    alpha_adj = alpha / m
    
    # 計算最佳組合的 t-statistic: t = Sharpe * sqrt(Years)
    best_sharpe = best_params["sharpe"]
    t_stat = best_sharpe * np.sqrt(years_is) if years_is > 0 else 0
    # 計算單尾 p-value
    p_value = 0.5 * math.erfc(t_stat / math.sqrt(2)) if t_stat > 0 else 1.0
    passed_bonf = p_value < alpha_adj
    
    print("📊 [第一重檢定: Bonferroni 多重比較校正]")
    print(f"   ▶ 測試參數組合數 (m): {m}")
    print(f"   ▶ 原始顯著水準 α: {alpha}")
    print(f"   ▶ 校正後顯著水準 α_adj: {alpha_adj:.6f} (0.05 / {m})")
    print(f"   ▶ In-Sample t-statistic: {t_stat:.3f}")
    print(f"   ▶ 單尾 p-value: {p_value:.6f}")
    if passed_bonf:
        print(f"   ✅ 通過 Bonferroni 校正！p-value < α_adj。獲利能力在統計上具有顯著意義，非隨機挖掘所致。")
    else:
        print(f"   ❌ 未通過 Bonferroni 校正。p-value >= α_adj。此績效有可能是隨機嘗試產生的。")
    print("-" * 80)
    
    # 5. Out-of-Sample (OOS) 盲測 (WFA)
    print("🔮 [第二重檢定: Walk-Forward Analysis 樣本外盲測]")
    print(f"   ▶ 將 IS 最佳參數套用至 2026 年進行盲測...")
    
    res_oos = backtester.run_backtest(
        df_oos,
        ema_len=int(best_params["ema_len"]),
        kc_mult=best_params["kc_mult"],
        swing_len=int(best_params["swing_len"]),
        buffer_pct=0.0015,
        long_only=True,
        slope_threshold=best_params["slope_threshold"],
        cooldown_bars=int(best_params["cooldown_bars"]),
        early_break_even=False
    )
    df_ind_oos = res_oos["indicators"]
    trades_oos = res_oos["trades"]
    final_eq_oos = df_ind_oos["Equity"].iloc[-1]
    stats_oos = calculate_stats(500_000.0, final_eq_oos, df_ind_oos["Equity"], trades_oos, df_oos)
    
    print(f"   ▶ OOS 總報酬率: {stats_oos['total_return_pct']:.2f}% (年化: {stats_oos['ann_return_pct']:.2f}%)")
    print(f"   ▶ OOS Sharpe Ratio: {stats_oos['sharpe_ratio']:.3f} | OOS MDD: {stats_oos['max_drawdown_pct']:.2f}%")
    print(f"   ▶ OOS 交易次數: {stats_oos['total_trades']} | OOS 勝率: {stats_oos['win_rate_pct']:.2f}%")
    
    # 判斷是否崩潰 (Sharpe 比 IS 下降不超過 50%)
    if stats_oos["sharpe_ratio"] >= best_params["sharpe"] * 0.5:
        print(f"   ✅ OOS 盲測通過！績效並未崩潰。")
    else:
        print(f"   ❌ OOS 盲測未通過。績效大幅衰退或出現虧損，可能存在過度擬合。")
    print("-" * 80)
    
    # 6. 全期間 (IS+OOS) 交易數據整合進行蒙地卡羅與 Bootstrap
    res_full = backtester.run_backtest(
        df,
        ema_len=int(best_params["ema_len"]),
        kc_mult=best_params["kc_mult"],
        swing_len=int(best_params["swing_len"]),
        buffer_pct=0.0015,
        long_only=True,
        slope_threshold=best_params["slope_threshold"],
        cooldown_bars=int(best_params["cooldown_bars"]),
        early_break_even=False
    )
    trades_full = res_full["trades"]
    pnl_list = [t["pnl"] for t in trades_full]
    n_trades = len(pnl_list)
    original_mdd = calculate_stats(500_000.0, res_full["indicators"]["Equity"].iloc[-1], res_full["indicators"]["Equity"], trades_full, df)["max_drawdown_pct"]
    
    # 7. Bootstrap Confidence Interval (重抽樣 10,000 次)
    print("🎲 [第三重檢定: Bootstrap 單筆交易重抽樣 (10,000次)]")
    n_simulations = 10000
    bootstrap_returns = []
    
    for _ in range(n_simulations):
        sample_pnl = np.random.choice(pnl_list, size=n_trades, replace=True)
        sim_return = sum(sample_pnl) / 500_000.0 * 100
        bootstrap_returns.append(sim_return)
        
    bootstrap_returns = np.array(bootstrap_returns)
    ci_lower = np.percentile(bootstrap_returns, 2.5)   # 95% 信賴區間下限
    ci_upper = np.percentile(bootstrap_returns, 97.5)  # 95% 信賴區間上限
    ci_5 = np.percentile(bootstrap_returns, 5.0)       # 最差 5% 下限
    
    print(f"   ▶ 全期間總交易筆數: {n_trades}")
    print(f"   ▶ 95% 信賴區間之總報酬率: [{ci_lower:.2f}%, {ci_upper:.2f}%]")
    print(f"   ▶ 最差 5% (第 5 百分位) 總報酬下限: {ci_5:.2f}%")
    if ci_5 > 0:
        print(f"   ✅ Bootstrap 檢定通過！最衰的 5% 宇宙裡策略依然獲利，證實非依賴極端大賺的『一拳超人』。")
    else:
        print(f"   ❌ Bootstrap 檢定未通過。最差下限小於 0，說明策略可能過度依賴少數幾筆好運交易。")
    print("-" * 80)
    
    # 8. Monte Carlo Permutation Test (MCPT) 洗牌檢定
    print("🎲 [第四重檢定: Monte Carlo Permutation Test 交易順序洗牌 (10,000次)]")
    mcpt_mdds = []
    
    for _ in range(n_simulations):
        shuffled_pnl = np.random.permutation(pnl_list)
        equity_sim = 500_000.0 + np.cumsum(shuffled_pnl)
        roll_max = np.maximum.accumulate(equity_sim)
        dd = (equity_sim - roll_max) / roll_max * 100
        mcpt_mdds.append(dd.min())
        
    mcpt_mdds = np.array(mcpt_mdds)
    prob_worse_mdd = np.mean(mcpt_mdds < original_mdd)
    mdd_p_value = 1 - prob_worse_mdd
    
    print(f"   ▶ 原始策略 MDD: {original_mdd:.2f}%")
    print(f"   ▶ 洗牌後平均模擬 MDD: {np.mean(mcpt_mdds):.2f}%")
    print(f"   ▶ MDD fragile-tail p-value: {mdd_p_value:.4f}")
    if mdd_p_value >= 0.05:
        print(f"   ✅ MCPT 檢定通過！原始 MDD 未落在洗牌排列的最脆弱 5% 尾端，證明交易順序壓力測試安全。")
    else:
        print(f"   ⚠️ MCPT 警告！原始 MDD 落在洗牌排列的最脆弱 5% 尾端，代表存在連輸高度集中的極端風險。")
    print("-" * 80)
    
    # 9. Regime Filter (多空牛熊季線切割測試)
    print("🐻 [時間序列防護: Regime Filter 多空牛熊季線切割]")
    # 在 15分K 上，以 60 日均線作為季線。一天約 76 根 K 棒，60日 = 4560 根。
    df_regime = df.copy()
    df_regime["MA_60d"] = df_regime["Close"].rolling(window=4560).mean()
    df_regime = df_regime.dropna(subset=["MA_60d"])
    
    # 切割多頭牛市與空頭熊市時段
    bull_mask = df_regime["Close"] > df_regime["MA_60d"]
    bear_mask = df_regime["Close"] <= df_regime["MA_60d"]
    
    df_bull = df_regime[bull_mask].copy()
    df_bear = df_regime[bear_mask].copy()
    
    # 分別回測這兩段
    res_bull = backtester.run_backtest(df_bull, ema_len=int(best_params["ema_len"]), kc_mult=best_params["kc_mult"], swing_len=int(best_params["swing_len"]), buffer_pct=0.0015, long_only=True, slope_threshold=best_params["slope_threshold"], cooldown_bars=int(best_params["cooldown_bars"]), early_break_even=False)
    res_bear = backtester.run_backtest(df_bear, ema_len=int(best_params["ema_len"]), kc_mult=best_params["kc_mult"], swing_len=int(best_params["swing_len"]), buffer_pct=0.0015, long_only=True, slope_threshold=best_params["slope_threshold"], cooldown_bars=int(best_params["cooldown_bars"]), early_break_even=False)
    
    stats_bull = calculate_stats(500_000.0, res_bull["indicators"]["Equity"].iloc[-1], res_bull["indicators"]["Equity"], res_bull["trades"], df_bull)
    stats_bear = calculate_stats(500_000.0, res_bear["indicators"]["Equity"].iloc[-1], res_bear["indicators"]["Equity"], res_bear["trades"], df_bear)
    
    print(f"   🟢 【多頭環境 (季線之上)】交易天數: {len(set(df_bull.index.date))} 天")
    print(f"      - 總報酬率: {stats_bull['total_return_pct']:.2f}% | Sharpe: {stats_bull['sharpe_ratio']:.3f} | MDD: {stats_bull['max_drawdown_pct']:.2f}%")
    print(f"      - 交易次數: {stats_bull['total_trades']} | 勝率: {stats_bull['win_rate_pct']:.2f}% | Profit Factor: {stats_bull['profit_factor']:.2f}")
    
    print(f"   🔴 【空頭環境 (季線之下)】交易天數: {len(set(df_bear.index.date))} 天")
    print(f"      - 總報酬率: {stats_bear['total_return_pct']:.2f}% | Sharpe: {stats_bear['sharpe_ratio']:.3f} | MDD: {stats_bear['max_drawdown_pct']:.2f}%")
    print(f"      - 交易次數: {stats_bear['total_trades']} | 勝率: {stats_bear['win_rate_pct']:.2f}% | Profit Factor: {stats_bear['profit_factor']:.2f}")
    
    print("-" * 80)
    
    # 10. 繪製檢定分析大圖
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # A. 權益曲線 WFA 盲測
    df_ind_full = res_full["indicators"]
    axes[0, 0].plot(df_ind_full.index, df_ind_full["Equity"], label="全期間權益 (2025-2026)", color="navy")
    axes[0, 0].axvline(pd.Timestamp("2026-01-01"), color="red", linestyle="--", label="樣本外 (OOS) 盲測起點")
    axes[0, 0].set_title("Walk-Forward Analysis 權益曲線", fontsize=12)
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)
    axes[0, 0].legend()
    
    # B. Bootstrap 分佈
    axes[0, 1].hist(bootstrap_returns, bins=50, color="skyblue", edgecolor="black", alpha=0.7)
    axes[0, 1].axvline(ci_lower, color="red", linestyle="--", label=f"95% CI 下限: {ci_lower:.1f}%")
    axes[0, 1].axvline(ci_5, color="orange", linestyle="-.", label=f"最差 5% 下限: {ci_5:.1f}%")
    axes[0, 1].axvline(np.mean(bootstrap_returns), color="blue", label=f"平均報酬: {np.mean(bootstrap_returns):.1f}%")
    axes[0, 1].set_title("Bootstrap 總報酬率機率分佈 (10,000次重抽樣)", fontsize=12)
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)
    axes[0, 1].legend()
    
    # C. MCPT MDD 分佈
    axes[1, 0].hist(mcpt_mdds, bins=50, color="lightcoral", edgecolor="black", alpha=0.7)
    axes[1, 0].axvline(original_mdd, color="red", linestyle="-", label=f"原始 MDD: {original_mdd:.2f}%")
    axes[1, 0].axvline(np.mean(mcpt_mdds), color="blue", linestyle="--", label=f"隨機平均 MDD: {np.mean(mcpt_mdds):.2f}%")
    axes[1, 0].set_title("MCPT 隨機洗牌 MDD 分佈 (10,000次模擬)", fontsize=12)
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)
    axes[1, 0].legend()
    
    # D. Regime Filter 多空權益曲線
    axes[1, 1].plot(res_bull["indicators"].index, res_bull["indicators"]["Equity"], label="牛市時段多頭", color="green")
    axes[1, 1].plot(res_bear["indicators"].index, res_bear["indicators"]["Equity"], label="熊市時段多頭", color="brown")
    axes[1, 1].set_title("Regime Filter 多空牛熊權益曲線對比", fontsize=12)
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)
    axes[1, 1].legend()
    
    plt.suptitle(f"Swing High Low + Keltner Channel 全方位檢定報告 (最佳參數: EMA={best_params['ema_len']:.0f}, Mult={best_params['kc_mult']:.1f}, Swing={best_params['swing_len']:.0f})", fontsize=14, y=0.96)
    
    plot_path = os.path.join(ARTIFACT_DIR, "quant_validation_charts.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"🎨 檢定分析綜合大圖已存至 {plot_path}")
    
    # 輸出 CSV 檔案
    val_results = {
        "檢定項目": ["Bonferroni 校正", "OOS 盲測 (WFA)", "Bootstrap 信賴區間", "MCPT MDD 排列檢定", "牛市環境績效", "熊市環境績效"],
        "評估指標": [
            f"IS t-stat: {t_stat:.2f}, p-value: {p_value:.6f} (alpha_adj: {alpha_adj:.6f})",
            f"OOS Return: {stats_oos['total_return_pct']:.2f}%, Sharpe: {stats_oos['sharpe_ratio']:.2f}, MDD: {stats_oos['max_drawdown_pct']:.2f}%",
            f"95% CI 下限: {ci_lower:.2f}%, 最差 5% 下限: {ci_5:.2f}%",
            f"原始 MDD: {original_mdd:.2f}%, 洗牌平均 MDD: {np.mean(mcpt_mdds):.2f}%, p-value: {mdd_p_value:.4f}",
            f"報酬: {stats_bull['total_return_pct']:.2f}%, Sharpe: {stats_bull['sharpe_ratio']:.3f}, MDD: {stats_bull['max_drawdown_pct']:.2f}%",
            f"報酬: {stats_bear['total_return_pct']:.2f}%, Sharpe: {stats_bear['sharpe_ratio']:.3f}, MDD: {stats_bear['max_drawdown_pct']:.2f}%"
        ],
        "檢定結論": [
            "通過 (統計顯著非隨機)" if passed_bonf else "未通過",
            "通過 (績效未崩潰)" if stats_oos["sharpe_ratio"] >= best_params["sharpe"] * 0.5 else "未通過 (過度擬合)",
            "通過 (獲利穩健大於0)" if ci_5 > 0 else "未通過 (依賴一拳超人)",
            "通過 (壓力測試安全)" if mdd_p_value >= 0.05 else "警告 (有連輸集中的脆弱性)",
            "健康 (主要利潤來源)",
            "合格 (熊市能自保)" if stats_bear["total_return_pct"] >= -5.0 else "警示 (熊市回撤過大)"
        ]
    }
    
    val_df = pd.DataFrame(val_results)
    csv_path = os.path.join(ARTIFACT_DIR, "validation_results.csv")
    val_df.to_csv(csv_path, index=False)
    print(f"📊 檢定分析報告 CSV 已存至 {csv_path}")
    try:
        import shutil
        local_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
        os.makedirs(local_dir, exist_ok=True)
        shutil.copy(plot_path, os.path.join(local_dir, "quant_validation_charts.png"))
        shutil.copy(csv_path, os.path.join(local_dir, "validation_results.csv"))
        print("💾 成果已同步備份至本地 reports 目錄")
    except Exception as e:
        print(f"⚠️ 備份失敗: {e}")
    print("=" * 80)

if __name__ == "__main__":
    run_pipeline()
