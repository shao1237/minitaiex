"""
執行 Swing High Low + Keltner Channel 策略的回測與對比 (最佳實戰參數版)
========================================================================
- 5分K 僅做多 (原始)
- 15分K 僅做多 (原始)
- 15分K 僅做多 (優化實戰版): 斜率門檻=0.04, 停損冷卻=12根K棒, 停利保本維持原設定
- 15分K 多空雙向 (優化實戰版): 斜率門檻=0.04, 停損冷卻=12根K棒, 停利保本維持原設定
- 繪製對比圖表並輸出為 CSV
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from swing_kc_backtest import SwingKeltnerBacktester, calculate_stats

# 設定 Matplotlib 字體以正常顯示中文
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# Artifact 輸出目錄
ARTIFACT_DIR = r"C:\Users\User\.gemini\antigravity\brain\b66c39bc-2b4c-41a5-b216-d91083c6b9a3"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def run_all_backtests():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    path_5m = os.path.join(data_dir, "mxf_5min.parquet")
    path_15m = os.path.join(data_dir, "mxf_15min.parquet")
    
    print("⏳ 載入數據...")
    df_5m = pd.read_parquet(path_5m).sort_index()
    df_15m = pd.read_parquet(path_15m).sort_index()
    
    initial_cap = 500_000.0
    backtester = SwingKeltnerBacktester(initial_capital=initial_cap, commission=20, slippage=1.0, multiplier=50.0)
    
    # 測試方案列表
    configs = [
        {
            "name": "5min_Long_Original", "df": df_5m, "label": "5分K 僅做多 (原始)",
            "params": {"long_only": True, "slope_threshold": 0.0, "cooldown_bars": 0, "early_break_even": False}
        },
        {
            "name": "15min_Long_Original", "df": df_15m, "label": "15分K 僅做多 (原始)",
            "params": {"long_only": True, "slope_threshold": 0.0, "cooldown_bars": 0, "early_break_even": False}
        },
        {
            "name": "15min_Long_Optimized", "df": df_15m, "label": "15分K 僅做多 (優化實戰版)",
            "params": {"long_only": True, "slope_threshold": 0.04, "cooldown_bars": 12, "early_break_even": False}
        },
        {
            "name": "15min_LS_Optimized", "df": df_15m, "label": "15分K 多空雙向 (優化實戰版)",
            "params": {"long_only": False, "slope_threshold": 0.04, "cooldown_bars": 12, "early_break_even": False}
        }
    ]
    
    results = {}
    plt.figure(figsize=(14, 8))
    
    for cfg in configs:
        name = cfg["name"]
        print(f"🚀 執行回測: {cfg['label']} ...")
        
        # 執行回測
        res = backtester.run_backtest(cfg["df"], buffer_pct=0.0015, **cfg["params"])
        
        df_indicators = res["indicators"]
        trades = res["trades"]
        final_eq = df_indicators["Equity"].iloc[-1]
        
        # 計算統計
        stats = calculate_stats(initial_cap, final_eq, df_indicators["Equity"], trades, cfg["df"])
        results[name] = stats
        
        # 繪製權益曲線
        linestyle = "--" if "Original" in name else "-"
        plt.plot(df_indicators.index, df_indicators["Equity"], label=f"{cfg['label']} (ROI: {stats['total_return_pct']:.1f}%)", linestyle=linestyle)
        
        print(f"   總收益: {stats['total_return_pct']:.2f}% | 年化: {stats['ann_return_pct']:.2f}% | MDD: {stats['max_drawdown_pct']:.2f}%")
        print(f"   交易筆數: {stats['total_trades']} | 勝率: {stats['win_rate_pct']:.2f}% | Profit Factor: {stats['profit_factor']:.2f} | Sharpe: {stats['sharpe_ratio']:.2f}")
        print("-" * 50)
        
    plt.title("Swing High Low + Keltner Channel 策略最佳實戰對比 (2025 - 2026)", fontsize=14)
    plt.xlabel("日期", fontsize=12)
    plt.ylabel("權益 (TWD)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=10, loc="upper left")
    
    # 存檔對比圖
    plot_path = os.path.join(ARTIFACT_DIR, "equity_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"🎨 權益對比圖已存至 {plot_path}")
    
    # 轉為 DataFrame 格式輸出
    summary_df = pd.DataFrame(results).T
    summary_df.index.name = "策略組合"
    
    # 格式化輸出
    summary_df["total_return_pct"] = summary_df["total_return_pct"].map("{:.2f}%".format)
    summary_df["ann_return_pct"] = summary_df["ann_return_pct"].map("{:.2f}%".format)
    summary_df["max_drawdown_pct"] = summary_df["max_drawdown_pct"].map("{:.2f}%".format)
    summary_df["sharpe_ratio"] = summary_df["sharpe_ratio"].map("{:.2f}".format)
    summary_df["win_rate_pct"] = summary_df["win_rate_pct"].map("{:.2f}%".format)
    summary_df["profit_factor"] = summary_df["profit_factor"].map("{:.2f}".format)
    summary_df["avg_win"] = summary_df["avg_win"].map("{:.1f} TWD".format)
    summary_df["avg_loss"] = summary_df["avg_loss"].map("{:.1f} TWD".format)
    summary_df["final_equity"] = summary_df["final_equity"].map("{:,.1f} TWD".format)
    
    summary_csv_path = os.path.join(ARTIFACT_DIR, "backtest_summary.csv")
    summary_df.to_csv(summary_csv_path)
    print(f"📊 統計摘要表已存至 {summary_csv_path}")

if __name__ == "__main__":
    run_all_backtests()
