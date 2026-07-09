import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
執行 Swing High Low + Keltner Channel 策略的指定與優化核心參數回測對比
==========================================================================
- 原核心參數對比 (EMA=20, ATR=2.0, Swing=2, 無斜率無冷卻)
- 黃金實戰參數對比 (15min: EMA=10, ATR=2.5, Swing=1, 斜率=0.04, 冷卻=12)
- 黃金實戰參數對比 (5min: EMA=10, ATR=2.5, Swing=1, 斜率=0.02, 冷卻=12)
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
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
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
            "name": "5min_Long_Original", "df": df_5m, "label": "5分K 僅做多 (原核心參數)",
            "params": {"ema_len": 20, "kc_mult": 2.0, "swing_len": 2, "slope_threshold": 0.0, "cooldown_bars": 0, "early_break_even": False}
        },
        {
            "name": "15min_Long_Original", "df": df_15m, "label": "15分K 僅做多 (原核心參數)",
            "params": {"ema_len": 20, "kc_mult": 2.0, "swing_len": 2, "slope_threshold": 0.0, "cooldown_bars": 0, "early_break_even": False}
        },
        {
            "name": "15min_Long_Tuned", "df": df_15m, "label": "15分K 僅做多 (黃金實戰調教)",
            "params": {"ema_len": 10, "kc_mult": 2.5, "swing_len": 1, "slope_threshold": 0.04, "cooldown_bars": 12, "early_break_even": False}
        },
        {
            "name": "5min_Long_Tuned", "df": df_5m, "label": "5分K 僅做多 (黃金實戰調教)",
            "params": {"ema_len": 10, "kc_mult": 2.5, "swing_len": 1, "slope_threshold": 0.02, "cooldown_bars": 12, "early_break_even": False}
        }
    ]
    
    results = {}
    plt.figure(figsize=(14, 8))
    
    for cfg in configs:
        name = cfg["name"]
        print(f"🚀 執行回測: {cfg['label']} ...")
        
        # 執行回測
        res = backtester.run_backtest(
            cfg["df"], 
            ema_len=cfg["params"]["ema_len"], 
            kc_mult=cfg["params"]["kc_mult"], 
            swing_len=cfg["params"]["swing_len"],
            buffer_pct=0.0015, 
            long_only=True,
            slope_threshold=cfg["params"]["slope_threshold"],
            cooldown_bars=cfg["params"]["cooldown_bars"],
            early_break_even=cfg["params"]["early_break_even"]
        )
        
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
        
    plt.title("Swing High Low + Keltner Channel 核心參數調教對比 (2025 - 2026)", fontsize=14)
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
    try:
        import shutil
        local_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
        os.makedirs(local_dir, exist_ok=True)
        shutil.copy(plot_path, os.path.join(local_dir, "equity_comparison.png"))
        shutil.copy(summary_csv_path, os.path.join(local_dir, "backtest_summary.csv"))
        print("💾 成果已同步備份至本地 reports 目錄")
    except Exception as e:
        print(f"⚠️ 備份失敗: {e}")

if __name__ == "__main__":
    run_all_backtests()
