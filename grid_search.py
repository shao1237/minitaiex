"""
Swing High Low + Keltner Channel 策略參數尋優 (Grid Search)
=============================================================
- 針對 5 分 K 與 15 分 K
- 測試多組斜率門檻 (slope_threshold: 0.0, 0.01, 0.02, 0.04, 0.08)
- 測試多組冷卻期 (cooldown_bars: 0, 10, 20)
- 找出最符合使用者「既要過濾盤整、又不要連續停損、更不能追高」的黃金參數組合
"""
import os
import pandas as pd
import numpy as np
from swing_kc_backtest import SwingKeltnerBacktester, calculate_stats

def run_grid_search():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    path_5m = os.path.join(data_dir, "mxf_5min.parquet")
    path_15m = os.path.join(data_dir, "mxf_15min.parquet")
    
    df_5m = pd.read_parquet(path_5m).sort_index()
    df_15m = pd.read_parquet(path_15m).sort_index()
    
    initial_cap = 500_000.0
    backtester = SwingKeltnerBacktester(initial_capital=initial_cap, commission=20, slippage=1.0, multiplier=50.0)
    
    slopes = [0.0, 0.01, 0.02, 0.04, 0.06]
    cooldowns = [0, 12, 24]
    ebes = [False, True]
    
    records = []
    
    print("⏳ 開始進行 15 分 K 參數尋優...")
    for slope in slopes:
        for cd in cooldowns:
            for ebe in ebes:
                res = backtester.run_backtest(df_15m, buffer_pct=0.0015, long_only=True,
                                              slope_threshold=slope, cooldown_bars=cd, early_break_even=ebe)
                df_indicators = res["indicators"]
                trades = res["trades"]
                final_eq = df_indicators["Equity"].iloc[-1]
                stats = calculate_stats(initial_cap, final_eq, df_indicators["Equity"], trades, df_15m)
                
                records.append({
                    "timeframe": "15min",
                    "slope_threshold": slope,
                    "cooldown_bars": cd,
                    "early_break_even": ebe,
                    "total_return": stats["total_return_pct"],
                    "mdd": stats["max_drawdown_pct"],
                    "sharpe": stats["sharpe_ratio"],
                    "win_rate": stats["win_rate_pct"],
                    "trades": stats["total_trades"],
                    "profit_factor": stats["profit_factor"]
                })
                
    print("⏳ 開始進行 5 分 K 參數尋優...")
    for slope in slopes:
        for cd in cooldowns:
            for ebe in ebes:
                res = backtester.run_backtest(df_5m, buffer_pct=0.0015, long_only=True,
                                              slope_threshold=slope, cooldown_bars=cd, early_break_even=ebe)
                df_indicators = res["indicators"]
                trades = res["trades"]
                final_eq = df_indicators["Equity"].iloc[-1]
                stats = calculate_stats(initial_cap, final_eq, df_indicators["Equity"], trades, df_5m)
                
                records.append({
                    "timeframe": "5min",
                    "slope_threshold": slope,
                    "cooldown_bars": cd,
                    "early_break_even": ebe,
                    "total_return": stats["total_return_pct"],
                    "mdd": stats["max_drawdown_pct"],
                    "sharpe": stats["sharpe_ratio"],
                    "win_rate": stats["win_rate_pct"],
                    "trades": stats["total_trades"],
                    "profit_factor": stats["profit_factor"]
                })
                
    result_df = pd.DataFrame(records)
    # 按 Sharpe Ratio 降序排列
    result_df = result_df.sort_values(by="sharpe", ascending=False)
    
    print("\n🏆 Top 10 最佳參數組合 (以 Sharpe Ratio 排序)：")
    print(result_df.head(10).to_string(index=False))
    
    # 存檔尋優結果
    search_csv_path = os.path.join(r"C:\Users\User\.gemini\antigravity\brain\b66c39bc-2b4c-41a5-b216-d91083c6b9a3", "grid_search_results.csv")
    result_df.to_csv(search_csv_path, index=False)
    print(f"\n💾 參數尋優結果已存至 {search_csv_path}")

if __name__ == "__main__":
    run_grid_search()
