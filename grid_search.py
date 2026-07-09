"""
Swing High Low + Keltner Channel 策略參數尋優 (Grid Search) (參數化版)
========================================================================
- 使用核心指定參數：EMA=20, ATR_mult=2.0, swing_len=2
- 測試多組斜率門檻與冷卻期
- 輸出最優實戰搭配
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
    
    # 核心指定參數
    ema_len = 20
    kc_mult = 2.0
    swing_len = 2
    
    slopes = [0.0, 0.01, 0.02, 0.04, 0.06]
    cooldowns = [0, 12, 24]
    ebes = [False, True]
    
    records = []
    
    print("⏳ 開始進行 15 分 K 參數尋優...")
    for slope in slopes:
        for cd in cooldowns:
            for ebe in ebes:
                res = backtester.run_backtest(
                    df_15m, 
                    ema_len=ema_len,
                    kc_mult=kc_mult,
                    swing_len=swing_len,
                    buffer_pct=0.0015, 
                    long_only=True,
                    slope_threshold=slope, 
                    cooldown_bars=cd, 
                    early_break_even=ebe
                )
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
                
    result_df = pd.DataFrame(records)
    result_df = result_df.sort_values(by="sharpe", ascending=False)
    
    search_csv_path = os.path.join(r"C:\Users\User\.gemini\antigravity\brain\b66c39bc-2b4c-41a5-b216-d91083c6b9a3", "grid_search_results.csv")
    result_df.to_csv(search_csv_path, index=False)
    print(f"💾 參數尋優結果已存至 {search_csv_path}")

if __name__ == "__main__":
    run_grid_search()
