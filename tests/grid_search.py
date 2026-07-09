import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
肯特納通道核心參數與輔助參數之 Grid Search 尋優腳本
====================================================
- 針對 5 分 K 與 15 分 K 進行大範圍核心參數尋優：
  - 中軌 (EMA 週期): 10, 15, 20, 30
  - 寬度 (ATR 乘數): 1.5, 2.0, 2.5
  - 確認 K 棒數 (swing_len): 1, 2, 3
  - 斜率門檻 (slope_threshold): 0.0, 0.02, 0.04
  - 停損冷卻 (cooldown_bars): 0, 12
- 將尋優結果存至 Artifact 目錄
"""
import os
import pandas as pd
import numpy as np
from swing_kc_backtest import SwingKeltnerBacktester, calculate_stats

# Artifact 輸出目錄
ARTIFACT_DIR = r"C:\Users\User\.gemini\antigravity\brain\b66c39bc-2b4c-41a5-b216-d91083c6b9a3"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def run_grid_search():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    path_5m = os.path.join(data_dir, "mxf_5min.parquet")
    path_15m = os.path.join(data_dir, "mxf_15min.parquet")
    
    print("⏳ 載入回測數據...")
    df_5m = pd.read_parquet(path_5m).sort_index()
    df_15m = pd.read_parquet(path_15m).sort_index()
    
    initial_cap = 500_000.0
    backtester = SwingKeltnerBacktester(initial_capital=initial_cap, commission=20, slippage=1.0, multiplier=50.0)
    
    # 參數空間定義
    ema_lens = [10, 15, 20, 30]
    kc_mults = [1.5, 2.0, 2.5]
    swing_lens = [1, 2, 3]
    slopes = [0.0, 0.02, 0.04]
    cooldowns = [0, 12]
    
    timeframes = [
        {"name": "15min", "df": df_15m},
        {"name": "5min", "df": df_5m}
    ]
    
    records = []
    
    total_combinations = len(timeframes) * len(ema_lens) * len(kc_mults) * len(swing_lens) * len(slopes) * len(cooldowns)
    print(f"🚀 開始執行參數尋優，總組合數: {total_combinations} ...")
    
    counter = 0
    for tf in timeframes:
        tf_name = tf["name"]
        df_tf = tf["df"]
        print(f"⏳ 正在尋優 {tf_name} 週期...")
        
        for ema in ema_lens:
            for mult in kc_mults:
                for swing in swing_lens:
                    for slope in slopes:
                        for cd in cooldowns:
                            # 執行回測 (僅做多)
                            res = backtester.run_backtest(
                                df_tf,
                                ema_len=ema,
                                kc_mult=mult,
                                swing_len=swing,
                                buffer_pct=0.0015,
                                long_only=True,
                                slope_threshold=slope,
                                cooldown_bars=cd,
                                early_break_even=False
                            )
                            df_indicators = res["indicators"]
                            trades = res["trades"]
                            final_eq = df_indicators["Equity"].iloc[-1]
                            
                            stats = calculate_stats(initial_cap, final_eq, df_indicators["Equity"], trades, df_tf)
                            
                            records.append({
                                "timeframe": tf_name,
                                "ema_len": ema,
                                "kc_mult": mult,
                                "swing_len": swing,
                                "slope_threshold": slope,
                                "cooldown_bars": cd,
                                "total_return": stats["total_return_pct"],
                                "mdd": stats["max_drawdown_pct"],
                                "sharpe": stats["sharpe_ratio"],
                                "win_rate": stats["win_rate_pct"],
                                "trades": stats["total_trades"],
                                "profit_factor": stats["profit_factor"]
                            })
                            
                            counter += 1
                            if counter % 50 == 0:
                                print(f"   進度: {counter}/{total_combinations} 組合已完成")
                                
    result_df = pd.DataFrame(records)
    # 按 Sharpe Ratio 降序排序
    result_df = result_df.sort_values(by="sharpe", ascending=False)
    
    # 儲存完整的尋優 CSV
    search_csv_path = os.path.join(ARTIFACT_DIR, "grid_search_results.csv")
    result_df.to_csv(search_csv_path, index=False)
    try:
        import shutil
        local_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
        os.makedirs(local_dir, exist_ok=True)
        shutil.copy(search_csv_path, os.path.join(local_dir, "grid_search_results.csv"))
        print("💾 成果已同步備份至本地 reports 目錄")
    except Exception as e:
        print(f"⚠️ 備份失敗: {e}")
    print(f"\n💾 完整的參數尋優結果已存至 {search_csv_path}")
    
    # 分別列出 15分K 與 5分K 的 Top 5 參數組合以利分析
    print("\n🏆 【15分K 僅做多】Top 5 最佳參數組合：")
    print(result_df[result_df["timeframe"] == "15min"].head(5).to_string(index=False))
    
    print("\n🏆 【5分K 僅做多】Top 5 最佳參數組合：")
    print(result_df[result_df["timeframe"] == "5min"].head(5).to_string(index=False))

if __name__ == "__main__":
    run_grid_search()
