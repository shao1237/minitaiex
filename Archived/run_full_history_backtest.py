import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
執行全歷史回測 (2018-至今)
- 載入所有策略
- 下載/讀取 2018-01-01 至最新的台指期日線資料
- 執行回測並輸出 CSV 報表
"""
import pandas as pd
import traceback
from strategies import STRATEGIES
from backtester import MiniTaiexBacktester
from download_data import download_taiex

def run_all_tests():
    # 1. 準備數據 (2018 ~ present)
    start_date = "2018-01-01"
    print(f"正在載入歷史數據 ({start_date} ~ code)...")
    try:
        df = download_taiex(period="max")
        df = df[df.index >= start_date]
        print(f"數據範圍: {df.index[0]} ~ {df.index[-1]} (共 {len(df)} 筆)")
    except Exception as e:
        print(f"數據載入失敗: {e}")
        return

    # 2. 初始化回測器
    # 假設：小台一口, 把本金設大一點避免破產 (雖不影響點數計算)
    backtester = MiniTaiexBacktester(
        initial_capital=500_000,
        commission=20,  # 單邊 20 元
        slippage=1,     # 單邊滑價 1 點
        multiplier=50   # 小台
    )

    results = []

    # 3. 跑所有策略
    print(f"開始回測 {len(STRATEGIES)} 個策略...")
    
    for name, func in STRATEGIES.items():
        try:
            # 特殊處理需要參數的策略
            if "Trendlines" in name:
                # 測試 lookback=5 (之前優化結果較好)
                name = f"{name} (LB=5)"
                stats = backtester.run(df, func, lookback=5)
            elif "SuperTrend" in name:
                stats = backtester.run(df, func)
            else:
                stats = backtester.run(df, func)
            
            if not stats:
                continue

            results.append({
                "Strategy": name,
                "Return (%)": round(stats["total_return_pct"], 2),
                "Ann. Return (%)": round(stats["ann_return_pct"], 2),
                "Sharpe": round(stats["sharpe_ratio"], 3),
                "MDD (%)": round(stats["max_drawdown_pct"], 2),
                "Win Rate (%)": round(stats["win_rate_pct"], 2),
                "Trades": stats["total_trades"],
                "Final Equity": int(stats["final_equity"])
            })
            print(f"  [OK] {name:<35} | Ret: {stats['total_return_pct']:6.2f}% | Sharpe: {stats['sharpe_ratio']:.3f}")

        except Exception as e:
            print(f"  [ERR] {name}: {e}")
            traceback.print_exc()

    # 4. 存檔
    if results:
        res_df = pd.DataFrame(results)
        # 排序：Sharpe 高到低 (或可改為此期間總報酬高到低)
        res_df = res_df.sort_values("Sharpe", ascending=False)
        
        output_file = "full_historical_results_2018_present.csv"
        res_df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"\n回測完成！結果已存至 {output_file}")
        print(res_df.head(10).to_string())
    else:
        print("沒有產生任何回測結果。")

if __name__ == "__main__":
    run_all_tests()
