"""
進階量化檢驗：Walk-Forward Analysis 與 Bonferroni Correction
================================================================
1. 將 2025-2026 資料分為：
   - In-Sample (IS): 2025-01-01 ~ 2025-12-31
   - Out-of-Sample (OOS): 2026-01-01 至今
2. Bonferroni Correction: 在 IS 期間測試 16 個策略，調整顯著性門檻。
3. Walk-Forward: 挑選 IS 前 3 名策略，在 OOS 進行盲測，檢視績效是否崩潰。
"""
import sys
import os
import traceback
import pandas as pd
import numpy as np
import math
from statistics import NormalDist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategies import STRATEGIES
from intraday_backtester import IntradayBacktester
from download_kbar import load_5min_data


def run_advanced_validation():
    print("=" * 100)
    print("🔬 進階量化檢驗：Walk-Forward Analysis & Bonferroni Correction")
    print("=" * 100)

    # 1. 載入資料
    df = load_5min_data()
    if df.empty:
        print("❌ 無法載入資料")
        return

    # 2. 分割樣本 IS / OOS
    is_mask = (df.index >= "2025-01-01") & (df.index <= "2025-12-31")
    oos_mask = df.index >= "2026-01-01"

    df_is = df[is_mask].copy()
    df_oos = df[oos_mask].copy()

    if df_is.empty or df_oos.empty:
        print(f"❌ 資料不足以進行分割。IS: {len(df_is)} 筆, OOS: {len(df_oos)} 筆")
        return

    # 計算年份 (用來計算 Sharpe t-statistic)
    days_is = len(set(df_is.index.date))
    years_is = days_is / 252.0
    days_oos = len(set(df_oos.index.date))

    print(f"📅 樣本內 (IS, 訓練挑選) : 2025-01-01 ~ 2025-12-31 ({days_is} 個交易日)")
    print(f"📅 樣本外 (OOS, 盲測驗證): 2026-01-01 ~ 至今 ({days_oos} 個交易日)")
    print(f"策略數量 (N): {len(STRATEGIES)}")
    print("=" * 100)

    # Bonferroni 校正設定
    N = len(STRATEGIES)
    alpha = 0.05
    bonferroni_alpha = alpha / N
    # 根據 bonferroni_alpha 反推 t 值門檻 (單尾檢定)
    t_threshold = NormalDist().inv_cdf(1 - bonferroni_alpha)
    
    print(f"📊 Bonferroni 統計校正")
    print(f"  - 原始顯著水準 α: {alpha}")
    print(f"  - 校正後顯著水準 α_adj: {bonferroni_alpha:.6f} (0.05 / {N})")
    print(f"  - 要求的最低 t-statistic: {t_threshold:.3f}")
    print("=" * 100)

    # 3. 在 IS 期間跑所有策略
    bt_is = IntradayBacktester()
    
    is_results = []
    print("⏳ 正在進行 In-Sample (2025) 測試...")

    for name, func in STRATEGIES.items():
        try:
            if "Trendlines" in name:
                display_name = f"{name} (LB=5)"
                stats_res = bt_is.run(df_is, func, lookback=5)
            else:
                display_name = name
                stats_res = bt_is.run(df_is, func)

            if not stats_res:
                continue
            
            sr = stats_res["sharpe_ratio"]
            # 簡化的 Sharpe t-statistic 計算: t = SR * sqrt(Years)
            t_stat = sr * np.sqrt(years_is) if years_is > 0 else 0
            # 計算單尾 p-value
            p_val = 0.5 * math.erfc(t_stat / math.sqrt(2)) if t_stat > 0 else 1.0

            is_results.append({
                "Strategy": display_name,
                "func": func,
                "IS_Return": stats_res["total_return_pct"],
                "IS_Sharpe": sr,
                "IS_MDD": stats_res["max_drawdown_pct"],
                "t_stat": t_stat,
                "p_value": p_val,
                "passed_bonferroni": p_val < bonferroni_alpha
            })
        except Exception as e:
            pass

    # 4. IS 排名並挑選前三名
    is_df = pd.DataFrame(is_results)
    is_df = is_df.sort_values("IS_Sharpe", ascending=False)
    
    print("\n🏆 In-Sample (2025) 前 5 名策略：")
    for i, row in is_df.head(5).iterrows():
        is_bonf = "✅" if row["passed_bonferroni"] else "❌"
        print(f"  {row['Strategy']:<40} | IS Sharpe: {row['IS_Sharpe']:+.3f} | p-value: {row['p_value']:.6f} | 通過 Bonferroni: {is_bonf}")

    top_3_strategies = is_df.head(3).to_dict('records')

    # 5. OOS 盲測
    print("\n" + "=" * 100)
    print("🔮 Out-of-Sample (2026) 盲測：檢驗是否過度擬合")
    print("=" * 100)
    
    bt_oos = IntradayBacktester()
    
    oos_results = []
    for strat in top_3_strategies:
        name = strat["Strategy"]
        func = strat["func"]
        
        if "Trendlines" in name:
            stats_oos = bt_oos.run(df_oos, func, lookback=5)
        else:
            stats_oos = bt_oos.run(df_oos, func)
            
        oos_results.append({
            "Strategy": name,
            "IS_Return(%)": round(strat["IS_Return"], 2),
            "IS_Sharpe": round(strat["IS_Sharpe"], 3),
            "OOS_Return(%)": round(stats_oos["total_return_pct"], 2),
            "OOS_Sharpe": round(stats_oos["sharpe_ratio"], 3),
            "OOS_MDD(%)": round(stats_oos["max_drawdown_pct"], 2),
            "Passed Bonferroni(IS)": strat["passed_bonferroni"]
        })

    # 6. 輸出最終報告
    final_df = pd.DataFrame(oos_results)
    print(final_df.to_string(index=False))

    print("\n💡 結論判定：")
    for row in oos_results:
        print(f"\n[{row['Strategy']}]")
        if not row["Passed Bonferroni(IS)"]:
            print("  ⚠️ 在 2025 年的表現未通過 Bonferroni 統計校正 (可能只是運氣好/資料挖礦)。")
        else:
            print("  ✅ 在 2025 年具備真實的統計顯著性 (Alpha)。")
            
        if row["OOS_Sharpe"] < 0:
            print("  ❌ 在 2026 年 (OOS) 績效崩潰 (Sharpe < 0)，證明是過度擬合，實盤不可用。")
        elif row["OOS_Sharpe"] < row["IS_Sharpe"] * 0.5:
            print("  ⚠️ 在 2026 年 (OOS) 績效大幅衰退，穩定性堪憂。")
        else:
            print("  ✅ 在 2026 年 (OOS) 表現穩健，是真正可用的當沖策略！")


if __name__ == "__main__":
    run_advanced_validation()
