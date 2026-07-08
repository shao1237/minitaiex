"""
Monte Carlo Permutation Test (MCPT) & Bootstrap Confidence Intervals
用來檢定策略獲利是否只依賴少數極端好運的交易。
"""
import sys
import os
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strategies import STRATEGIES
from intraday_backtester import IntradayBacktester
from download_kbar import load_5min_data

def run_monte_carlo():
    print("=" * 100)
    print("🎲 蒙地卡羅排列檢定 (MCPT) & Bootstrap 信心水準分析")
    print("=" * 100)

    # 1. 載入資料 (只取 2026)
    df = load_5min_data()
    if df.empty:
        print("❌ 無法載入資料")
        return
    df = df[df.index >= "2026-01-01"]

    bt = IntradayBacktester(initial_capital=500_000, commission=20, slippage=1, multiplier=50)

    strategies_to_test = ["Session Edge Ensemble", "Swing High Low", "Keltner Channel"]
    n_simulations = 10000

    for name in strategies_to_test:
        print(f"\n🔍 正在分析策略: {name} ...")
        
        # 取得策略函數
        func = None
        for k, v in STRATEGIES.items():
            if name in k:
                func = v
                break
        
        if not func:
            print(f"❌ 找不到策略 {name}")
            continue

        # 執行回測取得交易紀錄
        stats = bt.run(df, func)
        if not stats or not stats.get("trade_list"):
            print("❌ 回測無交易紀錄")
            continue

        trades = [t["pnl_cash"] for t in stats["trade_list"]]
        original_return = sum(trades) / 500_000 * 100
        n_trades = len(trades)
        
        print(f"   ▶ 原始交易次數: {n_trades} 次")
        print(f"   ▶ 原始總報酬率: {original_return:.2f}%")
        print(f"   ▶ 原始最大交易獲利: {max(trades):.0f} TWD")
        print(f"   ▶ 原始最大交易虧損: {min(trades):.0f} TWD")

        # ==========================================
        # 1. Bootstrap Confidence Interval (重抽樣，取後放回)
        # 目的：打亂極端值出現的機率。如果只靠一兩筆大賺，重抽樣很容易抽不到它們，導致總報酬崩潰。
        # ==========================================
        bootstrap_returns = []
        for _ in range(n_simulations):
            # 取後放回抽樣
            sample_trades = np.random.choice(trades, size=n_trades, replace=True)
            sim_return = sum(sample_trades) / 500_000 * 100
            bootstrap_returns.append(sim_return)

        bootstrap_returns = np.array(bootstrap_returns)
        ci_lower = np.percentile(bootstrap_returns, 2.5)   # 95% CI 下限
        ci_upper = np.percentile(bootstrap_returns, 97.5)  # 95% CI 上限
        ci_5 = np.percentile(bootstrap_returns, 5.0)       # 90% CI 下限

        print("\n   [Bootstrap 95% 信心區間 (重抽樣 10,000 次)]")
        print(f"   ▶ 報酬率 95% CI: [{ci_lower:.2f}%, {ci_upper:.2f}%]")
        
        if ci_5 > 0:
            print(f"   ✅ 最差 5% 的報酬率下限為 {ci_5:.2f}% (大於 0) -> 獲利具備統計穩健性，不是靠單筆賽到的！")
        else:
            print(f"   ❌ 最差 5% 的報酬率下限為 {ci_5:.2f}% (小於 0) -> 如果運氣不好沒抓到那幾筆極端大賺，策略會賠錢。")

        # ==========================================
        # 2. Monte Carlo Permutation Test (MCPT) (打亂順序，取後不放回)
        # 目的：檢驗 MDD 是否只是因為運氣好 (連贏/連輸的排列)。
        # ==========================================
        original_mdd = stats["max_drawdown_pct"]
        mcpt_mdds = []

        for _ in range(n_simulations):
            # 取後不放回 (洗牌)
            shuffled_trades = np.random.permutation(trades)
            
            # 計算該排列下的 MDD
            equity = 500_000 + np.cumsum(shuffled_trades)
            rolling_max = np.maximum.accumulate(equity)
            drawdowns = (equity - rolling_max) / rolling_max
            mcpt_mdds.append(drawdowns.min() * 100)

        mcpt_mdds = np.array(mcpt_mdds)
        prob_worse_mdd = np.mean(mcpt_mdds < original_mdd)
        mdd_fragility_p = 1 - prob_worse_mdd

        print(f"\n   [MCPT 排列檢定 (打亂交易順序 10,000 次)]")
        print(f"   ▶ 原始 MDD: {original_mdd:.2f}%")
        print(f"   ▶ 隨機洗牌平均 MDD: {np.mean(mcpt_mdds):.2f}%")
        print(f"   ▶ 有 {prob_worse_mdd*100:.1f}% 的隨機排列會產生比原始更慘的 MDD")
        print(f"   ▶ MDD fragile-tail p-value: {mdd_fragility_p:.3f}")
        
        if mdd_fragility_p >= 0.05:
            print(f"   ✅ 原始 MDD 未落在排列分布最脆弱 5% 尾端 -> 交易順序壓力測試通過。")
        else:
            print(f"   ⚠️ 原始 MDD 落在排列分布最脆弱 5% 尾端 -> 交易順序可能存在連虧集中風險。")

        print("-" * 80)

if __name__ == "__main__":
    run_monte_carlo()
