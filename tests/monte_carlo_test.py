"""
Monte Carlo Permutation Test (MCPT) & Bootstrap 10,000 次壓力測試
===================================================================
針對 💎 Swing + Keltner Hybrid (Causal, breakout_confirm: EMA=60, Mult=2.0, SW=5)
進行兩大嚴格檢定：
1. Bootstrap Confidence Interval (取後放回重抽樣 10,000 次)：
   檢驗獲利是否過度依賴少數極端大賺交易（厚尾依賴 Fat-Tail Dependency）。
   判斷標準：95% CI 下限 (2.5%) 或最差 5% 下限 (5.0%) > 0%。
2. Monte Carlo Permutation Test (MCPT，打亂交易順序 10,000 次)：
   檢驗最大回撤 (MDD) 是否對交易順序過度脆弱。
   判斷標準：fragile-tail p-value >= 0.05。
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.download_kbar import load_5min_data
from core.intraday_backtester import IntradayBacktester
from core.strategies import swing_keltner_hybrid


def run_stress_test_for_period(df_sub: pd.DataFrame, period_label: str, bt: IntradayBacktester,
                               swing_len: int = 5, ema_len: int = 60, atr_mult: float = 2.0,
                               n_simulations: int = 10000):
    stats = bt.run(df_sub, swing_keltner_hybrid, swing_len=swing_len, ema_len=ema_len, atr_mult=atr_mult)
    trade_list = stats.get("trade_list", [])
    if not trade_list:
        print(f"❌ {period_label} 無交易紀錄")
        return None

    trades = np.array([t["pnl_cash"] for t in trade_list], dtype=float)
    n_trades = len(trades)
    initial_cap = bt.initial_capital
    original_return = stats["total_return_pct"]
    original_mdd = stats["max_drawdown_pct"]

    # 計算每日真實年化 Sharpe
    eq_diff = stats["equity_curve"].diff().fillna(0)
    daily_ret = eq_diff.groupby(eq_diff.index.date).sum() / initial_cap
    daily_sharpe = float((daily_ret.mean() / daily_ret.std()) * np.sqrt(252)) if daily_ret.std() != 0 else 0.0

    rng = np.random.default_rng(42)

    # 1. Bootstrap (10,000 次取後放回重抽樣，向量化加速)
    boot_indices = rng.integers(0, n_trades, size=(n_simulations, n_trades))
    boot_samples = trades[boot_indices]
    boot_returns = boot_samples.sum(axis=1) / initial_cap * 100.0

    ci_2_5 = float(np.percentile(boot_returns, 2.5))
    ci_5_0 = float(np.percentile(boot_returns, 5.0))
    ci_50 = float(np.percentile(boot_returns, 50.0))
    ci_97_5 = float(np.percentile(boot_returns, 97.5))
    prob_positive = float(np.mean(boot_returns > 0) * 100.0)

    # 2. MCPT (10,000 次隨機洗牌排列，計算每次模擬的 MDD)
    mcpt_mdds = np.empty(n_simulations, dtype=float)
    for i in range(n_simulations):
        shuffled = rng.permutation(trades)
        eq = np.empty(n_trades + 1, dtype=float)
        eq[0] = initial_cap
        eq[1:] = initial_cap + np.cumsum(shuffled)
        roll_max = np.maximum.accumulate(eq)
        dd = (eq - roll_max) / roll_max * 100.0
        mcpt_mdds[i] = dd.min()

    # 隨機洗牌比原始 MDD 更慘 (更負) 的比例
    prob_worse_mdd = float(np.mean(mcpt_mdds < original_mdd))
    # 若原始 MDD 落在最慘的 5% 尾端 (即只有不到 5% 的隨機排列比原始更慘)，代表交易順序存在脆弱連虧風險
    mdd_fragility_p = prob_worse_mdd
    mean_mcpt_mdd = float(np.mean(mcpt_mdds))
    p5_mcpt_mdd = float(np.percentile(mcpt_mdds, 5.0))

    print("=" * 100)
    print(f"🎲 【{period_label}】 breakout_confirm (EMA={ema_len}, Mult={atr_mult}, SW={swing_len}) — 10,000 次壓力測試")
    print("=" * 100)
    print(f"📌 原始回測表現：")
    print(f"   - 總交易筆數   : {n_trades} 筆 (勝率: {stats['win_rate_pct']:.1f}%, Profit Factor: {stats['profit_factor']:.2f})")
    print(f"   - 原始總報酬率 : {original_return:+.2f}% (最終權益: {stats['final_equity']:,.0f} TWD)")
    print(f"   - Daily Sharpe : {daily_sharpe:+.3f} (Bar Sharpe: {stats['sharpe_ratio']:+.3f})")
    print(f"   - 原始最大回撤 : {original_mdd:.2f}%")
    print(f"   - 單筆極端值   : 最大單筆獲利 {trades.max():+,.0f} TWD | 最大單筆虧損 {trades.min():+,.0f} TWD")
    print(f"   - 平均獲利/虧損: 平均獲利 {stats['avg_win']:+,.0f} TWD | 平均虧損 {stats['avg_loss']:+,.0f} TWD")

    print(f"\n📊 [檢定 1] Bootstrap 單筆重抽樣檢定 (10,000 次取後放回)：")
    print(f"   - 獲利機率 (Return > 0%) : {prob_positive:.2f}%")
    print(f"   - 報酬率中位數 (50%ile)  : {ci_50:+.2f}%")
    print(f"   - 95% 信賴區間 (2.5%~97.5%): [{ci_2_5:+.2f}%, {ci_97_5:+.2f}%]")
    print(f"   - 最差 5% 報酬下界 (5%ile) : {ci_5_0:+.2f}%")
    if ci_2_5 > 0:
        print(f"   ✅ 完美通過！連 95% CI 下界 ({ci_2_5:+.2f}%) 都大於 0，證明獲利極度紮實，完全不靠單筆運氣！")
    elif ci_5_0 > 0:
        print(f"   ✅ 通過檢定！最差 5% 下界 ({ci_5_0:+.2f}%) 大於 0，具備統計穩健性！")
    else:
        print(f"   ⚠️ 未通過嚴格下界：最差 5% 下界 ({ci_5_0:+.2f}%) 小於 0，仍有部分厚尾依賴。")

    print(f"\n📊 [檢定 2] MCPT 蒙地卡羅順序洗牌檢定 (10,000 次打亂順序)：")
    print(f"   - 原始策略 MDD         : {original_mdd:.2f}%")
    print(f"   - 洗牌後平均模擬 MDD   : {mean_mcpt_mdd:.2f}% (最差 5% 模擬 MDD: {p5_mcpt_mdd:.2f}%)")
    print(f"   - 隨機排列比原始更慘比例: {prob_worse_mdd * 100:.1f}%")
    print(f"   - MDD fragile-tail p值 : {mdd_fragility_p:.4f}")
    if mdd_fragility_p >= 0.05:
        print(f"   ✅ 通過 MCPT 壓力測試！(p = {mdd_fragility_p:.4f} >= 0.05) 原始回撤未落入脆弱尾端，無異常連虧風險。")
    else:
        print(f"   ⚠️ MCPT 警告 (p = {mdd_fragility_p:.4f} < 0.05)：交易順序存在連虧集中現象。")

    return {
        "period": period_label,
        "n_trades": n_trades,
        "original_return": original_return,
        "daily_sharpe": daily_sharpe,
        "original_mdd": original_mdd,
        "prob_positive": prob_positive,
        "ci_2_5": ci_2_5,
        "ci_5_0": ci_5_0,
        "ci_97_5": ci_97_5,
        "mean_mcpt_mdd": mean_mcpt_mdd,
        "mdd_fragility_p": mdd_fragility_p,
    }


if __name__ == "__main__":
    df = load_5min_data()
    df_oos = df[(df.index >= "2026-01-01") & (df.index <= "2026-10-16 23:59:59")].copy()
    df_full = df[(df.index >= "2025-01-01") & (df.index <= "2026-10-16 23:59:59")].copy()

    bt = IntradayBacktester(
        initial_capital=500_000,
        commission=20,
        slippage=1,
        multiplier=50,
        force_close_time=None,
        rollover_mode="skip_settlement_day",
    )

    run_stress_test_for_period(df_oos, "2026 樣本外盲測期 (2026/01/01 ~ 2026/10/16)", bt, swing_len=5, ema_len=60, atr_mult=2.0)
    print()
    run_stress_test_for_period(df_full, "2025~2026 完整跨年全週期 (2025/01/01 ~ 2026/10/16)", bt, swing_len=5, ema_len=60, atr_mult=2.0)
