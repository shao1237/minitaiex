"""
回測所有策略 2025-01-01 ~ 至今
=============================================
"""
import pandas as pd
import traceback
from strategies import STRATEGIES
from backtester import MiniTaiexBacktester
from download_data import load_local_data


def run_2025():
    # 1. 載入數據
    start_date = "2025-01-01"
    df = load_local_data()
    df = df[df.index >= start_date]
    print(f"回測期間: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')} (共 {len(df)} 個交易日)")
    print(f"指數走勢: {df['Close'].iloc[0]:.0f} → {df['Close'].iloc[-1]:.0f} ({(df['Close'].iloc[-1]/df['Close'].iloc[0]-1)*100:+.2f}%)")
    print("=" * 100)

    # 2. 回測器 (小台)
    bt = MiniTaiexBacktester(
        initial_capital=500_000,
        commission=20,
        slippage=1,
        multiplier=50,
    )

    results = []

    # 3. 跑所有策略
    for name, func in STRATEGIES.items():
        try:
            if "Trendlines" in name:
                display_name = f"{name} (LB=5)"
                stats = bt.run(df, func, lookback=5)
            else:
                display_name = name
                stats = bt.run(df, func)

            if not stats:
                continue

            results.append({
                "Strategy": display_name,
                "Return (%)": round(stats["total_return_pct"], 2),
                "Sharpe": round(stats["sharpe_ratio"], 3),
                "MDD (%)": round(stats["max_drawdown_pct"], 2),
                "Win Rate (%)": round(stats["win_rate_pct"], 2),
                "Trades": stats["total_trades"],
                "Final Equity": int(stats["final_equity"]),
            })
            emoji = "✅" if stats["total_return_pct"] > 0 else "❌"
            print(f"  {emoji} {display_name:<40} | Ret: {stats['total_return_pct']:+7.2f}% | Sharpe: {stats['sharpe_ratio']:+.3f} | MDD: {stats['max_drawdown_pct']:.2f}% | WR: {stats['win_rate_pct']:.1f}% | Trades: {stats['total_trades']}")

        except Exception as e:
            print(f"  ❌ {name}: {e}")
            traceback.print_exc()

    # 4. 輸出排名表
    if results:
        res_df = pd.DataFrame(results).sort_values("Sharpe", ascending=False)
        output_file = "backtest_results_2025_present.csv"
        res_df.to_csv(output_file, index=False, encoding="utf-8-sig")

        print("\n" + "=" * 100)
        print(f"📊 2025 至今策略排名 (依 Sharpe Ratio)")
        print("=" * 100)
        print(res_df.to_string(index=False))
        print(f"\n結果已存至 {output_file}")
    else:
        print("沒有產生任何回測結果。")


if __name__ == "__main__":
    run_2025()
