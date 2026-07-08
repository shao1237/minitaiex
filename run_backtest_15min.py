"""
15 分 K 跨日波段回測 — 全策略排名
==============================
"""
import sys
import os
import traceback
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategies import STRATEGIES
from intraday_backtester import IntradayBacktester

def load_15min_data():
    data_path = os.path.join(os.path.dirname(__file__), "data", "mxf_15min.parquet")
    if not os.path.exists(data_path):
        return pd.DataFrame()
    df = pd.read_parquet(data_path)
    return df

def run_intraday_backtest():
    print("=" * 100)
    print("📊 15 分 K 跨日波段回測系統")
    print("=" * 100)

    df = load_15min_data()
    if df.empty:
        print("❌ 無法載入 15 分 K 資料，請先執行: python aggregate_15min.py")
        return

    # 時間範圍限制今年到現在 (2026)
    df = df[df.index >= "2026-01-01"]

    trading_days = len(set(df.index.date))
    first_day = df.index[0].strftime("%Y-%m-%d")
    last_day = df.index[-1].strftime("%Y-%m-%d")
    total_bars = len(df)

    print(f"\n回測期間: {first_day} ~ {last_day}")
    print(f"交易日數: {trading_days} 天 | K 棒數: {total_bars} 根（15 分鐘）")
    print(f"合約: 小台 MXF（乘數=50, 手續費=20/邊, 滑價=1 點/邊）")
    print(f"規則: 跨日波段持倉（含夜盤），結算日 13:25 強制平倉暫停")
    print("=" * 100)

    bt = IntradayBacktester(
        initial_capital=500_000,
        commission=20,
        slippage=1,
        multiplier=50,
    )

    results = []

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
                "Win Rate (%)": round(stats["win_rate_pct"], 1),
                "Trades": stats["total_trades"],
                "Profit Factor": round(stats["profit_factor"], 2) if stats["profit_factor"] != float("inf") else "∞",
                "Avg Duration (min)": round(stats["avg_duration_min"], 1),
                "Trades/Day": round(stats["trades_per_day"], 1),
                "Final Equity": int(stats["final_equity"]),
            })

            emoji = "✅" if stats["total_return_pct"] > 0 else "❌"
            print(
                f"  {emoji} {display_name:<40} | "
                f"Ret: {stats['total_return_pct']:+8.2f}% | "
                f"Sharpe: {stats['sharpe_ratio']:+.3f} | "
                f"WR: {stats['win_rate_pct']:.1f}% | "
                f"PF: {stats['profit_factor']:.2f} | "
                f"Trades: {stats['total_trades']:>4} | "
                f"Avg: {stats['avg_duration_min']:.0f}min"
            )

        except Exception as e:
            print(f"  ❌ {name}: {e}")
            traceback.print_exc()

    if results:
        res_df = pd.DataFrame(results)
        res_df_sorted = res_df.copy()
        res_df_sorted["_sharpe"] = pd.to_numeric(res_df_sorted["Sharpe"], errors="coerce")
        res_df_sorted = res_df_sorted.sort_values("_sharpe", ascending=False).drop(columns=["_sharpe"])

        output_file = "backtest_15min_continuous_results.csv"
        res_df_sorted.to_csv(output_file, index=False, encoding="utf-8-sig")

        print("\n" + "=" * 100)
        print(f"📊 15 分 K 跨日波段策略排名 (依 Sharpe Ratio)")
        print("=" * 100)
        display_cols = ["Strategy", "Return (%)", "Sharpe", "MDD (%)",
                        "Win Rate (%)", "Trades", "Profit Factor",
                        "Trades/Day", "Final Equity"]
        print(res_df_sorted[display_cols].to_string(index=False))
        print(f"\n完整結果已存至 {output_file}")

if __name__ == "__main__":
    run_intraday_backtest()
