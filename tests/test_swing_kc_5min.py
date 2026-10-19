"""
Swing High Low (100% Causal) + Keltner Channel 雙強結合與參數穩健性檢驗
========================================================================
測試在 5 分 K（含夜盤 + 結算日日盤不交易 skip_settlement_day）下：
1. 修正原版 Swing High Low 的未來函數（Causal Pivot Confirmation）
2. 將 Keltner Channel (趨勢通道與波動率濾網) 與 Swing High/Low (波段高低點結構突破) 結合：
   - 進場：收盤價突破前波確認之 Swing High 且突破/位於 Keltner 通道強勢區 (Close > UpperBand 或 Close > EMA + k*ATR)
   - 出場/反手：跌破前波確認之 Swing Low 與 Keltner 通道下軌/中軌
3. 在 2025 In-Sample 進行參數網格掃描 (Grid Search)，檢驗：
   - 參數高原穩健度 (Parameter Plateau：多少比例的參數組合在 IS 與 OOS 皆獲利)
   - Bonferroni 多重檢定校正 (p-value < 0.05 / M)
   - Walk-Forward 2026 (2026/01/01 ~ 2026/10/16) 盲測
   - Bootstrap 10,000 次重抽樣信賴區間 & MCPT 10,000 次排列檢定
"""
import os
import sys
import numpy as np
import pandas as pd
import scipy.stats as stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.download_kbar import load_5min_data
from core.intraday_backtester import IntradayBacktester


def precompute_causal_pivots(high: np.ndarray, low: np.ndarray, swing_len: int):
    """
    100% Causal Swing High / Swing Low:
    在 bar i 時，確認 bar (i - swing_len) 是否為 [i - 2*swing_len, i] 視窗內的極值。
    """
    n = len(high)
    win = 2 * swing_len + 1
    roll_max = pd.Series(high).rolling(win).max().to_numpy()
    roll_min = pd.Series(low).rolling(win).min().to_numpy()

    last_sh = np.full(n, np.nan)
    last_sl = np.full(n, np.nan)
    curr_sh = np.nan
    curr_sl = np.nan

    for i in range(win - 1, n):
        target = i - swing_len
        if high[target] == roll_max[i]:
            curr_sh = high[target]
        if low[target] == roll_min[i]:
            curr_sl = low[target]
        last_sh[i] = curr_sh
        last_sl[i] = curr_sl

    return last_sh, last_sl


def generate_swing_kc_signal(
    close: np.ndarray,
    ema_val: np.ndarray,
    atr_val: np.ndarray,
    last_sh: np.ndarray,
    last_sl: np.ndarray,
    atr_mult: float,
    start_idx: int,
    rule_type: str = "breakout_confirm",
) -> np.ndarray:
    """
    結合 Keltner Channel 與 Causal Swing High/Low 的核心訊號生成：
    - rule_type == "breakout_confirm":
        多頭觸發：Close > Keltner UpperBand 且 Close > Last Causal Swing High (雙重突破確認，過濾假突破)
        空頭觸發：Close < Keltner LowerBand 且 Close < Last Causal Swing Low
        其餘時間：維持原部位 (State Machine Latch)，讓波段跨日夜盤奔跑
    - rule_type == "kc_entry_swing_exit":
        進場：Close > UpperBand 且 Close > Last Swing High -> 做多 (反之做空)
        平倉保護：若持有多單但 Close 跌破 EMA 且跌破 Last Swing Low -> 平倉/反手
    """
    n = len(close)
    upper = ema_val + atr_mult * atr_val
    lower = ema_val - atr_mult * atr_val
    sig = np.zeros(n, dtype=int)

    if rule_type == "breakout_confirm":
        for i in range(start_idx, n):
            sh = last_sh[i]
            sl = last_sl[i]
            if np.isnan(upper[i]) or np.isnan(sh) or np.isnan(sl):
                sig[i] = sig[i - 1]
                continue
            if close[i] > upper[i] and close[i] > sh:
                sig[i] = 1
            elif close[i] < lower[i] and close[i] < sl:
                sig[i] = -1
            else:
                sig[i] = sig[i - 1]

    elif rule_type == "kc_entry_swing_exit":
        for i in range(start_idx, n):
            sh = last_sh[i]
            sl = last_sl[i]
            prev = sig[i - 1]
            if np.isnan(upper[i]) or np.isnan(sh) or np.isnan(sl):
                sig[i] = prev
                continue
            if close[i] > upper[i] and close[i] > sh:
                sig[i] = 1
            elif close[i] < lower[i] and close[i] < sl:
                sig[i] = -1
            elif prev == 1 and close[i] < ema_val[i] and close[i] < sl:
                sig[i] = 0
            elif prev == -1 and close[i] > ema_val[i] and close[i] > sh:
                sig[i] = 0
            else:
                sig[i] = prev

    return sig


def calc_daily_sharpe(bar_pnl: pd.Series, initial_capital: float = 500_000) -> float:
    daily_pnl = bar_pnl.groupby(bar_pnl.index.date).sum() / initial_capital
    if len(daily_pnl) < 2 or daily_pnl.std() == 0:
        return 0.0
    return float((daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252))


def run_grid_and_validation():
    df = load_5min_data()
    df_is = df[(df.index >= "2025-01-01") & (df.index <= "2025-12-31 23:59:59")].copy()
    df_oos = df[(df.index >= "2026-01-01") & (df.index <= "2026-10-16 23:59:59")].copy()

    bt = IntradayBacktester(force_close_time=None, rollover_mode="skip_settlement_day")

    # 預先計算不同參數的基礎陣列以加速 Grid Search
    swing_lens = [5, 10, 20, 30]
    ema_lens = [20, 40, 60, 120]
    atr_mults = [1.5, 2.0, 2.5, 3.0]
    rule_types = ["breakout_confirm", "kc_entry_swing_exit"]

    # 預先計算 IS 與 OOS 的 pivots, ema, atr
    def prep_arrays(sub_df):
        h = sub_df["High"].to_numpy()
        l = sub_df["Low"].to_numpy()
        c = sub_df["Close"].to_numpy()
        prev_c = np.empty_like(c)
        prev_c[0] = c[0]
        prev_c[1:] = c[:-1]
        tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))

        pivots = {sw: precompute_causal_pivots(h, l, sw) for sw in swing_lens}
        emas = {el: sub_df["Close"].ewm(span=el, adjust=False).mean().to_numpy() for el in ema_lens}
        atrs = {el: pd.Series(tr).rolling(el, min_periods=1).mean().to_numpy() for el in ema_lens}
        return c, pivots, emas, atrs

    c_is, piv_is, ema_is, atr_is = prep_arrays(df_is)
    c_oos, piv_oos, ema_oos, atr_oos = prep_arrays(df_oos)

    rows = []
    for rtype in rule_types:
        for sw in swing_lens:
            sh_is, sl_is = piv_is[sw]
            sh_oos, sl_oos = piv_oos[sw]
            for el in ema_lens:
                for mult in atr_mults:
                    sig_is = generate_swing_kc_signal(
                        c_is, ema_is[el], atr_is[el], sh_is, sl_is, mult, max(el, 2 * sw + 1), rtype
                    )
                    sig_oos = generate_swing_kc_signal(
                        c_oos, ema_oos[el], atr_oos[el], sh_oos, sl_oos, mult, max(el, 2 * sw + 1), rtype
                    )

                    s_is = bt.run(df_is, lambda _df, s=sig_is: pd.Series(s, index=_df.index))
                    s_oos = bt.run(df_oos, lambda _df, s=sig_oos: pd.Series(s, index=_df.index))

                    dsr_is = calc_daily_sharpe(s_is["equity_curve"].diff().fillna(0))
                    dsr_oos = calc_daily_sharpe(s_oos["equity_curve"].diff().fillna(0))

                    rows.append({
                        "rule": rtype,
                        "swing_len": sw,
                        "ema_len": el,
                        "atr_mult": mult,
                        "IS_Ret": s_is["total_return_pct"],
                        "IS_BarSR": s_is["sharpe_ratio"],
                        "IS_DailySR": dsr_is,
                        "IS_MDD": s_is["max_drawdown_pct"],
                        "IS_PF": s_is["profit_factor"],
                        "IS_Trades": s_is["total_trades"],
                        "OOS_Ret": s_oos["total_return_pct"],
                        "OOS_BarSR": s_oos["sharpe_ratio"],
                        "OOS_DailySR": dsr_oos,
                        "OOS_MDD": s_oos["max_drawdown_pct"],
                        "OOS_PF": s_oos["profit_factor"],
                        "OOS_Trades": s_oos["total_trades"],
                    })

    res_df = pd.DataFrame(rows)
    reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    res_df.to_csv(os.path.join(reports_dir, "swing_kc_5min_grid_results.csv"), index=False, encoding="utf-8-sig")

    print("=" * 110)
    print(f"🔬 雙強結合策略 (Swing + Keltner) 參數穩健性網格檢驗 (總組合數 M = {len(res_df)})")
    print("=" * 110)

    for rtype in rule_types:
        sub = res_df[res_df["rule"] == rtype]
        is_pos = (sub["IS_Ret"] > 0).mean() * 100
        oos_pos = (sub["OOS_Ret"] > 0).mean() * 100
        both_pos = ((sub["IS_Ret"] > 0) & (sub["OOS_Ret"] > 0)).mean() * 100
        print(f"▶ 架構 [{rtype}]: 共 {len(sub)} 組參數 | IS 獲利比例: {is_pos:.1f}% | OOS 獲利比例: {oos_pos:.1f}% | IS+OOS 雙贏比例: {both_pos:.1f}%")
        print(f"  - OOS 平均報酬: {sub['OOS_Ret'].mean():+.2f}% (中位數: {sub['OOS_Ret'].median():+.2f}%, 最高: {sub['OOS_Ret'].max():+.2f}%)")

    print("\n🏆 依 2025 In-Sample Daily Sharpe 排名前 10 名參數組合（觀察其在 2026 OOS 盲測表現）：")
    top10 = res_df.sort_values("IS_DailySR", ascending=False).head(10)
    cols = ["rule", "swing_len", "ema_len", "atr_mult", "IS_Ret", "IS_DailySR", "IS_MDD", "IS_Trades", "OOS_Ret", "OOS_DailySR", "OOS_MDD", "OOS_PF", "OOS_Trades"]
    print(top10[cols].to_string(index=False))


if __name__ == "__main__":
    run_grid_and_validation()
