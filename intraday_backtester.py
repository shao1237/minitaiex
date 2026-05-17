"""
當沖回測引擎 (Intraday / Day-Trading Backtester)
===================================================
專為 5 分鐘 K 線 + 當沖模式設計：
  - 每日收盤前強制平倉（不留倉過夜）
  - 計算每筆交易（不是每根 K 棒）的勝率
  - 統計當沖專用指標：平均持倉時間、每日交易次數、Profit Factor
"""
import pandas as pd
import numpy as np
from typing import Dict, Callable
import traceback


class IntradayBacktester:
    """
    當沖回測器

    Parameters
    ----------
    initial_capital : float  初始資金 (TWD)
    commission : float  單邊手續費 (TWD/口)
    slippage : float  單邊滑價 (點數)
    multiplier : float  契約乘數 (小台=50)
    force_close_time : str  強制平倉時間 (HH:MM)
    """

    def __init__(self, initial_capital=500_000, commission=20, slippage=1,
                 multiplier=50, force_close_time="13:30"):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.multiplier = multiplier
        self.force_close_time = pd.Timestamp(force_close_time).time()

    def run(self, df: pd.DataFrame, strategy_func: Callable, **kwargs) -> Dict:
        """
        執行當沖回測

        Parameters
        ----------
        df : pd.DataFrame  5 分鐘 K 線（index=datetime，含 Open/High/Low/Close/Volume）
        strategy_func : Callable  策略函數

        Returns
        -------
        dict : 回測結果
        """
        # 1. 計算策略信號
        try:
            signals = strategy_func(df, **kwargs)
        except Exception as e:
            print(f"策略執行錯誤: {e}")
            traceback.print_exc()
            return {}

        # 2. 訊號延遲一根 K 棒（信號出現 → 下一根開盤執行）
        raw_position = signals.shift(1).fillna(0).astype(int)

        # 3. 強制每日收盤平倉
        position = raw_position.copy()
        dates = df.index.date
        unique_dates = sorted(set(dates))

        for d in unique_dates:
            day_mask = dates == d
            day_idx = df.index[day_mask]
            if len(day_idx) == 0:
                continue

            # 找到收盤前的 bar（>= force_close_time 的第一根）
            close_bars = day_idx[day_idx.time >= self.force_close_time]
            if len(close_bars) > 0:
                close_from = close_bars[0]
                position.loc[close_from:day_idx[-1]] = 0

        # 4. 逐 bar 計算損益
        price_change = df["Close"].diff()
        bar_pnl_points = position.shift(1).fillna(0) * price_change

        # 5. 交易成本（每次換倉）
        trades_indicator = position.diff().abs().fillna(0)
        cost_per_trade = self.commission + self.slippage * self.multiplier
        total_costs = trades_indicator * cost_per_trade

        # 6. 淨損益
        bar_pnl = bar_pnl_points * self.multiplier - total_costs
        bar_pnl = bar_pnl.fillna(0)

        # 7. 權益曲線
        equity_curve = self.initial_capital + bar_pnl.cumsum()

        # 8. 逐筆交易統計
        trade_list = self._extract_trades(df, position)

        # 9. 統計指標
        stats = self._calc_stats(equity_curve, bar_pnl, trade_list, df)
        stats["equity_curve"] = equity_curve

        return stats

    def _extract_trades(self, df, position) -> list:
        """從持倉變化中提取每筆交易（進場→出場）"""
        trades = []
        in_trade = False
        entry_bar = None
        entry_price = 0
        direction = 0

        for i in range(1, len(df)):
            prev_pos = position.iloc[i - 1]
            curr_pos = position.iloc[i]

            # 開倉
            if prev_pos == 0 and curr_pos != 0:
                in_trade = True
                entry_bar = df.index[i]
                entry_price = df["Close"].iloc[i]
                direction = curr_pos

            # 平倉或反手
            elif in_trade and curr_pos != direction:
                exit_price = df["Close"].iloc[i]
                pnl_points = (exit_price - entry_price) * direction
                pnl_cash = (pnl_points * self.multiplier
                            - self.commission * 2
                            - self.slippage * self.multiplier * 2)

                duration = (df.index[i] - entry_bar).total_seconds() / 60

                trades.append({
                    "entry_time": entry_bar,
                    "exit_time": df.index[i],
                    "direction": "LONG" if direction == 1 else "SHORT",
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_points": pnl_points,
                    "pnl_cash": pnl_cash,
                    "duration_min": duration,
                })

                # 如果是反手，立即開新倉
                if curr_pos != 0:
                    entry_bar = df.index[i]
                    entry_price = df["Close"].iloc[i]
                    direction = curr_pos
                else:
                    in_trade = False
                    direction = 0

        return trades

    def _calc_stats(self, equity_curve, bar_pnl, trade_list, df) -> Dict:
        """計算統計指標"""
        total_return = (equity_curve.iloc[-1] - self.initial_capital) / self.initial_capital

        # 交易統計
        n_trades = len(trade_list)
        if n_trades > 0:
            pnl_list = [t["pnl_cash"] for t in trade_list]
            wins = [p for p in pnl_list if p > 0]
            losses = [p for p in pnl_list if p <= 0]

            win_rate = len(wins) / n_trades * 100
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0
            profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float("inf")
            avg_duration = np.mean([t["duration_min"] for t in trade_list])

            # 每日交易次數
            trading_days = len(set(df.index.date))
            trades_per_day = n_trades / trading_days if trading_days > 0 else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            avg_duration = 0
            trades_per_day = 0

        # Max Drawdown
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # Sharpe (用 bar 級別損益，年化以 5 分 K 棒數估算)
        # 一天 ≈ 60 根 5 分 K，一年 ≈ 250 天 → 15000 根
        bars_per_year = 60 * 250
        daily_return = bar_pnl / self.initial_capital
        sharpe = (daily_return.mean() / daily_return.std()) * (bars_per_year ** 0.5) if daily_return.std() != 0 else 0

        return {
            "total_return_pct": total_return * 100,
            "total_trades": n_trades,
            "win_rate_pct": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "avg_duration_min": avg_duration,
            "trades_per_day": trades_per_day,
            "max_drawdown_pct": max_drawdown * 100,
            "sharpe_ratio": sharpe,
            "final_equity": equity_curve.iloc[-1],
            "trade_list": trade_list,
        }
