"""
簡易台指期回測引擎 (Mini TAIEX Backtester)
==============================================
專為台指期設計的向量化回測框架。
支援多空操作、交易成本計算 (手續費 + 滑價)、部位管理。
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Callable
import traceback

class MiniTaiexBacktester:
    """
    台指期回測器

    Parameters
    ----------
    initial_capital : float  初始資金 (TWD)
    commission : float  單邊手續費 (TWD/口)
    slippage : float  單邊滑價 (點數)
    multiplier : float  契約乘數 (大台=200, 小台=50)
    """
    def __init__(self, initial_capital=500000, commission=20, slippage=1, multiplier=50):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.multiplier = multiplier

    def run(self, df: pd.DataFrame, strategy_func: Callable, **kwargs) -> Dict:
        """
        執行回測

        Parameters
        ----------
        df : pd.DataFrame  歷史資料 (需包含 Open, High, Low, Close)
        strategy_func : Callable  策略函數，接收 df 並回傳 signals Series
        **kwargs : dict  策略函數的參數

        Returns
        -------
        dict : 包含回測結果統計與權益曲線
        """
        # 1. 計算策略信號
        try:
            signals = strategy_func(df, **kwargs)
        except Exception as e:
            print(f"策略執行錯誤: {e}")
            traceback.print_exc()
            return {}

        # 2. 轉換信號為持倉 (Position)
        # signal: 1=多, -1=空, 0=平倉
        # position: 1=持有各多單, -1=持有空單, 0=空手
        # shift(1) 代表訊號產生後，下一根 K 棒開盤才動作
        position = signals.shift(1).fillna(0)

        # 3. 計算每日損益
        # 價格變動點數
        price_change = df["Close"].diff()

        # 持倉損益 (點數)
        # 今天收盤損益 = 昨天持倉 * 今天價格變動
        strategy_points = position.shift(1) * price_change

        # 4. 計算交易成本
        # 交易發生點：當持倉改變時
        trades = position.diff().abs().fillna(0)
        
        # 成本 = 交易次數 * (手續費 + 滑價成本)
        # 滑價成本 = 滑價點數 * 乘數
        cost_per_trade = self.commission + self.slippage * self.multiplier
        total_costs = trades * cost_per_trade

        # 5. 計算淨損益 (TWD)
        strategy_pnl = strategy_points * self.multiplier - total_costs
        strategy_pnl = strategy_pnl.fillna(0)

        # 累計權益
        equity_curve = self.initial_capital + strategy_pnl.cumsum()
        
        # 6. 計算統計指標
        total_return = (equity_curve.iloc[-1] - self.initial_capital) / self.initial_capital
        total_trades = trades.sum()
        
        # 勝率 (以日為單位簡化計算，或僅計算有交易的日子)
        winning_days = strategy_pnl[strategy_pnl > 0].count()
        losing_days = strategy_pnl[strategy_pnl < 0].count()
        win_rate = winning_days / (winning_days + losing_days) if (winning_days + losing_days) > 0 else 0

        # Max Drawdown
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # Sharpe Ratio (假設無風險利率=0，年化=252)
        daily_return = strategy_pnl / self.initial_capital
        sharpe = (daily_return.mean() / daily_return.std()) * (252 ** 0.5) if daily_return.std() != 0 else 0

        # 年化報酬率 (CAGR)
        days = (df.index[-1] - df.index[0]).days
        years = days / 365.25
        if years > 0 and self.initial_capital > 0:
             ann_return_pct = ((equity_curve.iloc[-1] / self.initial_capital) ** (1 / years) - 1) * 100
        else:
             ann_return_pct = 0

        return {
            "total_return_pct": total_return * 100,
            "ann_return_pct": ann_return_pct,
            "total_trades": int(total_trades),
            "win_rate_pct": win_rate * 100,
            "max_drawdown_pct": max_drawdown * 100,
            "sharpe_ratio": sharpe,
            "equity_curve": equity_curve,
            "final_equity": equity_curve.iloc[-1]
        }
