"""
Swing High Low + Keltner Channel 策略回測引擎 (優化版)
======================================================
- 支援 EMA 斜率絕對門檻過濾 (避免在走平的盤整區進場)
- 支援停損冷卻期 (Cooldown) 與同平坦區重複訊號忽略
- 支援反彈未過前高或 EMA 走平時的提早保本/小賠出場機制
- 支援 5 分 K 與 15 分 K 逐棒狀態機回測
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

def compute_indicators(df: pd.DataFrame, ema_len: int = 20, atr_len: int = 20, kc_mult: float = 2.0, slope_period: int = 5) -> pd.DataFrame:
    """計算策略所需的所有技術指標，並加入斜率計算"""
    df = df.copy()
    
    # 1. 肯特納通道中軌與 ATR
    df["Basis"] = df["Close"].ewm(span=ema_len, adjust=False).mean()
    
    # 計算 ATR
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(window=atr_len).mean()
    
    # 通道上下軌
    df["UpperBand"] = df["Basis"] + kc_mult * df["ATR"]
    df["LowerBand"] = df["Basis"] - kc_mult * df["ATR"]
    
    # 2. EMA 斜率標準化 (以 ATR 衡量，避免價格絕對值影響)
    # 斜率 = (Basis_t - Basis_{t - slope_period}) / ATR_t
    df["EMA_Slope"] = (df["Basis"] - df["Basis"].shift(slope_period)) / df["ATR"]
    
    # 3. 大趨勢濾網 (過去 20 根 K 棒中，Close 高於/低於中軌的比例)
    close_above_ema = (df["Close"] > df["Basis"]).astype(int)
    close_below_ema = (df["Close"] < df["Basis"]).astype(int)
    df["Pct_Above_EMA"] = close_above_ema.rolling(20).mean()
    df["Pct_Below_EMA"] = close_below_ema.rolling(20).mean()
    
    # 過去 20 根 K 棒中，是否曾強勢觸及/突破上軌或下軌
    high_touch_upper = (df["High"] >= df["UpperBand"]).astype(int)
    low_touch_lower = (df["Low"] <= df["LowerBand"]).astype(int)
    df["Touched_Upper"] = high_touch_upper.rolling(20).max() > 0
    df["Touched_Lower"] = low_touch_lower.rolling(20).max() > 0
    
    # 4. 價值區拉回判定
    df["In_Value_Long"] = (df["Low"] <= df["Basis"]) & (df["Close"] >= df["LowerBand"])
    df["Pulled_Back_Long"] = df["In_Value_Long"].rolling(10).max() > 0
    
    df["In_Value_Short"] = (df["High"] >= df["Basis"]) & (df["Close"] <= df["UpperBand"])
    df["Pulled_Back_Short"] = df["In_Value_Short"].rolling(10).max() > 0
    
    return df

class SwingKeltnerBacktester:
    """ Swing High Low + Keltner Channel 逐棒回測器 (包含盤整過濾與提早保本) """
    def __init__(self, initial_capital: float = 500_000, commission: float = 20, slippage: float = 1.0, multiplier: float = 50.0):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.multiplier = multiplier
        
    def run_backtest(self, df: pd.DataFrame, buffer_pct: float = 0.0015, long_only: bool = True, 
                     slope_threshold: float = 0.05, cooldown_bars: int = 20, early_break_even: bool = True) -> Dict:
        """ 
        執行逐棒回測 
        
        Parameters
        ----------
        slope_threshold : float  中軌斜率門檻，大於此值才視為有趨勢
        cooldown_bars : int  停損後的冷卻 K 棒數 (避免在平坦盤整區連續被雙巴)
        early_break_even : bool  是否在反彈未過前高或中軌走平時立刻保本/小賠出場
        """
        # 1. 預先計算指標
        df_indicators = compute_indicators(df)
        
        # 轉成 numpy arrays 以加速
        dates = df_indicators.index.values
        opens = df_indicators["Open"].values
        highs = df_indicators["High"].values
        lows = df_indicators["Low"].values
        closes = df_indicators["Close"].values
        
        basis = df_indicators["Basis"].values
        upper = df_indicators["UpperBand"].values
        lower = df_indicators["LowerBand"].values
        atr = df_indicators["ATR"].values
        ema_slope = df_indicators["EMA_Slope"].values
        
        pct_above_ema = df_indicators["Pct_Above_EMA"].values
        pct_below_ema = df_indicators["Pct_Below_EMA"].values
        touched_upper = df_indicators["Touched_Upper"].values
        touched_lower = df_indicators["Touched_Lower"].values
        
        pulled_back_long = df_indicators["Pulled_Back_Long"].values
        pulled_back_short = df_indicators["Pulled_Back_Short"].values
        
        n_bars = len(df_indicators)
        
        # 回測狀態變數
        position = 0.0  # 1.0=多單滿倉, 0.5=多單半倉, -1.0=空單滿倉, -0.5=空單半倉
        entry_price = 0.0
        entry_time = None
        stop_loss = 0.0
        
        # 用於追蹤前高的變數 (判斷是否過前高)
        prior_swing_high = 0.0
        prior_swing_low = 0.0
        
        # 冷卻期計數器 (大於 0 表示處於冷卻期禁止交易)
        cooldown_counter = 0
        
        # 記錄上一次交易是否為停損
        last_trade_is_loss = False
        
        trades_log = []
        equity = [self.initial_capital]
        current_equity = self.initial_capital
        one_way_cost = self.commission + self.slippage * self.multiplier
        
        # 歷史波段高低點紀錄 (用於判斷前高前低)
        recent_swing_highs = []
        recent_swing_lows = []
        
        # 逐棒執行
        for t in range(4, n_bars - 1):
            next_open = opens[t+1]
            next_high = highs[t+1]
            next_low = lows[t+1]
            next_close = closes[t+1]
            
            # 遞減冷卻計數器
            if cooldown_counter > 0:
                cooldown_counter -= 1
                
            # --- 1. 計算局部波段高低點 ---
            is_swing_low = (lows[t-2] < lows[t-4]) and (lows[t-2] < lows[t-3]) and (lows[t-2] < lows[t-1]) and (lows[t-2] < lows[t])
            swing_low_near_support = is_swing_low and (lows[t-2] <= basis[t-2]) and (lows[t-2] >= lower[t-2] - 0.2 * atr[t-2])
            if is_swing_low:
                recent_swing_lows.append((dates[t-2], lows[t-2]))
                if len(recent_swing_lows) > 10:
                    recent_swing_lows.pop(0)
            
            is_swing_high = (highs[t-2] > highs[t-4]) and (highs[t-2] > highs[t-3]) and (highs[t-2] > highs[t-1]) and (highs[t-2] > highs[t])
            swing_high_near_resistance = is_swing_high and (highs[t-2] >= basis[t-2]) and (highs[t-2] <= upper[t-2] + 0.2 * atr[t-2])
            if is_swing_high:
                recent_swing_highs.append((dates[t-2], highs[t-2]))
                if len(recent_swing_highs) > 10:
                    recent_swing_highs.pop(0)
            
            # --- 2. 持倉出場與移動止損判定 ---
            
            # A. 做多持倉處理
            if position > 0:
                # 檢查是否觸發停損 (以 next_low 檢查)
                if next_low <= stop_loss:
                    exit_p = min(next_open, stop_loss)
                    pnl = (exit_p - entry_price) * position * self.multiplier - (position * one_way_cost)
                    current_equity += pnl
                    
                    # 判斷是否為停損虧損 (而非移動止盈利潤)
                    is_loss = (exit_p < entry_price)
                    trades_log.append({
                        "entry_time": entry_time,
                        "exit_time": dates[t+1],
                        "direction": "LONG",
                        "entry_price": entry_price,
                        "exit_price": exit_p,
                        "pnl": pnl,
                        "type": "STOP_LOSS" if is_loss else "TRAILING_STOP",
                        "size": position
                    })
                    position = 0.0
                    
                    if is_loss:
                        cooldown_counter = cooldown_bars  # 觸發停損，進入冷卻期
                        last_trade_is_loss = True
                
                # 檢查是否觸發第一階段停利 (滿倉且觸及上軌)
                elif position == 1.0 and next_high >= upper[t+1]:
                    exit_p = max(next_open, upper[t+1])
                    pnl_half = (exit_p - entry_price) * 0.5 * self.multiplier - (0.5 * one_way_cost)
                    current_equity += pnl_half
                    
                    trades_log.append({
                        "entry_time": entry_time,
                        "exit_time": dates[t+1],
                        "direction": "LONG",
                        "entry_price": entry_price,
                        "exit_price": exit_p,
                        "pnl": pnl_half,
                        "type": "TAKE_PROFIT_1",
                        "size": 0.5
                    })
                    position = 0.5
                    stop_loss = entry_price  # 移到保本價
                
                # 實戰優化：反彈未過前高且 EMA 走平，拉保本停損或提早出場
                elif position == 1.0 and early_break_even:
                    # 尋找進場後新形成的 Swing High
                    if is_swing_high:
                        # 找出進場前的「前一個 Swing High」
                        prev_shs = [sh[1] for sh in recent_swing_highs if sh[0] < entry_time]
                        if prev_shs:
                            prior_sh = prev_shs[-1]
                            # 如果進場後反彈形成的 Swing High 低於進場前的前高
                            if highs[t-2] < prior_sh:
                                # 反彈未過前高，立即把停損拉至保本
                                stop_loss = entry_price
                                
                    # 或是當 EMA 斜率變平 (趨勢走平)
                    if ema_slope[t] < slope_threshold * 0.3:
                        stop_loss = entry_price
                
                # 移動止盈 (半倉狀態下，尋找新波段低點上調停損)
                if position == 0.5:
                    if swing_low_near_support:
                        new_sl = lows[t-2] * (1 - buffer_pct)
                        if new_sl > stop_loss:
                            stop_loss = new_sl
                            
            # B. 做空持倉處理
            elif position < 0:
                # 檢查是否觸發停損
                if next_high >= stop_loss:
                    exit_p = max(next_open, stop_loss)
                    pnl = (entry_price - exit_p) * abs(position) * self.multiplier - (abs(position) * one_way_cost)
                    current_equity += pnl
                    
                    is_loss = (exit_p > entry_price)
                    trades_log.append({
                        "entry_time": entry_time,
                        "exit_time": dates[t+1],
                        "direction": "SHORT",
                        "entry_price": entry_price,
                        "exit_price": exit_p,
                        "pnl": pnl,
                        "type": "STOP_LOSS" if is_loss else "TRAILING_STOP",
                        "size": abs(position)
                    })
                    position = 0.0
                    
                    if is_loss:
                        cooldown_counter = cooldown_bars
                        last_trade_is_loss = True
                        
                # 檢查是否觸發第一階段停利
                elif position == -1.0 and next_low <= lower[t+1]:
                    exit_p = min(next_open, lower[t+1])
                    pnl_half = (entry_price - exit_p) * 0.5 * self.multiplier - (0.5 * one_way_cost)
                    current_equity += pnl_half
                    
                    trades_log.append({
                        "entry_time": entry_time,
                        "exit_time": dates[t+1],
                        "direction": "SHORT",
                        "entry_price": entry_price,
                        "exit_price": exit_p,
                        "pnl": pnl_half,
                        "type": "TAKE_PROFIT_1",
                        "size": 0.5
                    })
                    position = -0.5
                    stop_loss = entry_price
                    
                # 實戰優化：反彈未過前低且 EMA 走平，提早拉保本
                elif position == -1.0 and early_break_even:
                    if is_swing_low:
                        prev_sls = [sl[1] for sl in recent_swing_lows if sl[0] < entry_time]
                        if prev_sls:
                            prior_sl = prev_sls[-1]
                            if lows[t-2] > prior_sl:  # 反彈低點沒跌破前低
                                stop_loss = entry_price
                                
                    if ema_slope[t] > -slope_threshold * 0.3:
                        stop_loss = entry_price
                        
                # 移動止盈 (半倉狀態下，尋找新波段高點下調停損)
                if position == -0.5:
                    if swing_high_near_resistance:
                        new_sh = highs[t-2] * (1 + buffer_pct)
                        if new_sh < stop_loss:
                            stop_loss = new_sh
                            
            # --- 3. 檢查進場信號 (只有在空手且不在冷卻期時) ---
            if position == 0.0 and cooldown_counter == 0:
                
                # 做多大趨勢濾網：中軌斜率大於門檻 + 60% K棒收在中軌上 + 曾碰觸上軌
                is_trend_long = (ema_slope[t] > slope_threshold) and (pct_above_ema[t] >= 0.6) and touched_upper[t]
                
                # 做多進場：趨勢向上 + 價值區拉回 + 收盤確認 Swing Low (且該 Low 在中軌到下軌之間)
                if is_trend_long and pulled_back_long[t] and swing_low_near_support:
                    position = 1.0
                    entry_price = next_open
                    entry_time = dates[t+1]
                    stop_loss = lows[t-2] * (1 - buffer_pct)
                    current_equity -= (1.0 * one_way_cost)
                    
                # 做空進場 (非 long_only)
                elif not long_only:
                    is_trend_short = (ema_slope[t] < -slope_threshold) and (pct_below_ema[t] >= 0.6) and touched_lower[t]
                    if is_trend_short and pulled_back_short[t] and swing_high_near_resistance:
                        position = -1.0
                        entry_price = next_open
                        entry_time = dates[t+1]
                        stop_loss = highs[t-2] * (1 + buffer_pct)
                        current_equity -= (1.0 * one_way_cost)
            
            # 計算未實現損益
            unrealized_pnl = 0.0
            if position > 0:
                unrealized_pnl = (next_close - entry_price) * position * self.multiplier
            elif position < 0:
                unrealized_pnl = (entry_price - next_close) * abs(position) * self.multiplier
                
            equity.append(current_equity + unrealized_pnl)
            
        while len(equity) < len(df_indicators):
            equity.append(equity[-1])
            
        df_indicators["Equity"] = equity
        return {
            "indicators": df_indicators,
            "trades": trades_log
        }

def calculate_stats(initial_capital: float, final_equity: float, equity_series: pd.Series, trades: List[Dict], df: pd.DataFrame) -> Dict:
    """計算回測的統計指標"""
    total_return = (final_equity - initial_capital) / initial_capital * 100
    
    # 交易統計
    n_trades = len(trades)
    wins = [t["pnl"] for t in trades if t["pnl"] > 0]
    losses = [t["pnl"] for t in trades if t["pnl"] <= 0]
    
    win_rate = len(wins) / n_trades * 100 if n_trades > 0 else 0
    profit_factor = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float("inf")
    
    # Max Drawdown
    rolling_max = equity_series.cummax()
    drawdown = (equity_series - rolling_max) / rolling_max * 100
    max_drawdown = drawdown.min()
    
    # 年化報酬率 (CAGR)
    days = (df.index[-1] - df.index[0]).days
    years = days / 365.25
    ann_return = ((final_equity / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0
    
    # Sharpe Ratio (用日變動率計算)
    daily_equity = equity_series.resample("D").last().ffill()
    daily_returns = daily_equity.pct_change().dropna()
    sharpe = (daily_returns.mean() / daily_returns.std() * (252 ** 0.5)) if daily_returns.std() != 0 else 0
    
    # 平均每筆獲利/虧損
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    
    return {
        "total_return_pct": total_return,
        "ann_return_pct": ann_return,
        "max_drawdown_pct": max_drawdown,
        "sharpe_ratio": sharpe,
        "win_rate_pct": win_rate,
        "profit_factor": profit_factor,
        "total_trades": n_trades,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "final_equity": final_equity
    }
