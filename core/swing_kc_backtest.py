"""
Swing High Low + Keltner Channel 策略回測引擎 (數據純化與完全參數化優化版)
========================================================================
- 實作合約換倉機制 (結算日 13:25 強制平倉，暫停交易至 15:00 換新合約)
- 摩擦成本真實化：單邊手續費、滑價，以及成交金額的 10 萬分之 2 期交稅
- 支援核心參數完全參數化 (EMA 週期、ATR 乘數、左右確認 K 線數)
- 支援 EMA 斜率絕對門檻過濾與停損冷卻期 (Cooldown)
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

def compute_indicators(df: pd.DataFrame, ema_len: int = 20, atr_len: int = 20, kc_mult: float = 2.0, slope_period: int = 5) -> pd.DataFrame:
    """計算技術指標"""
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
    
    # 2. EMA 斜率標準化
    df["EMA_Slope"] = (df["Basis"] - df["Basis"].shift(slope_period)) / df["ATR"]
    
    # 3. 大趨勢濾網
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
    """ Swing High Low + Keltner Channel 逐棒回測器 (真實成本與結算日換倉) """
    def __init__(self, initial_capital: float = 500_000, commission: float = 20, slippage: float = 1.0, multiplier: float = 50.0):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.multiplier = multiplier
        self.tax_rate = 0.00002  # 期交稅十萬分之二
        
    def _calc_one_way_cost(self, size: float, price: float) -> float:
        """ 計算單邊真實交易成本 (手續費 + 滑價 + 期交稅) """
        # size: 1.0 代表 2 口小台 (100點乘數), 0.5 代表 1 口小台 (50點乘數)
        contracts = size / 0.5
        cost_fixed = contracts * (self.commission + self.slippage * self.multiplier)
        cost_tax = price * (contracts * self.multiplier) * self.tax_rate
        return cost_fixed + cost_tax

    def run_backtest(self, df: pd.DataFrame, 
                     ema_len: int = 20, 
                     kc_mult: float = 2.0, 
                     swing_len: int = 2,
                     buffer_pct: float = 0.0015, 
                     long_only: bool = True, 
                     slope_threshold: float = 0.04, 
                     cooldown_bars: int = 12, 
                     early_break_even: bool = False) -> Dict:
        """ 執行完全參數化的逐棒回測 """
        # 1. 預先計算指標
        df_indicators = compute_indicators(df, ema_len=ema_len, atr_len=ema_len, kc_mult=kc_mult)
        
        # 轉成 numpy arrays
        dates = df_indicators.index.values
        ts_dates = pd.to_datetime(dates)
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
        
        cooldown_counter = 0
        trades_log = []
        equity = [self.initial_capital]
        current_equity = self.initial_capital
        
        recent_swing_highs = []
        recent_swing_lows = []
        
        start_bar = 2 * swing_len
        
        # 逐棒執行
        for t in range(start_bar, n_bars - 1):
            curr_time = ts_dates[t]
            next_open = opens[t+1]
            next_high = highs[t+1]
            next_low = lows[t+1]
            next_close = closes[t+1]
            
            if cooldown_counter > 0:
                cooldown_counter -= 1
                
            # --- A. 合約結算日換倉邏輯 (每月第三個星期三) ---
            # 結算日: weekday == 2 且日期在 15~21 號之間
            is_settlement_day = (curr_time.weekday() == 2) and (15 <= curr_time.day <= 21)
            is_rollover_time = is_settlement_day and (curr_time.hour == 13 and curr_time.minute >= 25)
            
            # 結算日 13:25 強制平倉
            if is_rollover_time and position != 0.0:
                exit_p = closes[t]  # 以當前結算價平倉
                direction = "LONG" if position > 0 else "SHORT"
                pnl = 0.0
                
                # 計算真實出場成本
                cost_exit = self._calc_one_way_cost(abs(position), exit_p)
                
                if position > 0:
                    pnl = (exit_p - entry_price) * position * self.multiplier - cost_exit
                else:
                    pnl = (entry_price - exit_p) * abs(position) * self.multiplier - cost_exit
                    
                current_equity += pnl
                trades_log.append({
                    "entry_time": entry_time,
                    "exit_time": curr_time,
                    "direction": direction,
                    "entry_price": entry_price,
                    "exit_price": exit_p,
                    "pnl": pnl,
                    "type": "ROLLOVER_CLOSE",
                    "size": abs(position)
                })
                position = 0.0
                
            # --- 1. 計算參數化波段高低點 ---
            target_bar = t - swing_len
            
            # 判斷是否為 Swing Low
            is_swing_low = True
            for i in range(2 * swing_len + 1):
                if i != swing_len:
                    if lows[t - i] <= lows[target_bar]:
                        is_swing_low = False
                        break
            
            swing_low_near_support = is_swing_low and (lows[target_bar] <= basis[target_bar]) and (lows[target_bar] >= lower[target_bar] - 0.2 * atr[target_bar])
            if is_swing_low:
                recent_swing_lows.append((dates[target_bar], lows[target_bar]))
                if len(recent_swing_lows) > 10:
                    recent_swing_lows.pop(0)
            
            # 判斷是否為 Swing High
            is_swing_high = True
            for i in range(2 * swing_len + 1):
                if i != swing_len:
                    if highs[t - i] >= highs[target_bar]:
                        is_swing_high = False
                        break
                        
            swing_high_near_resistance = is_swing_high and (highs[target_bar] >= basis[target_bar]) and (highs[target_bar] <= upper[target_bar] + 0.2 * atr[target_bar])
            if is_swing_high:
                recent_swing_highs.append((dates[target_bar], highs[target_bar]))
                if len(recent_swing_highs) > 10:
                    recent_swing_highs.pop(0)
            
            # --- 2. 持倉出場與移動止損判定 ---
            
            # A. 做多持倉處理
            if position > 0:
                if next_low <= stop_loss:
                    exit_p = min(next_open, stop_loss)
                    cost_exit = self._calc_one_way_cost(position, exit_p)
                    pnl = (exit_p - entry_price) * position * self.multiplier - cost_exit
                    current_equity += pnl
                    
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
                        cooldown_counter = cooldown_bars
                
                elif position == 1.0 and next_high >= upper[t+1]:
                    exit_p = max(next_open, upper[t+1])
                    cost_exit = self._calc_one_way_cost(0.5, exit_p)
                    pnl_half = (exit_p - entry_price) * 0.5 * self.multiplier - cost_exit
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
                    stop_loss = entry_price
                
                elif position == 1.0 and early_break_even:
                    if is_swing_high:
                        prev_shs = [sh[1] for sh in recent_swing_highs if sh[0] < entry_time]
                        if prev_shs:
                            prior_sh = prev_shs[-1]
                            if highs[target_bar] < prior_sh:
                                stop_loss = entry_price
                    if ema_slope[t] < slope_threshold * 0.3:
                        stop_loss = entry_price
                
                if position == 0.5:
                    if swing_low_near_support:
                        new_sl = lows[target_bar] * (1 - buffer_pct)
                        if new_sl > stop_loss:
                            stop_loss = new_sl
                            
            # B. 做空持倉處理
            elif position < 0:
                if next_high >= stop_loss:
                    exit_p = max(next_open, stop_loss)
                    cost_exit = self._calc_one_way_cost(abs(position), exit_p)
                    pnl = (entry_price - exit_p) * abs(position) * self.multiplier - cost_exit
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
                        
                elif position == -1.0 and next_low <= lower[t+1]:
                    exit_p = min(next_open, lower[t+1])
                    cost_exit = self._calc_one_way_cost(0.5, exit_p)
                    pnl_half = (entry_price - exit_p) * 0.5 * self.multiplier - cost_exit
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
                    
                elif position == -1.0 and early_break_even:
                    if is_swing_low:
                        prev_sls = [sl[1] for sl in recent_swing_lows if sl[0] < entry_time]
                        if prev_sls:
                            prior_sl = prev_sls[-1]
                            if lows[target_bar] > prior_sl:
                                stop_loss = entry_price
                    if ema_slope[t] > -slope_threshold * 0.3:
                        stop_loss = entry_price
                        
                if position == -0.5:
                    if swing_high_near_resistance:
                        new_sh = highs[target_bar] * (1 + buffer_pct)
                        if new_sh < stop_loss:
                            stop_loss = new_sh
                            
            # --- 3. 檢查進場信號 (空手、非冷卻期，且非結算日換倉時段) ---
            # 結算日 13:25 到 15:00 禁止開倉
            is_banned_by_rollover = is_settlement_day and (curr_time.hour == 13 and curr_time.minute >= 25)
            
            if position == 0.0 and cooldown_counter == 0 and not is_banned_by_rollover:
                is_trend_long = (ema_slope[t] > slope_threshold) and (pct_above_ema[t] >= 0.6) and touched_upper[t]
                
                if is_trend_long and pulled_back_long[t] and swing_low_near_support:
                    position = 1.0
                    entry_price = next_open
                    entry_time = dates[t+1]
                    stop_loss = lows[target_bar] * (1 - buffer_pct)
                    #真實開倉成本
                    cost_entry = self._calc_one_way_cost(1.0, entry_price)
                    current_equity -= cost_entry
                    
                elif not long_only:
                    is_trend_short = (ema_slope[t] < -slope_threshold) and (pct_below_ema[t] >= 0.6) and touched_lower[t]
                    if is_trend_short and pulled_back_short[t] and swing_high_near_resistance:
                        position = -1.0
                        entry_price = next_open
                        entry_time = dates[t+1]
                        stop_loss = highs[target_bar] * (1 + buffer_pct)
                        cost_entry = self._calc_one_way_cost(1.0, entry_price)
                        current_equity -= cost_entry
            
            # 估算未實現損益併入 Equity
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
    
    n_trades = len(trades)
    wins = [t["pnl"] for t in trades if t["pnl"] > 0]
    losses = [t["pnl"] for t in trades if t["pnl"] <= 0]
    
    win_rate = len(wins) / n_trades * 100 if n_trades > 0 else 0
    profit_factor = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float("inf")
    
    rolling_max = equity_series.cummax()
    drawdown = (equity_series - rolling_max) / rolling_max * 100
    max_drawdown = drawdown.min()
    
    days = (df.index[-1] - df.index[0]).days
    years = days / 365.25
    ann_return = ((final_equity / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0
    
    daily_equity = equity_series.resample("D").last().ffill()
    daily_returns = daily_equity.pct_change().dropna()
    sharpe = (daily_returns.mean() / daily_returns.std() * (252 ** 0.5)) if daily_returns.std() != 0 else 0
    
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
