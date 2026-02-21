"""
TradingView 最熱門策略庫
=============================
將 TradingView「最熱門」榜單上的 15 種策略轉譯為 Python。
每個策略函數接收 DataFrame (OHLCV)，回傳 signal Series (1=多, -1=空, 0=空手)。
"""
import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════
# 工具函數
# ═══════════════════════════════════════════════════════════════

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _stdev(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).std()


def _linreg(series: pd.Series, period: int) -> pd.Series:
    """線性回歸值 (模擬 Pine Script linreg)"""
    result = pd.Series(np.nan, index=series.index)
    vals = series.values
    for i in range(period - 1, len(vals)):
        y = vals[i - period + 1: i + 1]
        if np.any(np.isnan(y)):
            continue
        x = np.arange(period)
        slope, intercept = np.polyfit(x, y, 1)
        result.iloc[i] = intercept + slope * (period - 1)
    return result


def _highest(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).max()


def _lowest(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).min()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


# ═══════════════════════════════════════════════════════════════
# 1. Smart Money Concepts (SMC) — LuxAlgo
#    簡化版：偵測 BOS/CHoCH + Order Block 反彈
# ═══════════════════════════════════════════════════════════════

def smart_money_concepts(df: pd.DataFrame, swing_len: int = 10) -> pd.Series:
    """
    Smart Money Concepts (SMC)
    - 用 swing high/low 偵測市場結構
    - BOS (Break of Structure) = 趨勢延續
    - CHoCH (Change of Character) = 趨勢翻轉
    """
    high, low, close = df["High"], df["Low"], df["Close"]
    signal = pd.Series(0, index=df.index)

    # 偵測 swing high/low
    swing_high = pd.Series(np.nan, index=df.index)
    swing_low = pd.Series(np.nan, index=df.index)

    for i in range(swing_len, len(df) - swing_len):
        if high.iloc[i] == high.iloc[i - swing_len: i + swing_len + 1].max():
            swing_high.iloc[i] = high.iloc[i]
        if low.iloc[i] == low.iloc[i - swing_len: i + swing_len + 1].min():
            swing_low.iloc[i] = low.iloc[i]

    # 填充最近的 swing 點
    last_sh = swing_high.ffill()
    last_sl = swing_low.ffill()

    # BOS/CHoCH 偵測
    trend = 0  # 1=多, -1=空
    for i in range(swing_len * 2, len(df)):
        if close.iloc[i] > last_sh.iloc[i - 1] and pd.notna(last_sh.iloc[i - 1]):
            if trend <= 0:
                signal.iloc[i] = 1  # CHoCH 多
            trend = 1
        elif close.iloc[i] < last_sl.iloc[i - 1] and pd.notna(last_sl.iloc[i - 1]):
            if trend >= 0:
                signal.iloc[i] = -1  # CHoCH 空
            trend = -1
        else:
            signal.iloc[i] = signal.iloc[i - 1] if i > 0 else 0

    return signal


# ═══════════════════════════════════════════════════════════════
# 2. Squeeze Momentum — LazyBear
#    BB 被 KC 包住 = 壓縮，釋放後跟隨線性回歸動量
# ═══════════════════════════════════════════════════════════════

def squeeze_momentum(df: pd.DataFrame, bb_len: int = 20, bb_mult: float = 2.0,
                      kc_len: int = 20, kc_mult: float = 1.5) -> pd.Series:
    """
    Squeeze Momentum Indicator (LazyBear)
    - Squeeze = BB 完全在 KC 裡面
    - 釋放後跟隨動量方向
    """
    close = df["Close"]
    high, low = df["High"], df["Low"]

    # Bollinger Bands
    bb_basis = _sma(close, bb_len)
    bb_dev = _stdev(close, bb_len) * bb_mult
    upper_bb = bb_basis + bb_dev
    lower_bb = bb_basis - bb_dev

    # Keltner Channels
    kc_basis = _sma(close, kc_len)
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    range_ma = _sma(tr, kc_len)
    upper_kc = kc_basis + range_ma * kc_mult
    lower_kc = kc_basis - range_ma * kc_mult

    # Squeeze 狀態
    sqz_on = (lower_bb > lower_kc) & (upper_bb < upper_kc)

    # 動量 (線性回歸)
    avg_hl = (_highest(high, kc_len) + _lowest(low, kc_len)) / 2
    avg_all = (avg_hl + _sma(close, kc_len)) / 2
    mom_src = close - avg_all
    mom = _linreg(mom_src, kc_len)

    # 信號：壓縮釋放後跟動量
    signal = pd.Series(0, index=df.index)
    was_squeeze = False

    for i in range(1, len(df)):
        if pd.isna(mom.iloc[i]):
            continue
        if sqz_on.iloc[i]:
            was_squeeze = True
        if was_squeeze and not sqz_on.iloc[i]:
            # 壓縮釋放
            was_squeeze = False
        # 跟隨動量方向
        if mom.iloc[i] > 0:
            signal.iloc[i] = 1
        elif mom.iloc[i] < 0:
            signal.iloc[i] = -1
        else:
            signal.iloc[i] = 0

    return signal


# ═══════════════════════════════════════════════════════════════
# 3. MACD Custom Histogram — ChrisMoody
#    MACD 柱狀圖四色分級
# ═══════════════════════════════════════════════════════════════

def macd_custom(df: pd.DataFrame, fast: int = 12, slow: int = 26, sig: int = 9) -> pd.Series:
    """
    MACD with custom histogram coloring (ChrisMoody)
    - 柱狀圖 > 0 且遞增 → 強多
    - 柱狀圖 > 0 且遞減 → 弱多
    - 柱狀圖 < 0 且遞減 → 強空
    - 柱狀圖 < 0 且遞增 → 弱空
    Signal: MACD line cross signal line
    """
    close = df["Close"]
    macd_line = _ema(close, fast) - _ema(close, slow)
    signal_line = _ema(macd_line, sig)
    hist = macd_line - signal_line

    signal = pd.Series(0, index=df.index)
    for i in range(1, len(df)):
        if pd.isna(hist.iloc[i]):
            continue
        # MACD 線穿越信號線
        if macd_line.iloc[i] > signal_line.iloc[i] and macd_line.iloc[i - 1] <= signal_line.iloc[i - 1]:
            signal.iloc[i] = 1
        elif macd_line.iloc[i] < signal_line.iloc[i] and macd_line.iloc[i - 1] >= signal_line.iloc[i - 1]:
            signal.iloc[i] = -1
        else:
            signal.iloc[i] = signal.iloc[i - 1]

    return signal


# ═══════════════════════════════════════════════════════════════
# 4. SuperTrend — KivancOzbilgic
#    ATR 動態追蹤止損，翻轉即反手
# ═══════════════════════════════════════════════════════════════

def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.Series:
    """
    SuperTrend (KivancOzbilgic)
    - 基於 ATR 的動態追蹤止損
    - 價格在 SuperTrend 之上 → 多
    - 價格在 SuperTrend 之下 → 空
    """
    high, low, close = df["High"], df["Low"], df["Close"]
    atr = _atr(df, period)

    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    supertrend_arr = np.zeros(len(df))
    direction = np.ones(len(df))  # 1=多, -1=空

    for i in range(1, len(df)):
        if pd.isna(atr.iloc[i]):
            supertrend_arr[i] = close.iloc[i]
            continue

        # 調整 bands
        if lower_band.iloc[i] < lower_band.iloc[i - 1] and close.iloc[i - 1] > lower_band.iloc[i - 1]:
            lower_band.iloc[i] = lower_band.iloc[i - 1]
        if upper_band.iloc[i] > upper_band.iloc[i - 1] and close.iloc[i - 1] < upper_band.iloc[i - 1]:
            upper_band.iloc[i] = upper_band.iloc[i - 1]

        if direction[i - 1] == 1:
            if close.iloc[i] < lower_band.iloc[i]:
                direction[i] = -1
                supertrend_arr[i] = upper_band.iloc[i]
            else:
                direction[i] = 1
                supertrend_arr[i] = lower_band.iloc[i]
        else:
            if close.iloc[i] > upper_band.iloc[i]:
                direction[i] = 1
                supertrend_arr[i] = lower_band.iloc[i]
            else:
                direction[i] = -1
                supertrend_arr[i] = upper_band.iloc[i]

    signal = pd.Series(direction, index=df.index).astype(int)
    return signal


# ═══════════════════════════════════════════════════════════════
# 5. CM Williams Vix Fix — ChrisMoody
#    偵測恐慌底部，做多信號
# ═══════════════════════════════════════════════════════════════

def williams_vix_fix(df: pd.DataFrame, pd_len: int = 22, bb_len: int = 20,
                      bb_mult: float = 2.0, pct_len: int = 50, pct_hi: float = 0.85) -> pd.Series:
    """
    CM Williams Vix Fix (ChrisMoody)
    - WVF 衡量市場恐慌程度
    - WVF 飆高 = 市場底部 → 做多信號
    """
    close, low = df["Close"], df["Low"]

    # Williams Vix Fix
    highest_close = _highest(close, pd_len)
    wvf = (highest_close - low) / highest_close * 100

    # BB on WVF
    wvf_sma = _sma(wvf, bb_len)
    wvf_std = _stdev(wvf, bb_len)
    upper_bb = wvf_sma + bb_mult * wvf_std

    # Percentile threshold
    wvf_pct_high = wvf.rolling(pct_len).quantile(pct_hi)

    # 信號：WVF 突破上軌或百分位門檻
    alert = (wvf >= upper_bb) | (wvf >= wvf_pct_high)

    signal = pd.Series(0, index=df.index)
    position = 0
    for i in range(1, len(df)):
        if pd.isna(upper_bb.iloc[i]):
            continue
        if alert.iloc[i] and position <= 0:
            position = 1
            signal.iloc[i] = 1
        elif not alert.iloc[i] and position == 1:
            # 恐慌結束後，持有一段時間再考慮退出
            if wvf.iloc[i] < wvf_sma.iloc[i]:
                position = 0
                signal.iloc[i] = 0
            else:
                signal.iloc[i] = 1
        else:
            signal.iloc[i] = position

    return signal


# ═══════════════════════════════════════════════════════════════
# 6. WaveTrend Oscillator — LazyBear
#    類 CCI 波動趨勢，超買/超賣交叉
# ═══════════════════════════════════════════════════════════════

def wavetrend(df: pd.DataFrame, ch_len: int = 10, avg_len: int = 21,
               ob: float = 53, os_level: float = -53) -> pd.Series:
    """
    WaveTrend Oscillator (LazyBear)
    - WT1/WT2 交叉 + 超買超賣
    """
    tp = (df["High"] + df["Low"] + df["Close"]) / 3

    esa = _ema(tp, ch_len)
    d = _ema((tp - esa).abs(), ch_len)
    ci = (tp - esa) / (0.015 * d + 1e-10)

    wt1 = _ema(ci, avg_len)
    wt2 = _sma(wt1, 4)

    signal = pd.Series(0, index=df.index)
    for i in range(1, len(df)):
        if pd.isna(wt2.iloc[i]):
            continue
        # WT1 上穿 WT2 且在超賣區 → 多
        if wt1.iloc[i] > wt2.iloc[i] and wt1.iloc[i - 1] <= wt2.iloc[i - 1]:
            signal.iloc[i] = 1
        # WT1 下穿 WT2 且在超買區 → 空
        elif wt1.iloc[i] < wt2.iloc[i] and wt1.iloc[i - 1] >= wt2.iloc[i - 1]:
            signal.iloc[i] = -1
        else:
            signal.iloc[i] = signal.iloc[i - 1]

    return signal


# ═══════════════════════════════════════════════════════════════
# 7. Support & Resistance — LuxAlgo
#    動態支撐/阻力位反彈交易
# ═══════════════════════════════════════════════════════════════

def support_resistance(df: pd.DataFrame, lookback: int = 20, threshold: float = 0.005) -> pd.Series:
    """
    Support & Resistance (LuxAlgo-style)
    - 用 swing 高低點建立動態支撐/阻力
    - 價格彈離支撐 → 多，跌破 → 空
    """
    high, low, close = df["High"], df["Low"], df["Close"]

    support = _lowest(low, lookback)
    resistance = _highest(high, lookback)

    signal = pd.Series(0, index=df.index)
    for i in range(lookback, len(df)):
        mid = (support.iloc[i] + resistance.iloc[i]) / 2
        range_pct = (resistance.iloc[i] - support.iloc[i]) / mid

        if range_pct < threshold:
            signal.iloc[i] = signal.iloc[i - 1]
            continue

        # 價格接近支撐且反彈
        if close.iloc[i] > support.iloc[i] and close.iloc[i - 1] <= support.iloc[i - 1] * 1.002:
            signal.iloc[i] = 1
        # 價格接近阻力且反轉
        elif close.iloc[i] < resistance.iloc[i] and close.iloc[i - 1] >= resistance.iloc[i - 1] * 0.998:
            signal.iloc[i] = -1
        else:
            # 趨勢判斷
            if close.iloc[i] > mid:
                signal.iloc[i] = 1
            else:
                signal.iloc[i] = -1

    return signal


# ═══════════════════════════════════════════════════════════════
# 8. Market Structure — EmreKb
#    Swing H/L 偵測結構突破與趨勢翻轉
# ═══════════════════════════════════════════════════════════════

def market_structure(df: pd.DataFrame, swing_len: int = 5) -> pd.Series:
    """
    Market Structure (EmreKb)
    - HH/HL = 多頭結構
    - LH/LL = 空頭結構
    - 結構翻轉 = 信號
    """
    high, low, close = df["High"], df["Low"], df["Close"]

    # 偵測 swing points
    swing_highs = []
    swing_lows = []

    signal = pd.Series(0, index=df.index)
    trend = 0

    for i in range(swing_len, len(df) - swing_len):
        # Swing High
        if high.iloc[i] == high.iloc[i - swing_len: i + swing_len + 1].max():
            swing_highs.append((i, high.iloc[i]))
        # Swing Low
        if low.iloc[i] == low.iloc[i - swing_len: i + swing_len + 1].min():
            swing_lows.append((i, low.iloc[i]))

    # 分析結構
    for i in range(len(df)):
        recent_sh = [s for s in swing_highs if s[0] <= i]
        recent_sl = [s for s in swing_lows if s[0] <= i]

        if len(recent_sh) >= 2 and len(recent_sl) >= 2:
            last_sh = recent_sh[-1][1]
            prev_sh = recent_sh[-2][1]
            last_sl = recent_sl[-1][1]
            prev_sl = recent_sl[-2][1]

            # Higher High + Higher Low = 多頭
            if last_sh > prev_sh and last_sl > prev_sl:
                if trend != 1:
                    trend = 1
                signal.iloc[i] = 1
            # Lower High + Lower Low = 空頭
            elif last_sh < prev_sh and last_sl < prev_sl:
                if trend != -1:
                    trend = -1
                signal.iloc[i] = -1
            else:
                signal.iloc[i] = trend

    return signal


# ═══════════════════════════════════════════════════════════════
# 9. ADX and DI — BeikabuOyaji
# ═══════════════════════════════════════════════════════════════

def adx_di(df: pd.DataFrame, period: int = 14, adx_threshold: float = 25) -> pd.Series:
    """
    ADX and DI (BeikabuOyaji)
    - ADX > 25 確認趨勢
    - +DI > -DI → 多，+DI < -DI → 空
    """
    high, low, close = df["High"], df["Low"], df["Close"]

    # +DM / -DM
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    atr = _atr(df, period)

    plus_di = 100 * _ema(plus_dm, period) / (atr + 1e-10)
    minus_di = 100 * _ema(minus_dm, period) / (atr + 1e-10)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = _ema(dx, period)

    signal = pd.Series(0, index=df.index)
    for i in range(period * 2, len(df)):
        if pd.isna(adx.iloc[i]):
            continue
        if adx.iloc[i] > adx_threshold:
            if plus_di.iloc[i] > minus_di.iloc[i]:
                signal.iloc[i] = 1
            else:
                signal.iloc[i] = -1
        else:
            signal.iloc[i] = 0  # 無趨勢，空手

    return signal


# ═══════════════════════════════════════════════════════════════
# 10. Bollinger + RSI — ChartArt
# ═══════════════════════════════════════════════════════════════

def bollinger_rsi(df: pd.DataFrame, bb_len: int = 20, bb_mult: float = 2.0,
                   rsi_len: int = 14, rsi_ob: float = 70, rsi_os: float = 30) -> pd.Series:
    """
    Bollinger Bands + RSI (ChartArt)
    - 價格 < 下軌 + RSI < 30 → 多
    - 價格 > 上軌 + RSI > 70 → 空
    """
    close = df["Close"]
    basis = _sma(close, bb_len)
    dev = _stdev(close, bb_len) * bb_mult
    upper = basis + dev
    lower = basis - dev
    rsi_val = _rsi(close, rsi_len)

    signal = pd.Series(0, index=df.index)
    for i in range(max(bb_len, rsi_len), len(df)):
        if pd.isna(rsi_val.iloc[i]) or pd.isna(lower.iloc[i]):
            continue
        if close.iloc[i] <= lower.iloc[i] and rsi_val.iloc[i] < rsi_os:
            signal.iloc[i] = 1
        elif close.iloc[i] >= upper.iloc[i] and rsi_val.iloc[i] > rsi_ob:
            signal.iloc[i] = -1
        else:
            signal.iloc[i] = signal.iloc[i - 1]

    return signal


# ═══════════════════════════════════════════════════════════════
# 11. UT Bot Alerts — QuantNomad
#     ATR 動態追蹤線
# ═══════════════════════════════════════════════════════════════

def ut_bot(df: pd.DataFrame, key_value: float = 1.0, atr_period: int = 10) -> pd.Series:
    """
    UT Bot Alerts (QuantNomad)
    - ATR trailing stop
    - 價格穿越 trailing line 即進場
    """
    close = df["Close"]
    atr = _atr(df, atr_period)
    n_loss = key_value * atr

    trailing_stop = pd.Series(0.0, index=df.index)
    signal = pd.Series(0, index=df.index)

    for i in range(1, len(df)):
        if pd.isna(n_loss.iloc[i]):
            trailing_stop.iloc[i] = close.iloc[i]
            continue

        if close.iloc[i] > trailing_stop.iloc[i - 1]:
            trailing_stop.iloc[i] = max(trailing_stop.iloc[i - 1], close.iloc[i] - n_loss.iloc[i])
        else:
            trailing_stop.iloc[i] = min(trailing_stop.iloc[i - 1], close.iloc[i] + n_loss.iloc[i])

        # 信號
        if close.iloc[i] > trailing_stop.iloc[i] and close.iloc[i - 1] <= trailing_stop.iloc[i - 1]:
            signal.iloc[i] = 1
        elif close.iloc[i] < trailing_stop.iloc[i] and close.iloc[i - 1] >= trailing_stop.iloc[i - 1]:
            signal.iloc[i] = -1
        else:
            signal.iloc[i] = signal.iloc[i - 1]

    return signal


# ═══════════════════════════════════════════════════════════════
# 12. Trendlines with Breaks — LuxAlgo
#     自動趨勢線 + 突破偵測
# ═══════════════════════════════════════════════════════════════

def trendline_breaks(df: pd.DataFrame, lookback: int = 14) -> pd.Series:
    """
    Trendlines with Breaks (LuxAlgo-style)
    - 連接 swing 高低點形成趨勢線
    - 突破趨勢線 → 進場
    """
    high, low, close = df["High"], df["Low"], df["Close"]

    # 用近期高低點形成簡化趨勢線
    upper_trend = _highest(high, lookback)
    lower_trend = _lowest(low, lookback)

    # 趨勢線斜率
    upper_slope = upper_trend.diff(lookback) / lookback
    lower_slope = lower_trend.diff(lookback) / lookback

    signal = pd.Series(0, index=df.index)

    for i in range(lookback * 2, len(df)):
        # 價格突破下降趨勢線 (阻力)
        if close.iloc[i] > upper_trend.iloc[i - 1] and close.iloc[i - 1] <= upper_trend.iloc[i - 2]:
            signal.iloc[i] = 1
        # 價格跌破上升趨勢線 (支撐)
        elif close.iloc[i] < lower_trend.iloc[i - 1] and close.iloc[i - 1] >= lower_trend.iloc[i - 2]:
            signal.iloc[i] = -1
        else:
            signal.iloc[i] = signal.iloc[i - 1]

    return signal


# ═══════════════════════════════════════════════════════════════
# 13. ICT Killzones (簡化版) — tradeforopp
#     日線簡化：Order Block 反彈 + 動量確認
# ═══════════════════════════════════════════════════════════════

def ict_killzones(df: pd.DataFrame, ob_lookback: int = 10, atr_mult: float = 1.5) -> pd.Series:
    """
    ICT Killzones (簡化版)
    - 日線無法分時段，改用 Order Block + 假突破反轉
    - 價格掃過近期高低後反轉 = 流動性掃取
    """
    high, low, close = df["High"], df["Low"], df["Close"]
    atr = _atr(df, 14)

    recent_high = _highest(high, ob_lookback).shift(1)
    recent_low = _lowest(low, ob_lookback).shift(1)

    signal = pd.Series(0, index=df.index)

    for i in range(ob_lookback + 1, len(df)):
        if pd.isna(atr.iloc[i]) or pd.isna(recent_high.iloc[i]):
            continue

        # 流動性掃取後反轉（日線版）
        # Bullish: 先破前低 (sweep) 再收在上面
        if low.iloc[i] < recent_low.iloc[i] and close.iloc[i] > recent_low.iloc[i]:
            signal.iloc[i] = 1
        # Bearish: 先破前高 (sweep) 再收在下面
        elif high.iloc[i] > recent_high.iloc[i] and close.iloc[i] < recent_high.iloc[i]:
            signal.iloc[i] = -1
        else:
            signal.iloc[i] = signal.iloc[i - 1]

    return signal


# ═══════════════════════════════════════════════════════════════
# 14. Swing High Low — Patternsmart
# ═══════════════════════════════════════════════════════════════

def swing_high_low(df: pd.DataFrame, swing_len: int = 5) -> pd.Series:
    """
    Swing High Low (Patternsmart)
    - 突破前一個 swing high → 多
    - 跌破前一個 swing low → 空
    """
    high, low, close = df["High"], df["Low"], df["Close"]

    last_swing_high = np.nan
    last_swing_low = np.nan
    signal = pd.Series(0, index=df.index)

    for i in range(swing_len, len(df) - swing_len):
        # 偵測 swing 點
        window_h = high.iloc[i - swing_len: i + swing_len + 1]
        window_l = low.iloc[i - swing_len: i + swing_len + 1]

        if high.iloc[i] == window_h.max():
            last_swing_high = high.iloc[i]
        if low.iloc[i] == window_l.min():
            last_swing_low = low.iloc[i]

        # 信號
        if not np.isnan(last_swing_high) and close.iloc[i] > last_swing_high:
            signal.iloc[i] = 1
        elif not np.isnan(last_swing_low) and close.iloc[i] < last_swing_low:
            signal.iloc[i] = -1
        elif i > 0:
            signal.iloc[i] = signal.iloc[i - 1]

    return signal


# ═══════════════════════════════════════════════════════════════
# 15. Keltner Channel + EMA
# ═══════════════════════════════════════════════════════════════

def keltner_channel(df: pd.DataFrame, ema_len: int = 20, atr_len: int = 14,
                     atr_mult: float = 2.0) -> pd.Series:
    """
    Keltner Channel + EMA
    - 突破上軌 → 多
    - 跌破下軌 → 空
    - EMA 方向過濾
    """
    close = df["Close"]
    ema_val = _ema(close, ema_len)
    atr = _atr(df, atr_len)

    upper = ema_val + atr_mult * atr
    lower = ema_val - atr_mult * atr

    signal = pd.Series(0, index=df.index)
    for i in range(max(ema_len, atr_len), len(df)):
        if pd.isna(upper.iloc[i]):
            continue
        if close.iloc[i] > upper.iloc[i]:
            signal.iloc[i] = 1
        elif close.iloc[i] < lower.iloc[i]:
            signal.iloc[i] = -1
        else:
            signal.iloc[i] = signal.iloc[i - 1]

    return signal


# ═══════════════════════════════════════════════════════════════
# 16. Hybrid: Squeeze Momentum + CM Williams Vix Fix
#     雙引擎：Squeeze 吃波段 + WVF 撈底
# ═══════════════════════════════════════════════════════════════

def hybrid_squeeze_vix_fix(df: pd.DataFrame) -> pd.Series:
    """
    Hybrid Strategy
    - 同時計算 Squeeze Momentum & Williams Vix Fix
    - 任一策略做多 → 做多 (OR Logic)
    - 互斥保護：若多空信號同時出現 → 空手
    """
    # 1. 計算 Squeeze Momentum 信號
    sig_sqz = squeeze_momentum(df)

    # 2. 計算 CM Williams Vix Fix 信號
    sig_wvf = williams_vix_fix(df)

    # 3. 合併信號
    signal = pd.Series(0, index=df.index)

    for i in range(len(df)):
        s1 = sig_sqz.iloc[i]
        s2 = sig_wvf.iloc[i]

        # 互斥保護：多空衝突
        if (s1 == 1 and s2 == -1) or (s1 == -1 and s2 == 1):
            signal.iloc[i] = 0
        # 任一做多
        elif s1 == 1 or s2 == 1:
            signal.iloc[i] = 1
        # 任一做空
        elif s1 == -1 or s2 == -1:
            signal.iloc[i] = -1
        else:
            signal.iloc[i] = 0

    return signal


# ═══════════════════════════════════════════════════════════════
# 策略註冊表
# ═══════════════════════════════════════════════════════════════

STRATEGIES = {
    "Smart Money Concepts (LuxAlgo)": smart_money_concepts,
    "Squeeze Momentum (LazyBear)": squeeze_momentum,
    "MACD Custom (ChrisMoody)": macd_custom,
    "SuperTrend (KivancOzbilgic)": supertrend,
    "CM Williams Vix Fix (ChrisMoody)": williams_vix_fix,
    "WaveTrend Oscillator (LazyBear)": wavetrend,
    "Support & Resistance (LuxAlgo)": support_resistance,
    "Market Structure (EmreKb)": market_structure,
    "ADX and DI (BeikabuOyaji)": adx_di,
    "Bollinger + RSI (ChartArt)": bollinger_rsi,
    "UT Bot Alerts (QuantNomad)": ut_bot,
    "Trendlines with Breaks (LuxAlgo)": trendline_breaks,
    "ICT Killzones (tradeforopp)": ict_killzones,
    "Swing High Low (Patternsmart)": swing_high_low,
    "Keltner Channel": keltner_channel,
    "🚀 Hybrid: Squeeze + Vix Fix": hybrid_squeeze_vix_fix,
}
