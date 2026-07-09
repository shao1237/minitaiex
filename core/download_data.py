"""
下載台灣加權指數 (^TWII) 歷史日線數據
- 下載後存為本地 CSV，後續回測從本地讀取
"""
import os
import pandas as pd
import yfinance as yf

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
CSV_PATH = os.path.join(DATA_DIR, "taiex_daily.csv")


def download_taiex(period="max", force=False):
    """下載 ^TWII 所有歷史日線資料，存到 data/taiex_daily.csv"""
    if os.path.exists(CSV_PATH) and not force:
        print(f"[INFO] 本地資料已存在: {CSV_PATH}，跳過下載。")
        return load_local_data()

    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"[INFO] 從 yfinance 下載 ^TWII ({period}) ...")
    ticker = yf.Ticker("^TWII")
    df = ticker.history(period=period)

    if df.empty:
        raise RuntimeError("下載失敗：yfinance 回傳空資料")

    # 只保留需要的欄位
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index.name = "Date"
    df.to_csv(CSV_PATH)
    print(f"[INFO] 已儲存 {len(df)} 筆資料到 {CSV_PATH}")
    return df


def load_local_data():
    """從本地 CSV 讀取資料"""
    if not os.path.exists(CSV_PATH):
        print(f"[WARN] 找不到本地資料: {CSV_PATH}，嘗試自動下載...")
        return download_taiex()
        
    df = pd.read_csv(CSV_PATH, index_col="Date", parse_dates=True)
    # print(f"[INFO] 已從本地載入 {len(df)} 筆資料")
    return df


if __name__ == "__main__":
    df = download_taiex()
    print(df.tail())
    print(f"\n資料範圍: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")
