import os
import pandas as pd

def aggregate_5m_to_15m():
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "mxf_5min.parquet")
    out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "mxf_15min.parquet")
    
    if not os.path.exists(data_path):
        print("❌ 找不到 5 分 K 資料檔案")
        return
        
    print("⏳ 載入 5 分 K 資料...")
    df = pd.read_parquet(data_path)
    
    # 確保 index 是 datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
        
    print(f"📊 原始資料: {len(df)} 根 K 棒")
    
    # 使用 pandas 的 resample 聚合
    # 注意台指期的開盤時間，通常使用閉區間，這裡使用預設即可 (label='right', closed='right')
    # 也可以簡單地用 '15T' 聚合
    agg_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }
    
    print("⏳ 正在聚合為 15 分 K...")
    df_15m = df.resample('15min').agg(agg_dict).dropna()
    
    print(f"📊 聚合後資料: {len(df_15m)} 根 K 棒")
    
    df_15m.to_parquet(out_path)
    print(f"✅ 成功儲存為 {out_path}")

if __name__ == "__main__":
    aggregate_5m_to_15m()
