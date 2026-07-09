from __future__ import annotations

"""
用 Shioaji API 下載小台指期 (MXF) 歷史 1 分鐘 K 線，並聚合為 5 分鐘 K 線
=============================================================================
- 使用 MXFR1（近月連續合約）
- 逐日下載 1 分 K → 聚合成 5 分 K → 存 Parquet
- 自動跳過週末 & 已下載日期

Shioaji kbars API 限制：
  - 期貨資料起始：2020-03-22
  - 回傳 1 分鐘 K 線（OHLCV）
  - 日期範圍越大，回傳越慢
"""
import os
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False

# 載入 .env（沿用 intraday-scanner 的 credentials）
env_paths = [
    os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"),
    r"D:\Antigravity\intraday-scanner\.env",
]
for p in env_paths:
    if os.path.exists(p):
        load_dotenv(p)
        break

try:
    import shioaji as sj
except ImportError:
    sj = None

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
KBAR_1M_DIR = os.path.join(DATA_DIR, "kbar_1m")
KBAR_5M_PATH = os.path.join(DATA_DIR, "mxf_5min.parquet")

os.makedirs(KBAR_1M_DIR, exist_ok=True)


def _login() -> sj.Shioaji:
    """登入 Shioaji（使用模擬帳號即可取歷史資料）"""
    if sj is None:
        raise ImportError("shioaji is required for downloading new kbar data")

    api_key = os.getenv("API_KEY", os.getenv("SHIOAJI_API_KEY", ""))
    secret_key = os.getenv("SECRET_KEY", os.getenv("SHIOAJI_SECRET_KEY", ""))

    if not api_key or not secret_key:
        print("❌ 找不到 API_KEY / SECRET_KEY，請確認 .env 設定")
        sys.exit(1)

    api = sj.Shioaji(simulation=True)
    api.login(
        api_key=api_key,
        secret_key=secret_key,
        contracts_timeout=10000,
    )
    print(f"✅ Shioaji 登入成功（模擬環境）")
    return api


def _get_trading_dates(start: str, end: str) -> list:
    """產生日期列表（排除週末）"""
    dates = pd.bdate_range(start=start, end=end)
    return [d.strftime("%Y-%m-%d") for d in dates]


def download_1m_kbars(api, start_date: str, end_date: str):
    """
    逐日下載 MXF 1 分鐘 K 線，存為 CSV。

    Shioaji kbars() 建議一次只抓 1~2 天，大範圍會超時。
    """
    contract = api.Contracts.Futures.MXF.MXFR1
    print(f"📦 合約: {contract.code} ({contract.name})")

    dates = _get_trading_dates(start_date, end_date)
    total = len(dates)
    downloaded = 0
    skipped = 0

    for idx, date_str in enumerate(dates):
        csv_path = os.path.join(KBAR_1M_DIR, f"{date_str}.csv")

        # 跳過已下載
        if os.path.exists(csv_path):
            skipped += 1
            continue

        try:
            kbars = api.kbars(
                contract=contract,
                start=date_str,
                end=date_str,
            )
            df = pd.DataFrame({**kbars})

            if df.empty:
                # 可能是國定假日
                continue

            df["ts"] = pd.to_datetime(df["ts"])
            df = df.rename(columns={"ts": "datetime"})

            # 取消時段過濾，保留全時段（包含夜盤 15:00 ~ 05:00）
            # df = df[(df["datetime"].dt.time >= pd.Timestamp("08:45").time()) &
            #          (df["datetime"].dt.time <= pd.Timestamp("13:45").time())]

            if df.empty:
                continue

            df.to_csv(csv_path, index=False)
            downloaded += 1

            if (idx + 1) % 10 == 0 or idx == total - 1:
                print(f"  [{idx+1}/{total}] 下載: {downloaded}, 跳過: {skipped}, 目前日期: {date_str}")

            # Rate limit: Shioaji 建議不要太快
            time.sleep(0.5)

        except Exception as e:
            print(f"  ⚠️ {date_str} 下載失敗: {e}")
            time.sleep(2)

    print(f"\n📥 下載完成！新增 {downloaded} 天，跳過 {skipped} 天")


def aggregate_to_5min() -> pd.DataFrame:
    """將所有 1 分 K CSV 聚合成 5 分鐘 K 線"""
    csv_files = sorted([
        f for f in os.listdir(KBAR_1M_DIR)
        if f.endswith(".csv")
    ])

    if not csv_files:
        print("❌ 沒有 1 分 K 資料，請先執行下載")
        return pd.DataFrame()

    print(f"🔄 聚合 {len(csv_files)} 天的 1 分 K → 5 分 K...")

    all_dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(os.path.join(KBAR_1M_DIR, f), parse_dates=["datetime"])
            all_dfs.append(df)
        except Exception as e:
            print(f"  ⚠️ 讀取 {f} 失敗: {e}")

    if not all_dfs:
        return pd.DataFrame()

    df_1m = pd.concat(all_dfs, ignore_index=True)
    df_1m = df_1m.sort_values("datetime").reset_index(drop=True)

    # 加入日期欄位（用於當沖分組）
    df_1m["date"] = df_1m["datetime"].dt.date

    # 以 5 分鐘為窗口聚合
    df_5m = df_1m.set_index("datetime").groupby("date").resample("5min").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }).dropna().reset_index()

    # 清理
    df_5m = df_5m.drop(columns=["date"])
    df_5m = df_5m.rename(columns={"datetime": "Date"})
    df_5m = df_5m.set_index("Date")

    # 存 Parquet（若目前環境沒有 pyarrow/fastparquet，仍回傳 DataFrame 供驗證使用）
    try:
        df_5m.to_parquet(KBAR_5M_PATH)
        print(f"✅ 5 分 K 已存至 {KBAR_5M_PATH}")
    except ImportError:
        print("⚠️ Parquet engine unavailable; using rebuilt DataFrame without saving parquet.")
    print(f"   範圍: {df_5m.index[0]} ~ {df_5m.index[-1]} | 共 {len(df_5m)} 根 K 棒")

    return df_5m


def load_5min_data() -> pd.DataFrame:
    """讀取已聚合的 5 分鐘 K 線"""
    if os.path.exists(KBAR_5M_PATH):
        try:
            return pd.read_parquet(KBAR_5M_PATH)
        except ImportError:
            print("⚠️ Parquet engine unavailable; rebuilding 5-minute bars from CSV cache...")
            return aggregate_to_5min()
    else:
        print("⚠️ 5 分 K 資料不存在，嘗試聚合...")
        return aggregate_to_5min()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="下載 MXF 歷史 1 分 K 並聚合為 5 分 K")
    parser.add_argument("--start", type=str, default="2026-01-02",
                        help="起始日期 (default: 2026-01-02)")
    parser.add_argument("--end", type=str, default=datetime.now().strftime("%Y-%m-%d"),
                        help="結束日期 (default: 今天)")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="只做聚合，不下載")
    args = parser.parse_args()

    if args.aggregate_only:
        aggregate_to_5min()
    else:
        api = _login()
        try:
            download_1m_kbars(api, args.start, args.end)
            aggregate_to_5min()
        finally:
            api.logout()
            print("🔌 已登出 Shioaji")
