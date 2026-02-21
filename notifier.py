"""
自動化交易系統 — Telegram 通知模組
====================================
負責推送交易信號、風控警告、每日損益摘要。
"""
import requests
import config


def send_message(text: str, parse_mode: str = "HTML") -> bool:
    """
    透過 Telegram Bot 發送訊息。

    Parameters
    ----------
    text : str  訊息內容（支援 HTML 格式）
    parse_mode : str  解析模式 ("HTML" 或 "Markdown")

    Returns
    -------
    bool : 是否發送成功
    """
    if not config.TELEGRAM_TOKEN or not config.TELEGRAM_CHAT_ID:
        print(f"[Notifier] Telegram 未設定，訊息僅印到 console：\n{text}")
        return False

    url = f"https://api.telegram.org/bot{config.TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": config.TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": parse_mode,
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code == 200:
            return True
        else:
            print(f"[Notifier] Telegram 發送失敗: {resp.status_code} {resp.text}")
            return False
    except Exception as e:
        print(f"[Notifier] Telegram 發送錯誤: {e}")
        return False


def notify_trade(action: str, direction: str, price: float, reason: str = ""):
    """
    推送交易通知。

    Parameters
    ----------
    action : str  "OPEN" 或 "CLOSE"
    direction : str  "LONG" 或 "SHORT"
    price : float  成交價
    reason : str  觸發原因
    """
    emoji = "🟢" if direction == "LONG" else "🔴"
    action_text = "開倉" if action == "OPEN" else "平倉"
    mode = "🧪 模擬盤" if config.PAPER_TRADING else "🔥 實盤"

    msg = (
        f"{emoji} <b>{action_text} {direction}</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"💰 價格: <code>{price:.0f}</code>\n"
        f"📊 策略: {config.PRIMARY_STRATEGY}\n"
        f"📝 原因: {reason}\n"
        f"🏷️ 模式: {mode}\n"
    )
    send_message(msg)


def notify_risk_alert(alert_type: str, details: str):
    """推送風控警告。"""
    msg = (
        f"🚨 <b>風控警報: {alert_type}</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"{details}\n"
        f"⚠️ 交易已暫停，請手動檢查。"
    )
    send_message(msg)


def notify_daily_summary(date: str, pnl: float, trades: int, win_trades: int,
                          equity: float):
    """推送每日收盤摘要。"""
    pnl_emoji = "📈" if pnl >= 0 else "📉"
    win_rate = (win_trades / trades * 100) if trades > 0 else 0
    mode = "🧪 模擬盤" if config.PAPER_TRADING else "🔥 實盤"

    msg = (
        f"📊 <b>每日交易摘要 — {date}</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"{pnl_emoji} 當日損益: <code>{pnl:+,.0f} TWD</code>\n"
        f"📋 交易次數: {trades} 筆\n"
        f"🎯 勝率: {win_rate:.0f}%\n"
        f"💼 帳戶權益: <code>{equity:,.0f} TWD</code>\n"
        f"🏷️ 模式: {mode}"
    )
    send_message(msg)


def notify_system(event: str, details: str = ""):
    """推送系統事件（啟動、關閉、錯誤）。"""
    msg = f"⚙️ <b>系統通知: {event}</b>"
    if details:
        msg += f"\n{details}"
    send_message(msg)
