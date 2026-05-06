import os
import requests
import json

# CREDENTIALS FROM ENVIRONMENT
TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

def send_ping():
    if not TOKEN or not CHAT_ID:
        print(f"ERROR: Missing Telegram credentials. Token: {bool(TOKEN)}, ChatID: {bool(CHAT_ID)}")
        return

    msg = "🔴 OVERRIDE CHECK: Sistema de alertas operativo. Telemetría de diagnóstico en curso."
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": msg
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        print("SUCCESS: Telegram message sent successfully (Status 200).")
    except Exception as e:
        print(f"FAILURE: Telegram message failed.")

if __name__ == "__main__":
    send_ping()
