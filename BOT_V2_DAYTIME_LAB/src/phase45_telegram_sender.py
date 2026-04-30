import os
import requests
import argparse
import json

class TelegramSender:
    def __init__(self, token=None, chat_id=None):
        self.token = token or os.environ.get("MANIPULANTE_TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.environ.get("MANIPULANTE_TELEGRAM_CHAT_ID")
        self.enabled = bool(self.token and self.chat_id)

    def send_message(self, text, dry_run=False):
        if not self.enabled:
            return {"status": "TELEGRAM_NOT_CONFIGURED"}
        
        if dry_run:
            print(f"[DRY-RUN] Telegram: {text}")
            return {"status": "DRY_RUN_OK"}

        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML"
        }
        
        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            return {"status": "SENT", "response": response.json()}
        except Exception as e:
            return {"status": "ERROR", "error": str(e)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--send-test", action="store_true")
    args = parser.parse_args()

    sender = TelegramSender()
    
    if args.send_test:
        msg = "<b>MANIPULANTE ALERT TEST</b>\nSistema de alertas configurado correctamente.\nNo se envio ninguna orden.\nNo se modifico la estrategia."
        res = sender.send_message(msg, dry_run=args.dry_run)
        print(json.dumps(res, indent=2))
    else:
        if not sender.enabled:
            print("TELEGRAM_NOT_CONFIGURED")
        else:
            print("TELEGRAM_READY")

if __name__ == "__main__":
    main()
