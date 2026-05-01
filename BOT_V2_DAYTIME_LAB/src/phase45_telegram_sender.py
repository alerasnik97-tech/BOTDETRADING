import os
import requests
import argparse
import json
import re

# NO guardar tokens aqui. Usar variables de entorno de Windows.
# TELEGRAM_BOT_TOKEN o BOT_TELEGRAM_TOKEN
# TELEGRAM_CHAT_ID o BOT_TELEGRAM_CHAT_ID

class TelegramSender:
    def __init__(self, token=None, chat_id=None):
        self.token = token or os.environ.get("TELEGRAM_BOT_TOKEN") or os.environ.get("BOT_TELEGRAM_TOKEN")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID") or os.environ.get("BOT_TELEGRAM_CHAT_ID")
        self.enabled = bool(self.token and self.chat_id)

    def _mask_token(self, text):
        if not text: return ""
        masked = str(text)
        if self.token:
            masked = masked.replace(self.token, "***MASKED***")
        return re.sub(r"bot[0-9]+:[a-zA-Z0-9_-]+", "bot***MASKED***", masked)

    def _get_bot_id(self):
        if not self.token: return None
        if ":" in self.token:
            return self.token.split(":")[0]
        return "Invalid"

    def get_diagnostic(self):
        token_present = bool(self.token)
        chat_id_present = bool(self.chat_id)

        diag = {
            "TOKEN_PRESENT": token_present,
            "TOKEN_LEN": len(self.token) if token_present else 0,
            "CHAT_ID_PRESENT": chat_id_present,
            "CHAT_ID_NUMERIC": str(self.chat_id).replace("-", "").isdigit() if chat_id_present else False,
            "CHAT_ID_LEN": len(str(self.chat_id)) if chat_id_present else 0
        }
        return diag

    def send_message(self, text, dry_run=False, parse_mode=None):
        if not text or not str(text).strip():
            return {"status": "ERROR", "error": "Empty message"}

        if not self.enabled:
            missing = []
            if not self.token: missing.append("TELEGRAM_BOT_TOKEN")
            if not self.chat_id: missing.append("TELEGRAM_CHAT_ID")
            return {"status": "ERROR", "error": f"Missing environment variables: {', '.join(missing)}"}

        if dry_run:
            print(f"[DRY-RUN] Telegram: {text}")
            return {"status": "DRY_RUN_OK"}

        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode

        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            body = response.json()
            result = body.get("result") if isinstance(body, dict) else {}
            return {
                "status": "SENT",
                "ok": bool(body.get("ok")) if isinstance(body, dict) else True,
                "message_id": result.get("message_id") if isinstance(result, dict) else None,
            }
        except requests.exceptions.HTTPError as e:
            # Mask token in URL if it appears in error
            err_msg = self._mask_token(str(e))
            return {"status": "ERROR", "error": f"HTTP Error: {err_msg}"}
        except Exception as e:
            return {"status": "ERROR", "error": self._mask_token(str(e))}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--send-test", action="store_true")
    parser.add_argument("--diag", action="store_true")
    args = parser.parse_args()

    sender = TelegramSender()

    if args.diag:
        print(json.dumps(sender.get_diagnostic(), indent=2))
        return

    if args.send_test:
        msg = "MANIPULANTE Telegram sender OK"
        res = sender.send_message(msg, dry_run=args.dry_run)
        print(json.dumps(res, indent=2))
    else:
        if not sender.enabled:
            print("TELEGRAM_NOT_CONFIGURED")
            diag = sender.get_diagnostic()
            print(f"DEBUG: Token Present: {diag['TOKEN_PRESENT']}, Chat ID Present: {diag['CHAT_ID_PRESENT']}")
        else:
            print("TELEGRAM_READY")

if __name__ == "__main__":
    main()
