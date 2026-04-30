import os
import time
import argparse
import json
from phase45_alert_engine import AlertEngine
from phase45_telegram_sender import TelegramSender
from phase45_email_sender import EmailSender
from phase45_alert_state import AlertState

def run_check(dry_run=False):
    root_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
    state_file = os.path.join(root_path, "MANIPULANTE", "16_OBSERVABILITY", "alerts", "alert_state.json")
    config_file = os.path.join(root_path, "MANIPULANTE", "16_OBSERVABILITY", "alerts", "alerts_config.local.json")
    if not os.path.exists(config_file):
         config_file = os.path.join(root_path, "MANIPULANTE", "16_OBSERVABILITY", "alerts", "alerts_config.example.json")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)

    engine = AlertEngine(root_path)
    telegram = TelegramSender()
    email = EmailSender()
    state = AlertState(state_file)

    alerts = engine.detect_alerts()
    
    sent_count = 0
    for alert in alerts:
        if state.should_send(alert["dedup_key"], alert["severity"], config.get("alert_cooldown_minutes", 10)):
            msg = f"<b>{alert['severity']}: {alert['event_type']}</b>\n\n{alert['title']}\n\n{alert['message']}\n\nAccion: {alert['recommended_action']}"
            
            # Telegram
            if config.get("telegram_enabled") or telegram.enabled:
                res_tg = telegram.send_message(msg, dry_run=dry_run)
                if res_tg["status"] == "SENT":
                    sent_count += 1
            
            # Email (Critical only if configured that way, or always if enabled)
            if config.get("email_enabled") or email.enabled:
                if not config.get("send_critical_only") or alert["severity"] == "CRITICAL":
                    email.send_email(f"MANIPULANTE ALERT: {alert['event_type']}", msg, dry_run=dry_run)

    print(f"Check completo. Alertas detectadas: {len(alerts)}. Alertas enviadas: {sent_count}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.loop:
        print(f"Iniciando loop de alertas cada {args.interval_seconds} segundos...")
        while True:
            run_check(dry_run=args.dry_run)
            time.sleep(args.interval_seconds)
    else:
        run_check(dry_run=args.dry_run)

if __name__ == "__main__":
    main()
