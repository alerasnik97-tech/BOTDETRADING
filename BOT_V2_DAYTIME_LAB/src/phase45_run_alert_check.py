import os
import time
import argparse
import json
import sys
import subprocess
from datetime import datetime
from phase45_alert_engine import AlertEngine
from phase45_telegram_sender import TelegramSender
from phase45_email_sender import EmailSender
from phase45_alert_state import AlertState

# PID/LOCK Management
ROOT_PATH = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
RUNTIME_DIR = os.path.join(ROOT_PATH, "MANIPULANTE", "16_OBSERVABILITY", "alerts", "runtime")
PID_FILE = os.path.join(RUNTIME_DIR, "alerts_loop.pid")
LOCK_FILE = os.path.join(RUNTIME_DIR, "alerts_loop.lock")
HEARTBEAT_FILE = os.path.join(RUNTIME_DIR, "alerts_loop.last_heartbeat.json")

def _read_pid_file():
    try:
        with open(PID_FILE, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        return int(raw) if raw else None
    except Exception:
        return None


def _process_command_line(pid):
    if not pid or pid <= 0:
        return None
    try:
        ps = (
            "$p=Get-CimInstance Win32_Process -Filter \"ProcessId = "
            + str(int(pid))
            + "\"; if ($p) { $p.CommandLine }"
        )
        res = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        command_line = (res.stdout or "").strip()
        return command_line or None
    except Exception:
        return None


def _is_alert_loop_command(command_line):
    if not command_line:
        return False
    cmd = command_line.lower()
    return (
        "bot de trading ultimo" in cmd
        and "phase45_run_alert_check.py" in cmd
        and "--loop" in cmd
    )


def _pid_is_running(pid):
    return _is_alert_loop_command(_process_command_line(pid))


def _read_json_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def get_alerts_status():
    pid = _read_pid_file()
    lock = _read_json_file(LOCK_FILE)
    heartbeat = _read_json_file(HEARTBEAT_FILE)
    if pid is None and isinstance(lock, dict):
        try:
            pid = int(lock.get("pid"))
        except Exception:
            pid = None

    command_line = _process_command_line(pid) if pid else None
    heartbeat_age_seconds = None
    if isinstance(heartbeat, dict):
        try:
            ts = float(heartbeat.get("timestamp"))
            heartbeat_age_seconds = max(0, int(time.time() - ts))
        except Exception:
            heartbeat_age_seconds = None

    env_ready = bool(
        (os.environ.get("TELEGRAM_BOT_TOKEN") or os.environ.get("BOT_TELEGRAM_TOKEN"))
        and (os.environ.get("TELEGRAM_CHAT_ID") or os.environ.get("BOT_TELEGRAM_CHAT_ID"))
    )

    if pid and command_line and _is_alert_loop_command(command_line):
        state = "ALERTS_RUNNING"
    elif pid and command_line:
        state = "ALERTS_PID_OWNER_MISMATCH"
    elif pid or os.path.exists(LOCK_FILE):
        state = "ALERTS_STALE_LOCK"
    else:
        state = "ALERTS_STOPPED"

    return {
        "state": state,
        "pid": pid,
        "env_ready": env_ready,
        "heartbeat_age_seconds": heartbeat_age_seconds,
        "heartbeat_present": heartbeat is not None,
    }


def cleanup_stale_runtime_files():
    for path in (PID_FILE, LOCK_FILE):
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass


def stop_alerts_loop():
    status = get_alerts_status()
    state = status["state"]
    pid = status["pid"]
    if state == "ALERTS_STOPPED":
        print("ALERTS_LOOP_NOT_RUNNING")
        return 0
    if state == "ALERTS_STALE_LOCK":
        cleanup_stale_runtime_files()
        print("ALERTS_STALE_LOCK_CLEANED")
        return 0
    if state == "ALERTS_PID_OWNER_MISMATCH":
        print("ALERTS_PID_OWNER_MISMATCH")
        return 3
    if state != "ALERTS_RUNNING" or not pid:
        print("ALERTS_UNKNOWN")
        return 2
    try:
        res = subprocess.run(
            ["taskkill", "/F", "/PID", str(int(pid))],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if res.returncode == 0:
            cleanup_stale_runtime_files()
            print("ALERTS_LOOP_STOPPED")
            return 0
        print("ALERTS_LOOP_STOP_FAILED")
        return 4
    except Exception as exc:
        print(f"ALERTS_LOOP_STOP_ERROR: {exc}")
        return 4

def acquire_lock():
    if not os.path.exists(RUNTIME_DIR):
        os.makedirs(RUNTIME_DIR, exist_ok=True)

    if os.path.exists(LOCK_FILE):
        try:
            with open(LOCK_FILE, 'r') as f:
                data = json.load(f)
                old_pid = data.get("pid")
                if _pid_is_running(old_pid):
                    return False, old_pid
        except:
            pass

    # Create lock
    lock_data = {
        "pid": os.getpid(),
        "started_at": datetime.now().isoformat(),
        "command": " ".join(sys.argv),
        "cwd": os.getcwd()
    }
    with open(LOCK_FILE, 'w') as f:
        json.dump(lock_data, f, indent=2)
    with open(PID_FILE, 'w') as f:
        f.write(str(os.getpid()))
    return True, os.getpid()

def update_heartbeat():
    try:
        hb_data = {
            "pid": os.getpid(),
            "heartbeat_at": datetime.now().isoformat(),
            "timestamp": time.time()
        }
        with open(HEARTBEAT_FILE, 'w') as f:
            json.dump(hb_data, f, indent=2)
    except:
        pass

def release_lock():
    try:
        if os.path.exists(LOCK_FILE): os.remove(LOCK_FILE)
        if os.path.exists(PID_FILE): os.remove(PID_FILE)
    except:
        pass

def run_check(dry_run=False):
    state_file = os.path.join(ROOT_PATH, "MANIPULANTE", "16_OBSERVABILITY", "alerts", "alert_state.json")
    config_file = os.path.join(ROOT_PATH, "MANIPULANTE", "16_OBSERVABILITY", "alerts", "alerts_config.local.json")
    if not os.path.exists(config_file):
         config_file = os.path.join(ROOT_PATH, "MANIPULANTE", "16_OBSERVABILITY", "alerts", "alerts_config.example.json")

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except:
        config = {}

    engine = AlertEngine(ROOT_PATH)
    telegram = TelegramSender()
    email = EmailSender()
    state = AlertState(state_file)

    alerts = engine.detect_alerts()

    sent_count = 0
    for alert in alerts:
        if state.should_send(alert["dedup_key"], alert["severity"], config.get("alert_cooldown_minutes", 10)):
            msg = f"<b>{alert['severity']}: {alert['event_type']}</b>\n\n{alert['title']}\n\n{alert['message']}\n\nAccion: {alert['recommended_action']}"

            # Telegram
            if config.get("telegram_enabled", False) or telegram.enabled:
                res_tg = telegram.send_message(msg, dry_run=dry_run)
                if res_tg["status"] == "SENT":
                    sent_count += 1

            # Email
            if config.get("email_enabled", False) or email.enabled:
                if not config.get("send_critical_only") or alert["severity"] == "CRITICAL":
                    email.send_email(f"MANIPULANTE ALERT: {alert['event_type']}", msg, dry_run=dry_run)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Check completo. Alertas detectadas: {len(alerts)}. Alertas enviadas: {sent_count}")
    update_heartbeat()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--status-json", action="store_true")
    parser.add_argument("--status-line", action="store_true")
    parser.add_argument("--stop-loop", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--send-start-message", action="store_true")
    args = parser.parse_args()

    if args.status_json:
        print(json.dumps(get_alerts_status(), indent=2, ensure_ascii=True))
        return

    if args.status_line:
        status = get_alerts_status()
        pid = status["pid"] if status["pid"] else "N/A"
        hb_age = status["heartbeat_age_seconds"]
        hb_text = "N/A" if hb_age is None else f"{hb_age}s"
        env_text = "SI" if status["env_ready"] else "NO"
        print(f"{status['state']} | PID={pid} | HEARTBEAT_AGE={hb_text} | ENV_READY={env_text}")
        return

    if args.stop_loop:
        sys.exit(stop_alerts_loop())

    if args.loop:
        success, pid = acquire_lock()
        if not success:
            print(f"ALERTS_LOOP_ALREADY_RUNNING (PID: {pid})")
            sys.exit(0)

        print(f"ALERTS_LOOP_STARTED (PID: {pid}) cada {args.interval_seconds}s")

        if args.send_start_message:
            telegram = TelegramSender()
            telegram.send_message(
                f"ALERTS_LOOP_STARTED\nSistema de monitoreo MANIPULANTE activo.\nIntervalo: {args.interval_seconds}s",
                dry_run=args.dry_run,
            )

        try:
            while True:
                run_check(dry_run=args.dry_run)
                time.sleep(args.interval_seconds)
        except KeyboardInterrupt:
            print("ALERTS_LOOP_STOPPED_BY_USER")
        finally:
            release_lock()
    else:
        run_check(dry_run=args.dry_run)

if __name__ == "__main__":
    main()
