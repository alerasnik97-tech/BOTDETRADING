import os
import sys
import pandas as pd
import json
import time
from datetime import datetime, date
from pathlib import Path

# --- CONFIGURACIÓN DE RUTAS ---
PROJECT_ROOT = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
OBS_PATH = os.path.join(PROJECT_ROOT, "MANIPULANTE", "16_OBSERVABILITY")
LOGS_PATH = os.path.join(PROJECT_ROOT, "MANIPULANTE", "10_LOGS_PAPER")
SCORECARD_PATH = os.path.join(PROJECT_ROOT, "MANIPULANTE", "15_FORWARD_DEMO_SCORECARD")
DAILY_REPORTS_PATH = os.path.join(PROJECT_ROOT, "BOT_V2_DAYTIME_LAB", "reports", "forward_demo_daily")

# Archivos clave
BOT_EVENTS_FILE = os.path.join(OBS_PATH, "jsonl", "bot_events.jsonl")
ALERTS_HEARTBEAT = os.path.join(OBS_PATH, "alerts", "runtime", "alerts_loop.last_heartbeat.json")
MASTER_SCORECARD_CSV = os.path.join(SCORECARD_PATH, "forward_demo_master_scorecard.csv")

class ForwardDemoTracker:
    def __init__(self):
        self.today_str = datetime.now().strftime("%Y-%m-%d")

    def load_events(self, target_date):
        events = []
        if not os.path.exists(BOT_EVENTS_FILE): return events
        with open(BOT_EVENTS_FILE, "r") as f:
            for line in f:
                try:
                    ev = json.loads(line)
                    if ev.get("timestamp", "").startswith(target_date):
                        events.append(ev)
                except: continue
        return events

    def analyze_day(self, target_date):
        events = self.load_events(target_date)
        
        metrics = {
            "date": target_date,
            "bot_started": False,
            "bot_stopped": False,
            "errors": 0,
            "warnings": 0,
            "decisions": 0,
            "no_trade_reasons": {},
            "trades": 0,
            "news_blocked": False,
            "data_quality_blocked": False,
            "mt5_available": False,
            "status": "UNKNOWN"
        }

        for ev in events:
            msg = ev.get("message", "").lower()
            level = ev.get("level", "INFO")
            
            if "bot started" in msg: metrics["bot_started"] = True
            if "bot stopped" in msg: metrics["bot_stopped"] = True
            if level == "ERROR": metrics["errors"] += 1
            if level == "WARNING": metrics["warnings"] += 1
            if "decision" in msg: metrics["decisions"] += 1
            if "trade executed" in msg: metrics["trades"] += 1
            if "news fortress" in msg and "blocked" in msg: metrics["news_blocked"] = True
            if "data quality" in msg and "blocked" in msg: metrics["data_quality_blocked"] = True
            if "mt5 connected" in msg: metrics["mt5_available"] = True

        # Clasificación simplificada
        if not metrics["bot_started"] and not events:
            metrics["status"] = "FORWARD_DAY_UNKNOWN_REQUIRES_REVIEW"
        elif metrics["trades"] > 0:
            metrics["status"] = "FORWARD_DAY_VALID_TRADE"
        elif metrics["news_blocked"]:
            metrics["status"] = "FORWARD_DAY_BLOCKED_BY_NEWS_OK"
        elif metrics["data_quality_blocked"]:
            metrics["status"] = "FORWARD_DAY_BLOCKED_BY_DATA_QUALITY_OK"
        elif metrics["errors"] > 5:
            metrics["status"] = "FORWARD_DAY_INVALID_TECHNICAL"
        else:
            metrics["status"] = "FORWARD_DAY_VALID_NO_TRADE"

        return metrics

    def update_scorecard(self, metrics):
        if os.path.exists(MASTER_SCORECARD_CSV):
            df = pd.read_csv(MASTER_SCORECARD_CSV)
            # Evitar duplicados por fecha
            df = df[df["date"] != metrics["date"]]
        else:
            df = pd.DataFrame()

        new_row = {
            "date": metrics["date"],
            "status": metrics["status"],
            "trades": metrics["trades"],
            "errors": metrics["errors"],
            "warnings": metrics["warnings"],
            "decisions": metrics["decisions"],
            "can_count_for_sample": 1 if "VALID" in metrics["status"] or "BLOCKED" in metrics["status"] else 0
        }
        
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.sort_values("date", inplace=True)
        df.to_csv(MASTER_SCORECARD_CSV, index=False)
        return MASTER_SCORECARD_CSV

    def generate_report(self, metrics):
        report_name = f"MANIPULANTE_FORWARD_DAILY_REVIEW_{metrics['date'].replace('-', '')}.md"
        report_path = os.path.join(DAILY_REPORTS_PATH, report_name)
        
        content = f"""# FORWARD DEMO DAILY REVIEW - {metrics['date']}

## 1. Lo más importante
Veredicto: **{metrics['status']}**

## 2. Estado Operativo
- Bot iniciado: {metrics['bot_started']}
- Bot detenido: {metrics['bot_stopped']}
- MT5 disponible: {metrics['mt5_available']}
- Errores: {metrics['errors']}
- Warnings: {metrics['warnings']}

## 3. Decisiones y Trades
- Decisiones evaluadas: {metrics['decisions']}
- Trades ejecutados: {metrics['trades']}
- Bloqueo por Noticias: {metrics['news_blocked']}
- Bloqueo por Data Quality: {metrics['data_quality_blocked']}

## 4. Conclusión
¿Cuenta para muestra forward?: **{"SÍ" if metrics['status'].startswith("FORWARD_DAY_VALID") or "BLOCKED" in metrics['status'] else "NO"}**
"""
        with open(report_path, "w") as f:
            f.write(content)
        return report_path

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str)
    parser.add_argument("--today", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    target_date = args.date if args.date else datetime.now().strftime("%Y-%m-%d")
    
    tracker = ForwardDemoTracker()
    metrics = tracker.analyze_day(target_date)
    
    if args.dry_run:
        print(f"[DRY-RUN] Analysis for {target_date}: {metrics['status']}")
    else:
        csv_path = tracker.update_scorecard(metrics)
        report_path = tracker.generate_report(metrics)
        print(f"[SUCCESS] Report: {report_path}")
        print(f"[SUCCESS] Scorecard updated: {csv_path}")
        print(f"Veredicto: {metrics['status']}")

if __name__ == "__main__":
    main()
