import pandas as pd
import json
import os
from pathlib import Path

ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
MANIPULANTE = ROOT / "MANIPULANTE"
SCORECARD_DIR = MANIPULANTE / "15_FORWARD_DEMO_SCORECARD"
DASHBOARD_MD = SCORECARD_DIR / "FORWARD_DEMO_DASHBOARD.md"

def update_dashboard():
    daily_dir = SCORECARD_DIR / "daily"
    if not daily_dir.exists():
        print("[WARNING] No daily scorecards found.")
        return

    # Aggregate daily json files
    scorecards = []
    for f in daily_dir.glob("*.json"):
        with open(f, "r", encoding="utf-8") as handle:
            scorecards.append(json.load(handle))

    if not scorecards:
        print("[WARNING] No scorecards data.")
        return

    df = pd.DataFrame(scorecards)
    days_observed = len(df)
    total_trades = df['trades_count'].sum()
    fails = len(df[df['verdict'].str.contains('FAIL', na=False)])
    
    progress_20 = min(100, (total_trades / 20) * 100)
    progress_30 = min(100, (total_trades / 30) * 100)
    
    can_promote = "NO"
    if total_trades >= 20 and fails == 0:
        can_promote = "SI"

    dashboard_content = f"""# MANIPULANTE - FORWARD DEMO DASHBOARD

## Estado Actual
- **Dias Observados**: {days_observed}
- **Trades Tomados (Demo)**: {total_trades}
- **Errores Criticos (FAIL)**: {fails}
- **¿Puede considerar cuenta paga?**: **{can_promote}**

## Progreso de Promocion
- **Hacia 20 Trades**: [{int(progress_20)}%] {'#' * (int(progress_20)//5)}{'.' * (20 - int(progress_20)//5)}
- **Hacia 30 Trades**: [{int(progress_30)}%] {'#' * (int(progress_30)//5)}{'.' * (20 - int(progress_30)//5)}

## Resumen de Veredictos
{df['verdict'].value_counts().to_markdown()}

---
*Ultima actualizacion: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
*Ejecuta BOT_V2_DAYTIME_LAB\src\phase42_update_forward_dashboard.py para actualizar.*
"""
    with open(DASHBOARD_MD, "w", encoding="utf-8") as f:
        f.write(dashboard_content)
        
    print(f"[SUCCESS] Dashboard updated at {DASHBOARD_MD}")

from datetime import datetime
if __name__ == "__main__":
    update_dashboard()
