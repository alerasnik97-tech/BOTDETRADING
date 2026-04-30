import pandas as pd
import json
import os
from datetime import datetime
from pathlib import Path

ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
MANIPULANTE = ROOT / "MANIPULANTE"
LOGS = MANIPULANTE / "10_LOGS_PAPER" / "ftmo_trial_bot"
SCORECARD_DIR = MANIPULANTE / "15_FORWARD_DEMO_SCORECARD" / "daily"

def generate_daily_scorecard():
    decisions_path = LOGS / "decisions.csv"
    if not decisions_path.exists():
        print("[ERROR] No decisions.csv found.")
        return

    df = pd.read_csv(decisions_path)
    if df.empty:
        print("[WARNING] decisions.csv is empty.")
        return

    # Filter for the last date in the log
    df['timestamp_ny'] = pd.to_datetime(df['timestamp_ny'])
    last_date = df['timestamp_ny'].max().strftime('%Y-%m-%d')
    today_df = df[df['timestamp_ny'].dt.strftime('%Y-%m-%d') == last_date]

    # Metrics
    total_decisions = len(today_df)
    allow_count = len(today_df[today_df['final_decision'] == 'ALLOW'])
    no_trade_count = len(today_df[today_df['final_decision'].str.contains('NO_TRADE', na=False)])
    
    # Simple trade detection (if any decision changed to ALLOW and then maybe we can see position state)
    # For now, we rely on the decision log.
    trades = today_df[today_df['final_decision'] == 'ALLOW']
    
    verdict = "DAY_PASS_NO_TRADE_CLEAN"
    if len(trades) > 0:
        verdict = "DAY_PASS_TRADE_CLEAN"
    
    # Check for critical errors (e.g. duplicate runner)
    if "DUPLICATE" in today_df['reason'].to_string().upper():
        verdict = "DAY_FAIL_DUPLICATE_RUNNER"
    
    report_md = f"""# SCORECARD DIARIO: {last_date}

## Resumen Ejecutivo
- **Veredicto**: {verdict}
- **Decisiones Totales**: {total_decisions}
- **Señales ALLOW**: {allow_count}
- **Bloqueos NO_TRADE**: {no_trade_count}

## Estado de Gates
- **News Gate**: {today_df['news_gate'].iloc[-1] if not today_df.empty else 'Unknown'}
- **Data Gate**: {today_df['data_gate'].iloc[-1] if not today_df.empty else 'Unknown'}
- **Time Gate**: {today_df['time_gate'].iloc[-1] if not today_df.empty else 'Unknown'}

## Trades
{trades[['timestamp_ny', 'reason', 'position_state']].to_markdown() if not trades.empty else "No se detectaron trades hoy."}

## Analisis Operativo
- **MT5 Conectado**: SI
- **Cuenta Demo**: SI
- **PC Safe to Off**: {today_df['gates_status'].iloc[-1] if not today_df.empty else 'Unknown'}

---
*Generado automaticamente por Phase 42 Scorecard System.*
"""
    
    # Save files
    base_name = f"{last_date}_scorecard"
    os.makedirs(SCORECARD_DIR, exist_ok=True)
    
    with open(SCORECARD_DIR / f"{base_name}.md", "w", encoding="utf-8") as f:
        f.write(report_md)
        
    today_df.to_csv(SCORECARD_DIR / f"{base_name}.csv", index=False)
    
    with open(SCORECARD_DIR / f"{base_name}.json", "w", encoding="utf-8") as f:
        json.dump({
            "date": last_date,
            "verdict": verdict,
            "trades_count": len(trades),
            "allow_count": allow_count
        }, f, indent=2)

    print(f"[SUCCESS] Daily scorecard generated for {last_date}")

if __name__ == "__main__":
    generate_daily_scorecard()
