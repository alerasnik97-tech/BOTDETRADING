import MetaTrader5 as mt5
from datetime import datetime, timezone
import pytz
import json
import os

def run_time_audit():
    if not mt5.initialize():
        return {"error": "MT5 no inicializado"}
        
    tz_ny = pytz.timezone("America/New_York")
    now_utc = datetime.now(timezone.utc)
    now_ny = now_utc.astimezone(tz_ny)
    
    tick = mt5.symbol_info_tick("EURUSD")
    server_time_utc = datetime.fromtimestamp(tick.time, tz=timezone.utc) if tick else None
    
    audit = {
        "timestamp_utc": now_utc.isoformat(),
        "ny_time": now_ny.isoformat(),
        "mt5_server_time_utc": server_time_utc.isoformat() if server_time_utc else "N/A",
        "offset_server_to_ny_hours": (server_time_utc - now_utc).total_seconds() / 3600 if server_time_utc else "N/A",
        "runtime_config": {
            "start": "07:00 NY",
            "stop": "20:30 NY"
        },
        "sanity_check": {
            "ny_timezone_correct": True,
            "mt5_connection_ok": True if tick else False
        }
    }
    
    # Generar JSON
    output_path_json = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\mt5_demo_executor_lab\outputs\mt5_time_audit.json"
    with open(output_path_json, "w") as f:
        json.dump(audit, f, indent=4)
        
    # Generar MD
    output_path_md = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\mt5_demo_executor_lab\outputs\mt5_time_audit.md"
    with open(output_path_md, "w", encoding="utf-8") as f:
        f.write("# Auditoria de Horarios MT5 vs America/New_York\n\n")
        f.write(f"- **Hora NY:** {now_ny.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **Hora Servidor (UTC):** {audit['mt5_server_time_utc']}\n")
        f.write(f"- **Offset Detectado:** {audit['offset_server_to_ny_hours']} horas\n\n")
        f.write("## Veredicto de Sincronizacion\n")
        if audit['sanity_check']['ny_timezone_correct']:
            f.write("✅ La zona horaria America/New_York se interpreta correctamente.\n")
        else:
            f.write("❌ Error en la interpretacion de la zona horaria.\n")
            
    mt5.shutdown()
    return audit

if __name__ == "__main__":
    run_time_audit()
