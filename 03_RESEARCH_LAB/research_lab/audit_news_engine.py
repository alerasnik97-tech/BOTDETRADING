import pandas as pd
from research_lab.news_filter import SUPPORTED_FIXED_SCHEDULES_NY, normalize_event_name

def audit_news_integrity():
    print("--- Auditoría de Integridad del Motor de Noticias ---")
    
    # Intentamos cargar el dataset validado
    try:
        df = pd.read_csv("data/news_eurusd_v2_utc.csv")
    except Exception as e:
        print(f"Error al cargar news_eurusd_v2_utc.csv: {e}")
        return

    # 1. Alineación de Timestamps NY
    df['timestamp_ny'] = pd.to_datetime(df['timestamp_ny'], utc=True).dt.tz_convert("America/New_York")
    
    print(f"Total eventos aprobados: {len(df)}")
    
    # 2. Verificación de Eventos Críticos (NFP, CPI)
    key_events = ["non-farm employment change", "cpi y/y", "fomc statement"]
    
    for event in key_events:
        subset = df[df['event_name_normalized'] == event].copy()
        if subset.empty:
            print(f"[FAIL] Evento '{event}' no encontrado en el dataset validado.")
            continue
            
        times = subset['timestamp_ny'].dt.strftime('%H:%M').unique()
        expected = SUPPORTED_FIXED_SCHEDULES_NY.get(event)
        
        print(f"Evento: {event}")
        print(f"  Horarios detectados: {times}")
        print(f"  Horario esperado NY: {expected}")
        
        if expected in times and len(times) == 1:
            print(f"  [PASS] Alineación horaria perfecta.")
        else:
            print(f"  [WARNING] Discrepancia horaria detectada.")

    print("\n--- Conclusión Preliminar ---")
    print("Si los eventos críticos están alineados a 08:30 y 14:00 NY, el motor es 'estructuralmente seguro'.")

if __name__ == "__main__":
    audit_news_integrity()
