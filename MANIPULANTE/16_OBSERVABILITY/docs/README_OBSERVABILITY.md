# MANIPULANTE - Observability

Esta carpeta agrega una capa local de observabilidad para MANIPULANTE.

Sirve para mirar mejor lo que ya esta pasando en forward demo:

- estado actual del bot;
- heartbeats;
- decisiones;
- gates principales;
- estado de noticias;
- estado de ordenes;
- operacion abierta si/no;
- scorecard diario;
- incidentes;
- fills manuales si se cargan despues.

No modifica:

- estrategia;
- TP, BE o BF;
- horario;
- riesgo;
- News Fortress;
- Data Quality Mask;
- Order Router;
- START / STATUS / STOP;
- MT5;
- ninguna cuenta real.

## Archivos principales

- `db/manipulante_observability.sqlite`: SQLite local generado por los scripts.
- `jsonl/bot_events.jsonl`: eventos estructurados locales.
- `daily/latest_health_snapshot.md`: snapshot simple de salud.
- `daily/YYYY-MM-DD_daily_observability_report.md`: resumen diario.
- `dashboard/dashboard.html`: dashboard local read-only cuando Streamlit no esta instalado.
- `dashboard/ABRIR_DASHBOARD_MANIPULANTE.bat`: abre el dashboard sin tocar el bot.

## Como generar datos

Desde la raiz del proyecto:

```bat
python BOT_V2_DAYTIME_LAB\src\phase44_observability_db.py
python BOT_V2_DAYTIME_LAB\src\phase44_ingest_manipulante_logs.py
python BOT_V2_DAYTIME_LAB\src\phase44_generate_health_snapshot.py
python BOT_V2_DAYTIME_LAB\src\phase44_generate_daily_report.py
```

## Como abrir el dashboard

Opcion simple:

```bat
MANIPULANTE\16_OBSERVABILITY\dashboard\ABRIR_DASHBOARD_MANIPULANTE.bat
```

Si Streamlit esta instalado:

```bat
python -m streamlit run BOT_V2_DAYTIME_LAB\src\phase44_dashboard.py
```

Si Streamlit no esta instalado, usar HTML:

```bat
python BOT_V2_DAYTIME_LAB\src\phase44_dashboard.py --export-html
```

Despues abrir:

```text
MANIPULANTE\16_OBSERVABILITY\dashboard\dashboard.html
```

## Local-only

La base SQLite y el JSONL pueden crecer y pueden contener tickets, estados de cuenta demo o datos operativos. Son locales por defecto.

No subir a GitHub si crecen o si incluyen datos sensibles:

- `db/*.sqlite`
- `jsonl/*.jsonl`
- reportes JSON diarios con tickets o datos sensibles
- logs pesados
- credenciales o tokens futuros

Los docs, scripts y dashboards livianos si se pueden versionar.
