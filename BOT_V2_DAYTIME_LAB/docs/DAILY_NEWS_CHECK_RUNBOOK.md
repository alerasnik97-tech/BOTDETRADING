
# RUNBOOK: PRE-CHEQUEO DIARIO DE NOTICIAS

## 1. Procedimiento Diario
Antes de cada sesión de trading (07:00 NY), se debe ejecutar el script de pre-chequeo:
`python src/run_daily_news_fortress_precheck.py`

## 2. Interpretación de Resultados
- **ALLOW_ALL_DAY**: No hay eventos críticos. Operativa normal.
- **PARTIAL_BLOCKS**: Hay ventanas de tiempo bloqueadas. Revisar el JSON para ver las horas exactas (UTC/NY).
- **NEWS_FEED_BLOCKED**: El calendario tiene problemas. **NO OPERAR**.

## 3. Acciones en caso de Alerta
Si se detecta un evento Ultra Crítico (ej: NFP):
- Verificar que el bot esté configurado con el buffer de 120m.
- Monitorizar el spread 5 minutos antes del evento.
- Si hay posiciones abiertas, asegurar que el SL esté a una distancia segura de la volatilidad esperada.
