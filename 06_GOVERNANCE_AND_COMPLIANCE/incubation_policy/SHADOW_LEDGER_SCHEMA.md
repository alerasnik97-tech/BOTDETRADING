# SHADOW LEDGER SCHEMA

El Shadow Ledger es la fuente de verdad para la fase de incubación. A continuación se define la estructura de datos para el registro de cada trade/señal.

## Columnas del Dataset

| Columna | Descripción |
| :--- | :--- |
| `trade_id` | Identificador único del trade (UUID o correlativo) |
| `date` | Fecha de la señal (YYYY-MM-DD) |
| `time_ny` | Hora exacta en New York (HH:MM:SS) |
| `symbol` | Par operado (ej. EURUSD) |
| `strategy_id` | Identificador de la estrategia (ej. M4_V1) |
| `setup_type` | Tipo de setup detectado |
| `htf_context` | Contexto de temporalidad superior |
| `ltf_trigger` | Gatillo en temporalidad inferior |
| `liquidity_level` | Nivel de liquidez institucional identificado |
| `direction` | BUY / SELL |
| `planned_entry` | Precio de entrada según el algoritmo |
| `actual_entry` | Precio de entrada real ejecutado |
| `stop_loss` | Precio de Stop Loss |
| `take_profit` | Precio de Take Profit |
| `risk_r` | Riesgo en unidades R (ej. 1R) |
| `gross_r` | Resultado bruto en R |
| `commission_r` | Costo de comisión en R |
| `slippage_r` | Costo de slippage en R |
| `net_r` | Resultado neto final en R |
| `spread_pips` | Spread real al momento de la entrada |
| `news_context` | Noticias de alto impacto cercanas |
| `rollover_blocked` | Indica si el trade fue bloqueado por rollover |
| `signal_valid` | Indica si la señal cumplía todas las reglas |
| `order_sent` | Boolean: ¿Se envió la orden al broker? |
| `order_filled` | Boolean: ¿Se ejecutó la orden? |
| `fill_quality` | Diferencia entre precio solicitado y ejecutado |
| `exit_reason` | TP, SL, Trail, Manual, Time-exit, Kill-Switch |
| `screenshot_path` | Ruta a la imagen del trade (opcional) |
| `notes` | Observaciones manuales |
| `technical_error` | Descripción de cualquier fallo técnico |
| `rule_violation` | Descripción si se violó alguna regla de gestión |
| `included_in_review` | Boolean para auditoría final |
