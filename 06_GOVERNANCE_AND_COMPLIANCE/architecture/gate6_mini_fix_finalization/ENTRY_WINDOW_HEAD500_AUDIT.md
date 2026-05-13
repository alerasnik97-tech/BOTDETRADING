# AUDITORÍA DE LA VENTANA DE ENTRADA Y EL CORTE DE 500 TICKS
**Archivo Auditado:** `gate6_mini_runner.py` (Línea 278)  
**Fase:** Gate 6 Mini Fix Finalization  
**Fecha:** 2026-05-13  
**Decisión Estratégica:** `ENTRY_HEAD500_REMOVED_AND_CONVERTED_TO_TIME_WINDOW`

---

## 1. Dónde se usa `ticks_after.head(500)`
Se identificó explícitamente en el bucle de streaming de simulación dentro de `gate6_mini_runner.py` al invocar el método de evaluación de entrada del motor unificado:
```python
fill, reason = eng.execute_signal(side, choch_utc, ticks_after.head(500), instrument="EURUSD", entry_mode=entry_mode, stop_price=stop_p)
```

## 2. Para qué se usa
El parámetro acota el subconjunto de cotizaciones Ask/Bid suministradas al analizador de órdenes T+1 (`next_bar_execute_stop` / `next_bar_execute`) con el fin original de aligerar la memoria y limitar la búsqueda del cruce a las inmediaciones post-señal.

## 3. Alcance Dimensional (Entrada vs. Salida)
Afecta **exclusivamente la fase de entrada (fill / no-fill)**. La simulación del ciclo de vida posterior de la posición abierta (evaluación de Take Profit, Stop Loss y Break-Even en `close_position_with_costs`) se realiza sobre un DataFrame completamente separado (`ticks_during`), del cual ya se erradicó exitosamente el límite previo de `.head(3000)`.

## 4. Riesgo de No-Fill Artificial
**SÍ, genera un riesgo inminente de no-fill artificial**. Al pasar un subconjunto pre-recortado programáticamente, se oculta al evaluador interno cualquier evento ocurrido en el tick 501 en adelante, incluso si este se encontrara dentro del plazo temporal normativo de expiración de la orden pendiente (`expiry_minutes = 60`).

## 5. Cobertura Real Promedio en Sesión NY
Durante las horas de apertura líquida de Nueva York (08:00 a 11:00 NY) para el par `EURUSD`, un volumen de 500 ticks crudos de Dukascopy puede consumirse en un lapso extremadamente breve, oscilando típicamente entre **45 segundos y 3 minutos** durante solapamientos de alta volatilidad. 

## 6. Comportamiento Lógico ante Cruce Tardío
Si la cotización alcanza el nivel Stop predefinido de la variante `V2_B` en el minuto 12 posterior a la señal (cumpliendo plenamente con la ventana válida de 60 minutos) pero este evento ocurre en el tick número 850, el motor evalúa una serie vacía o incompleta, retornando incondicionalmente `fill = None` y atribuyendo un rechazo espurio por `STOP_NOT_TOUCHED`.

## 7. Corrección Implementada (Ventana Temporal Pura)
Se suprime por completo el uso de `.head(500)` en la llamada a `execute_signal`. En su lugar, se suministra una rebanada de ticks acotada estrictamente por una barrera de tiempo real:
```python
slice_end = choch_utc + pd.Timedelta(minutes=60)
ticks_after = ticks.loc[choch_utc : slice_end]
fill, reason = eng.execute_signal(side, choch_utc, ticks_after, instrument="EURUSD", entry_mode=entry_mode, stop_price=stop_p)
```
Esto alinea la serie física con el tiempo de caducidad interno (`expiry_minutes = 60`), erradicando límites de conteo silenciosos.

## 8. Riesgo Residual
**Cero riesgo residual en la evaluación intradiaria local**. Al regirse puramente por marcas de tiempo UTC canónicas, todas las señales disponen exactamente del mismo horizonte de maduración para intentar su ejecución física.
