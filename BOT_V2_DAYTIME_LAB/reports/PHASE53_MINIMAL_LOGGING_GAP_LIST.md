# PHASE 53 — MINIMAL LOGGING GAP LIST

## 1. Obligatorio (Para Validar el Edge)
- **Fills Persistent Log:** Crear un `fills.csv` que registre cada `order_send` exitoso con su `executed_price`, `bid`, `ask` y `order_ticket`.
- **Exit Logic Logging:** Modificar `phase37x_safe_close.py` para que registre el precio de salida real, el motivo del cierre (`TIME_EXIT` vs `TP/SL`) y los spreads en el momento del cierre.
- **Cost Calculation:** Implementar el cálculo automático de `slippage_R` y `spread_R` en cada trade.

## 2. Recomendable (Para Auditoría Forense)
- **Order Lifecycle Trace:** Unir el `signal_id` con el `order_ticket` en todos los logs para permitir una trazabilidad completa desde la señal hasta el cierre.
- **Latency Monitoring:** Registrar el tiempo transcurrido entre la señal (`signal_time`) y la ejecución (`fill_time`).

## 3. Opcional (Para Optimización)
- **Market Depth Snapshot:** Registrar el libro de órdenes (si está disponible) al momento de la entrada para predecir slippage en lotajes mayores.
