# Gate Policy: DEMO_TP_PERFECT_TRADE_GATE

## 1. Propósito
Este gate establece la frontera técnica obligatoria entre la fase de laboratorio demo y la fase de **Live Sandbox (100 USD)**. Su objetivo no es validar la rentabilidad de la estrategia (eso se hace en shadow), sino validar la **integridad absoluta de la tubería de ejecución** Python ↔ MT5.

## 2. Requisito de Aprobación
Para que el gate emita un veredicto de `DEMO_TP_GATE_PASS`, debe existir al menos **un (1) trade demo completo** que cumpla con los siguientes criterios de perfección técnica:

### Criterios de Ejecución
- ✅ **Automático:** El trade debe haber sido detectado, enviado y gestionado 100% por el ejecutor Python.
- ✅ **Cuenta Demo:** Verificación de que la cuenta utilizada era de tipo DEMO.
- ✅ **Símbolo:** EURUSD únicamente.
- ✅ **Horario:** Entrada dentro de la ventana 07:00 - 20:30 NY.
- ✅ **News Guard:** Confirmación de que no hubo eventos de noticias en la ventana ±30m.

### Criterios Técnicos
- ✅ **Riesgo:** Lote calculado exactamente al 0.10% del balance.
- ✅ **SL/TP:** Colocados en el servidor de MT5 al momento de la apertura.
- ✅ **Fidelidad:** Precios de ejecución coherentes con los ticks registrados.
- ✅ **Slippage:** Desviación entre precio solicitado y precio de fill dentro de límites razonables (< 1 pip).

### Criterios de Cierre
- ✅ **Resultado:** El trade debe haber cerrado por **Take Profit (TP)**.
- ✅ **Telemetría:** Logs de `mt5_demo_log.csv` y `mt5_demo_telemetry.csv` completos y sin errores.
- ✅ **Heartbeat:** Latido activo y sin interrupciones durante toda la vida del trade.

## 3. Consecuencia del Veredicto
- **DEMO_TP_GATE_PASS:** Habilita la preparación del entorno Live Sandbox (100 USD).
- **DEMO_TP_GATE_FAIL:** Obliga a auditar el error, corregir el ejecutor y reiniciar la prueba en demo.
- **DEMO_TP_GATE_NOT_READY:** El sistema sigue operando en demo hasta que se capture el trade TP perfecto.

---
**La perfección técnica es el único estándar aceptable.**
