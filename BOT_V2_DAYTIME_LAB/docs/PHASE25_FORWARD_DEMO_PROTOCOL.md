
# PHASE 25: FORWARD DEMO PROTOCOL (PAPER ONLY)

## 1. RESTRICCIONES CRÍTICAS
- **SOLO CUENTA DEMO / PAPER**.
- **REAL TRADING BLOQUEADO**.
- **PROHIBIDO MODIFICAR PARÁMETROS** sin auditoría previa.

## 2. REQUISITOS DE EJECUCIÓN
- **Estrategia**: Phase 25 (TP 1.4 / BE 0.4).
- **Timeframe**: M3.
- **Ventana**: 07:00 - 16:30 NY.
- **News Fortress**: Activo (No operar si no hay ALLOW explícito).
- **Data Mask**: Activa (No operar si hay bloqueo por calidad).

## 3. PROTOCOLO DIARIO
1. **Pre-check**: Validar Hash de la config.
2. **News check**: Confirmar que News Fortress está actualizado.
3. **Execution**: Solo 1 trade por día máximo.
4. **Log**: Registrar cada trade con captura de pantalla y metadatos.

## 4. CRITERIOS DE REVISIÓN
- Se requieren mínimo **30 trades** para la primera evaluación estadística.
- Ideal **50 trades** para validación de robustez.
- Si el DD excede -6.0R (1.2x histórico), **PAUSAR**.

---
*Firma: FORWARD_DEMO_GATEKEEPER_20260428*
