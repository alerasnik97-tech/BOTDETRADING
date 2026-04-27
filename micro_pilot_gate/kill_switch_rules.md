# Reglas del Kill Switch (Micro Piloto)

El piloto se detendrá inmediatamente y volverá a estado `SHADOW_ONLY` si ocurre cualquiera de estos eventos:

## 1. Breaches de Riesgo
- **Drawdown Diario > 1.5%**: Pausa del día.
- **Drawdown Total Piloto > 5%**: Suspensión definitiva del piloto.
- **3 Pérdidas Consecutivas**: Revisión obligatoria del tribunal antes de seguir.

## 2. Breaches Técnicos
- **Desconexión con el Feeder**: Si los datos fallan, no hay trade.
- **Inconsistencia de Señal**: Si el Shadow Runner y el Real Runner difieren en la clasificación de una señal.
- **Error de Ejecución**: Fallo en el envío o cierre de órdenes.

## 3. Condiciones de Bloqueo Externo
- **Eventos de Cisne Negro**: Volatilidad extrema inexplicable.
- **Fallo del Shadow Autopilot**: Si la capa de gobernanza shadow se bloquea.

---
**ACCIÓN ANTE KILL SWITCH:**
1. Cerrar posiciones abiertas manualmente o mediante script de emergencia.
2. Desactivar el flag de `REAL_PILOT_ACTIVE`.
3. Documentar la causa en el log de gobernanza.
