# MT5 Risk Profile: Micro-Piloto (Manual Exception)

**Veredicto:** `NO_AUTOMATIC_LIVE_RISK_PROFILE_APPROVED`

Para la ejecución manual autorizada en MT5, se define el siguiente perfil de riesgo **innegociable**:

## 1. Límites de Exposición
- **Riesgo por trade:** 0.10% (Diez puntos básicos).
- **Lote:** Tamaño mínimo (microlotes) que permita cumplir el riesgo de 0.10%.
- **Máximo trades por día:** 1.
- **Máximo posiciones abiertas:** 1.

## 2. Límites de Pérdida Duros
- **Stop Diario:** 1.0% (Bloqueo de operativa ante errores de dedo o slippage).
- **Stop Semanal:** 2.5%.
- **Drawdown del Piloto:** 5.0% (Cierre definitivo de la fase de excepción manual).

## 3. Restricciones
- **Prohibido el compounding:** No se aumenta el riesgo tras ganar.
- **Prohibido el revenge trading:** Tras el trade diario, la plataforma se cierra.
- **Prohibido operar noticias:** Respetar estrictamente la ventana de ±30m.

---
**La protección del capital es el único objetivo. El profit es secundario.**
