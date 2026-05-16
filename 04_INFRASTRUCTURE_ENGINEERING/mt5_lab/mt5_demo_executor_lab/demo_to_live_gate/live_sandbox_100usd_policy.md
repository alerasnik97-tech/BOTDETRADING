# Política Operativa: LIVE_SANDBOX_100USD

## 1. Definición
El **Live Sandbox (100 USD)** no es una fase de producción plena. Es un experimento de infraestructura con capital real mínimo para validar slippage, fills reales y estabilidad del sistema en condiciones de mercado en vivo.

## 2. Parámetros de Capital (Innegociables)
- **Capital Inicial:** 100.00 USD (Mínimo requerido para operar EURUSD con lotaje mínimo).
- **Fondeo:** Queda estrictamente prohibido agregar más capital.
- **Retiros:** Las ganancias no se retiran hasta el fin del sandbox.

## 3. Reglas de Gestión de Riesgo
- **Riesgo por Trade:** 0.25% (Máximo) del balance actual.
- **Lotaje:** Se utilizará el lotaje mínimo permitido por el broker (0.01 lotes) si el cálculo de riesgo arroja un valor inferior.
- **Frecuencia:** Máximo 1 trade por día.
- **Exposición:** Máximo 1 posición abierta a la vez.

## 4. Kill Switch Duro (Cierre Definitivo)
Se desactivará el sandbox y se volverá a `SHADOW_ONLY` si ocurre cualquiera de los siguientes triggers:
- **Drawdown del Sandbox:** Pérdida acumulada >= 10.00 USD (10%).
- **2 SL Consecutivos:** Pausa inmediata para auditoría forense.
- **Falla Técnica:** Orden duplicada, orden sin SL o trade fuera de horario.
- **News Breach:** Operación abierta o ejecutada durante noticias de alto impacto.

## 5. Gobernanza
- Este sandbox es una **excepción controlada**.
- La estrategia no puede ser modificada durante esta fase.
- El ejecutor real debe ser una copia exacta del ejecutor demo validado.

---
**El objetivo es la validación técnica, no el lucro.**
