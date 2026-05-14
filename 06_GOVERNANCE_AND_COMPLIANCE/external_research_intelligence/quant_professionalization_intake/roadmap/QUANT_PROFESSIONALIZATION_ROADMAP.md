# QUANT PROFESSIONALIZATION ROADMAP — V49.7

## FASE A — AHORA MISMO (Higiene y Seguridad)
> [!NOTE]
> Acciones de bajo impacto en el código pero alto impacto en la calidad del laboratorio.
- **Root Cleanup:** Eliminar archivos temporales y ZIPs de la raíz.
- **Path Normalization:** Migrar a rutas relativas en todos los scripts de research.
- **Harness Anti-Lookahead:** Implementar verificaciones de causalidad en el engine.
- **Equity-based Risk Audit:** Validar que el control de pérdida diaria use el Equity flotante.

## FASE B — DESPUÉS DE V49.7B-R2 (Cimientos del Motor)
- **Refactor Core v8:** Separar lógicamente `Alpha`, `Risk` y `Execution`.
- **Data Cataloging:** Implementar un sistema de versionado de datasets con checksums.
- **Audit Logging:** Implementar un registro inmutable de decisiones del bot.

## FASE C — DESPUÉS DE V49.7C (Validación Avanzada)
- **Purged/Combinatorial CV:** Implementar validación cruzada avanzada para reducir leakage.
- **Slippage Modeling:** Crear una capa de simulación de fricción basada en volatilidad.
- **Monte Carlo Engine:** Simulación de caminos aleatorios sobre los resultados de backtest.

## FASE D — ANTES DE PAPER TRADING (Preparación Operativa)
- **Operational Monitoring:** Dashboard básico (Grafana) y alertas (Telegram).
- **Kill Switch:** Implementar mecanismo de parada de emergencia remoto.
- **Pre-trade Controls:** Límites de tamaño de orden y filtros de noticias UTC-correctos.

## FASE E — ANTES DE DEMO/FONDEO (Industrialización)
- **Containerization:** Despliegue del bot en Docker para garantizar estabilidad en VPS.
- **Post-trade Reconciliation:** Script de comparación automática Bot vs MT5 History.
- **Fail-closed System:** Garantizar que ante cualquier error el bot quede en estado neutro.

## FASE F — FUTURO PORTFOLIO (Gestión de Capital)
- **Correlation Matrix:** Análisis de correlación entre múltiples estrategias.
- **Risk Parity / Kelly Criterion:** Lógica avanzada de asignación de capital.
- **FIX Protocol Integration:** Evaluación de conectividad institucional directa.
