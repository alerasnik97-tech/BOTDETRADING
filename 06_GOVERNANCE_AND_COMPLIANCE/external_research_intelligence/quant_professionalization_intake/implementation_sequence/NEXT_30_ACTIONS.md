# NEXT 30 ACTIONS — PROFESSIONALIZATION SEQUENCE

## ACCIONES PARA HACER AHORA (NO INTERFIEREN CON RESEARCH)
1. **[Governance]** Completar el intake de investigación externa (HECHO).
2. **[Infrastructure]** Normalizar rutas en `run_canonical.py` y `engine.py`.
3. **[Research]** Auditar manualmente `01_CORE_PRODUCTION` para asegurar que no hay paths absolutos.
4. **[Risk]** Verificar la lógica de cálculo de pérdida diaria para incluir Equity flotante.
5. **[Infrastructure]** Eliminar archivos ZIP y temporales de la raíz del proyecto.
6. **[Governance]** Actualizar `.gitignore` para bloquear archivos binarios de research (.pdf, .docx) en carpetas operativas.
7. **[Security]** Realizar escaneo de secrets (tokens, keys) antes de próximos pushes.

## ACCIONES PARA DESPUÉS DE V49.7B-R2
8. **[Validation]** Implementar el arnés de detección de Look-ahead Bias.
9. **[Data]** Crear el catálogo de datos con hashes (Data Lineage).
10. **[Governance]** Formalizar el "Core v8" con separación de Alpha/Risk/Execution.
11. **[Infrastructure]** Configurar un entorno de desarrollo reproducible (DevContainer/Docker).
12. **[Validation]** Realizar el primer test de `Purged Cross-Validation`.

## ACCIONES PARA DESPUÉS DE V49.7C
13. **[Execution]** Crear el modelo de slippage variable para el engine de backtest.
14. **[Validation]** Implementar simulación de Monte Carlo para validación de robustez.
15. **[Infrastructure]** Configurar observabilidad básica (logging estructurado).
16. **[Risk]** Implementar el Kill Switch manual/remoto.

## ACCIONES PARA ANTES DE PAPER TRADING
17. **[Monitoring]** Integrar alertas de Telegram para eventos de riesgo.
18. **[Execution]** Validar latencia y fills en MT5 Demo vs Backtest.
19. **[Risk]** Implementar pre-trade fat-finger limits.
20. **[Risk]** Implementar pre-trade price tolerance limits.
21. **[Governance]** Realizar auditoría de cumplimiento FIA.
22. **[Infrastructure]** Endurecer el despliegue en VPS (Docker-first).

## ACCIONES PARA ANTES DE DEMO/FONDEO
23. **[Risk]** Implementar conciliación post-trade automática.
24. **[Governance]** Generar el primer reporte inmutable de auditoría operativa.
25. **[Execution]** Evaluar migración parcial a FIX Protocol para ejecución crítica.
26. **[Portfolio]** Analizar correlación de la estrategia V49.7 con benchmarks.
27. **[Risk]** Implementar lógica de "Drawdown Trailing" si aplica a las reglas de la Prop Firm.
28. **[Security]** Endurecer la gestión de secrets en el VPS.
29. **[Governance]** Validar el "Ready for Funding" Gate.
30. **[Infrastructure]** Backup final y plan de recuperación ante desastres (DRP).
