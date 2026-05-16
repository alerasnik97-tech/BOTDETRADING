Actuá como Claude Code Opus 4.7 Max en modo auditor institucional extremo, arquitecto senior de laboratorios cuantitativos y oficial de cumplimiento normativo de trading algorítmico.

OBJETIVO GENERAL:
Ejecutar el gate FINAL_PRE_LAB_AUDIT.

Este es el último gate antes de autorizar la apertura del laboratorio de estrategias para EURUSD.

CONTEXTO:
- EURUSD 2015-2026 Data Foundation: APPROVED (97/100).
- Train: 2015-2024.
- Holdout: 2025-2026 (Sealed).
- F06 Pipeline Infrastructure: READY.
- Governance Clean-up Phase D: COMPLETE.

TAREAS PARA ESTE GATE:
1. Validar la integridad final del registro de estrategias (STRATEGY_REGISTRY).
2. Validar que el motor (engine.py) esté congelado y no tenga cambios pendientes.
3. Auditar la capacidad del laboratorio para detectar y bloquear "future leakage" en tiempo de ejecución (no solo por path, sino por timestamp).
4. Verificar que el entorno de resultados (results/) esté limpio y listo para recibir evidencia de Fase 3.
5. Determinar si se autoriza la ejecución del primer `Phase 3 F06 Clean Train-only Rerun`.

REGLAS:
- NO backtest real todavía.
- NO optimización.
- NO tocar holdout.
- NO usar 2025/2026.

PRÓXIMO PASO:
Si este gate se aprueba, el laboratorio se declara OPEN_FOR_RESEARCH_TRAIN_ONLY.
