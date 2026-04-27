# CURRENT STATE OF THE TRADING LAB

## 1. Misión Principal
Desarrollar, auditar y operar estrategias de trading cuantitativo de alta fidelidad, con un enfoque inicial en EURUSD, bajo un rigor institucional de grado bancario (H6, SCBI).

## 2. Líneas de Investigación Activas

### A. Línea H6 (Benchmark Conceptual)
- **Estrategia**: `H6_SILVER_BULLET_HYBRID`.
- **Estatus**: `FORWARD_PAPER_PHASE_1`. Congelada operativamente hasta el 30 de abril de 2026.
- **Misión**: Benchmark conceptual de diseño industrial. Sirve como ancla de gobernanza para nuevas investigaciones.

### B. Línea SCBI (Alpha Principal)
- **Estrategia**: `SCBI_M5_GLOBAL`.
- **Estatus**: `PHASE1_PAPER_FORWARD`.
- **Hito**: Superó la **Full Campaign** (2018-2024) y la **Auditoría de Durabilidad** (2020-2025).
- **Misión**: Validar el edge estructural de la fractura de liquidez en M5 bajo condiciones reales de ejecución.

### C. Línea SCBI_CORE (Alpha Purificada)
- **Estrategia**: `SCBI_CORE` (London + Asia, No PDHL).
- **Estatus**: `PHASE1_FORWARD_READY`.
- **Hito**: Superó Stage-1, Stage-2 (Execution Quality) y Full Campaign formal.
- **Misión**: Servir como la evolución de alta eficiencia de SCBI, operando en paralelo para adjudicación comparativa.

## 3. Gobernanza y Rigor Operativo
- **News Fortress**: Política de exclusión de noticias de alto impacto (30m pre / 60m post) con kill-switch manual de 10m.
- **Cost Model**: Congelado institucionalmente en 0.4 pips (0.3 spread + 0.1 slippage) para EURUSD.
- **Evidence Separation**: Cada línea (Global vs Core) mantiene ledgers, reportes y métricas en namespaces físicos separados.
- **Adjudicación Dual**: Framework operativo para comparar el desempeño de ambas líneas sin sesgo emocional.
- **Daily Orchestration**: Pipeline automatizado que coordina la operación de ambas líneas y actualiza el scoreboard.

## 4. Auditoría de Hitos (Cronología Reciente)

### P. SCBI_CORE Research Branch Authorization (OPEN_SCBI_CORE_RESEARCH_BRANCH)
- **Motivo**: Investigar si la remoción de PDH/PDL (driver de baja calidad) purifica el edge.
- **Decision**: **OPEN_SCBI_CORE_RESEARCH_BRANCH**.

### Q. SCBI_CORE Scope Lock (LOCK_SCBI_CORE_AS_LONDON_PLUS_ASIA)
- **Decision**: **LOCK_SCBI_CORE_AS_LONDON_PLUS_ASIA**.

### R. SCBI Cost Model Lock (LOCK_GLOBAL_COST_POLICY)
- **Decision**: **LOCK_GLOBAL_COST_POLICY**. Baseline en 0.4 pips.

### S. SCBI_CORE Stage-1, Stage-2 & Full Campaign
- **Decision**: **SCBI_CORE_FULL_CAMPAIGN_RESEARCH_APPROVED**. Promoción a segunda línea oficial.

### T. Dual Line Forward Adjudication Framework (READY)
- **Decision**: **DUAL_LINE_FORWARD_FRAMEWORK_READY**.

### U. Real Readiness Gate Framework (READY)
- **Motivo**: Establecer estándares institucionales duros para la promoción de líneas de paper a demo y de demo a real.
- **Decision**: **REAL_READINESS_GATE_READY**.

### V. Dual Daily Chain Orchestration (READY)
- **Motivo**: Automatizar la captura de evidencia forward dual para GLOBAL y CORE en un único pipeline robusto.
- **Estado**: **READY**. Orquestador construido, ensayado y certificado.
- **Decision**: **DUAL_DAILY_CHAIN_READY**.

### X. Prop Firm Compatibility & Risk Layer (READY)
- **Motivo**: Traducir las restricciones de firmas de fondeo al lenguaje técnico del laboratorio para asegurar la supervivencia institucional.
- **Estado**: **READY**. Capa de riesgo (DHL, Concentration, Lot Audit) implementada e integrada al orquestador dual.
- **Artefactos**: `EURUSD_PROP_FIRM_RISK_LAYER_PROTOCOL.md`, `RESULTS.md`, `DECISION.md`, `STATUS.json`, `HEARTBEAT.md`, `RUNBOOK.md`, `scratch/prop_firm_risk_guards.py`.

### AJ. Temporal & Execution Integrity Hardening (CONFIRMED)
- **Motivo**: Blindar al laboratorio contra las debilidades materiales detectadas en la Red Team Audit (DST, Reruns, Slippage, Outliers).
- **Estado**: **CONFIRMED**. El motor es ahora "DST-Aware", el orquestador es idempotente y el modelo de costos es honesto/adversarial.
- **Artefactos**: `EURUSD_TEMPORAL_EXECUTION_HARDENING_PROTOCOL.md`, `RESULTS.md`, `DECISION.md`, `STATUS.json`, `HEARTBEAT.md`, `RUNBOOK.md`, `scratch/run_temporal_integrity_audit.py`, `scratch/run_rerun_integrity_check.py`.

## 5. Conclusión de Integridad
El laboratorio ha finalizado su **Endurecimiento de Integridad Temporal y Ejecución**. Se han cerrado las brechas de infraestructura que impedían la confianza plena. El laboratorio permanece en **Régimen de Lockdown Disciplinado** con un modelo de costos y tiempo blindado institucionalmente.

---
Última Canonización: 2026-04-22 (POST TEMPORAL & EXECUTION HARDENING)
Estado Global: `HARDENING_CONFIRMED — El laboratorio está temporal y operativamente blindado.`
