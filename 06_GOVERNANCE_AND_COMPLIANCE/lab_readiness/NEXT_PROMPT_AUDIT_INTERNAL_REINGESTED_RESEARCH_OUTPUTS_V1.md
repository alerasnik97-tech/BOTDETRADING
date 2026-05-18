# NEXT PROMPT: AUDIT INTERNAL REINGESTED RESEARCH OUTPUTS V1
**Date:** 2026-05-18
**Project:** Systematic Infrastructure Professionalization — Future Audit and Governance Check
**Status:** READY TO RUN ON NEXT INVOCATION — READ-ONLY QUALITY ASSURANCE

---

## 1. Context and Objective

Este prompt está diseñado para guiar al próximo agente auditor en una revisión exhaustiva y de solo lectura de la reingesta interna de los trabajos cuantitativos completada el 18 de mayo de 2026. 

El objetivo es realizar una auditoría de conformidad estricta para garantizar que el repositorio permanece limpio, ligero y que la custodia interna de las 16 fichas de investigación y síntesis es 100% conforme a las políticas vigentes de higiene del directorio raíz.

---

## 2. Mandatory Read-Only Checklist

El agente auditor deberá verificar rigurosamente los siguientes puntos sin realizar modificaciones en el código ni en el repositorio:

- [ ] **Discovery Inventory Conformance:** Verificar la existencia e integridad del inventario de descubrimiento en:
  `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/EXTERNAL_AGENT_OUTPUTS_DISCOVERY_INVENTORY_V1.csv`
- [ ] **Copied Files Custody:** Confirmar que los 16 archivos markdown copiados están presentes en sus carpetas correctas del proyecto:
  *   `03_RESEARCH_LAB/strategy_research_intake/external_research_20260518/ESTRATEGIAS_POSIBLES_ANALYSIS/` (10 archivos)
  *   `03_RESEARCH_LAB/strategy_research_intake/external_research_20260518/research_synthesis_audit/` (4 archivos)
  *   `03_RESEARCH_LAB/knowledge_intake/external_quant_project_growth_20260518/analysis/` (2 archivos)
- [ ] **Hash Match Verification:** Validar los registros en el manifiesto CSV (`INTERNAL_REINGESTED_AGENT_OUTPUTS_MANIFEST_V1.csv`) para confirmar que el SHA-256 de origen coincide al 100% con el SHA-256 de copia interna.
- [ ] **No External Canonical Dependency:** Comprobar que no existen dependencias activas de archivos localizados fuera del repositorio canónico de producción.
- [ ] **Policy Alignment:** Verificar que todos los entregables respetan la política de almacenamiento dictada en:
  `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/PROJECT_OUTPUT_LOCATION_POLICY_V1.md`
- [ ] **No Source Code or Tests Mutation:** Escanear el historial de Git para asegurar que no se ha modificado una sola línea de código fuente (`01_CORE_PRODUCTION`) ni de pruebas (`tests/`).
- [ ] **No Execution / Dry-Run Verification:** Confirmar que no se han iniciado backtests, optimizaciones, ni sweeps durante este proceso.
- [ ] **No Heavy Binaries Committed:** Inspeccionar el staging de Git para certificar que no se están subiendo archivos pesados (ZIPs, Parquets, Bases de datos crudas).
- [ ] **No Unauthorized Files Staged:** Asegurar que los únicos archivos en el stage de Git corresponden a los markdowns de gobernanza y CSVs livianos del manifiesto, abortando en caso de encontrar archivos ajenos.

---

## 3. Required Final Audit Report

El reporte de auditoría deberá presentarse con la confirmación unívoca del cumplimiento de las políticas del proyecto, firmando cada punto con `PASS`, `WARN` o `BLOCKER` con evidencia física de hashes y rowcounts.
