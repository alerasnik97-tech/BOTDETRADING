# ÍNDICE MAESTRO DE ESTRATEGIAS

Este es el registro oficial de todas las versiones, candidatos y experimentos del proyecto.

| Nombre / Variante | Fase de Origen | Parámetros Clave | Estado | Motivo / Diagnóstico | Usar en Fondeo | Ruta del Reporte/Directorio | Riesgo Permitido |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MANIPULANTE** | Phase25 | TP 1.4, BE 0.4, BF 70 | **CURRENT_AUTHORITY** | Máxima robustez, menor DD, estable por 10 años. | **SÍ** | `MANIPULANTE/` | 0.50% |
| **TP1.4_BE0.5_BF70** | Phase30 | TP 1.4, BE 0.5, BF 70 | **SHADOW_CANDIDATE_ONLY** | Mejora WR y rachas, pero menor PF y warnings de M3. | **NO** (Shadow) | `ESTRATEGIAS/02_CANDIDATOS_SHADOW/TP14_BE05_BF70` | 0.00% (Observación) |
| **PHASE18** | Phase18 | (Pre-H1 Sweep refinement) | **BASELINE_PROTECTED** | Primer modelo estable sin lookahead, superado por P25. | **NO** | `ESTRATEGIAS/01_BASELINES/PHASE18_BASELINE` | N/A |
| **PHASE24** | Phase24 | TP 1.4, BE 0.4, BF 60 | **SUPERSEDED / BACKUP** | Robusto, pero la P25 con BF 70 la superó en selectividad. | **NO** | `ESTRATEGIAS/01_BASELINES/PHASE24_BACKUP` | N/A |
| **NO_BE / WR Alta** | Phase8-15 | TP variable, Sin BE | **REJECTED** | WR alto pero DD inaceptable, quiebra cuentas de prop firms. | **NO** | `ESTRATEGIAS/03_RECHAZADAS` | N/A |
| **PHASE19** | Phase19 | Lookahead en M15/M3 | **ARCHIVED_DO_NOT_USE** | Resultados artificialmente inflados por bug de MT5/Pandas. | **NUNCA** | `ESTRATEGIAS/04_ARCHIVADAS/PHASE19` | N/A |
| **PHASE28 / 29** | Phase28-29 | WR / Loss Streak Comp. | **RESEARCH_EXPERIMENTS** | Estudios sobre filtros temporales y de momentum. | **NO** | `ESTRATEGIAS/05_EXPERIMENTOS` | N/A |

*Nota: Cualquier candidato no listado explícitamente se asume bajo el estado `UNKNOWN_REVIEW_REQUIRED` y no debe operarse.*
