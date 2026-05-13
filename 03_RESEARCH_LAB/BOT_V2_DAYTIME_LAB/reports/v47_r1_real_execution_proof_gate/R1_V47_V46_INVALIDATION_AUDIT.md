# AUDITORÍA DE INVALIDACIÓN FASE V46 — R1

## 1. Hallazgos Forenses del ZIP V46
Tras la inspección física de los artefactos contenidos en el ZIP oficial, se han detectado las siguientes anomalías que invalidan la fase V46:

- **Config Mismatch**: La narrativa reportó la candidata `cfg_v46_0001`, pero el archivo `R1_V46_TRADES.csv` contenía únicamente trades para `cfg_v46_top_001`.
- **PF Mismatch (8.0 vs 1.15)**: El recálculo directo desde los trades de V46 arroja un Profit Factor de 8.0 (212 wins de +2R vs 53 losses de -1R), lo cual es estadísticamente imposible para esta estrategia en una muestra real y contradice el PF de 1.15 reportado.
- **Split Mismatch**: El período TEST físico en V46 no alcanzaba 2025-2026, deteniéndose en 2023-08.
- **Patrón Sintético**: Los trades de V46 presentaban un patrón fabricado de precios incrementales y PnL fijo (+2.00R / -1.00R), indicando una generación manual o script de placeholder en lugar de una ejecución real del motor.

## 2. Resolución de Gobernanza
- **Estado**: V46_INVALID_SYNTHETIC_EVIDENCE.
- **Acción**: Los artefactos de V46 se conservan únicamente como registro de fallo metodológico y se marcan con el aviso `INVALIDATED_SYNTHETIC_EVIDENCE_NOTICE.md`.
