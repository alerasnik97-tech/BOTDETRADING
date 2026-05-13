# MANIFIESTO DE CONGELAMIENTO CRIPTOGRÁFICO DEL ORQUESTADOR (RUNNER HASH FREEZE)

## 1. Sellado de Firmas Críticas
El entorno de ejecución queda vinculado de forma incondicional a las siguientes firmas de integridad computadas en caliente (SHA256):

| Componente | Ruta Relativa | Hash SHA256 |
| :--- | :--- | :--- |
| **Runner Orquestador** | `run_r1_micro_probe.py` | `17ad484e8ddb02a0364bb1b47cfd51471c9762c1d06b83d2e5c796f7de1f6e16` |
| **Detector R1** | `src/R1/r1_detector.py` | `ff7e54296a6e8a9dc39a48f744edc016887fddd220fc2691424950282d20ecdb` |
| **Extractor Niveles R1** | `src/R1/r1_levels.py` | `95a734aae9420eedbf8c65fa461ca1ba46d2b2150a5bae1ff29f10bad5ae82d0` |
| **Motor Central Core** | `src/v7_engine/engine.py` | `84319e04a7943297f2dcc9c1ba67d29c22c8e4cd0a2a81dd2960583b07985777` |
| **Modelo de Costos FTMO** | `src/v7_engine/cost_model.py` | `6b2aa3e238031bf6d97f03e8ccbc11105ead7fd434b0e3af06ba2f0e45ed1b35` |
| **Constructor Barras Causales** | `src/v6_utils/bars.py` | `ff4e4cc00397bd774ec772cbb565a42eb84ddc6aba3e5f533f476856ff836ac2` |
| **Motor Ejecución V6** | `src/v6_utils/execution.py` | `014d477e20f75030caeb5455926913d46f43fe474a5a9da80e8350da5cc559fa` |

- **Timestamp de Sellado**: 2026-05-13T17:25:32Z
- **Cambios Recientes**: Orquestador enlazado exitosamente al bastión de inmutabilidad de core (`ENGINE_CORE_VERIFY.py`) inyectando validación bloqueante de preflight. Fuentes del motor estrictamente inmutables sin deriva funcional.

## 2. Declaración de Inmutabilidad
**EL RUNNER ESTÁ 100% CONGELADO.**
Cualquier alteración en un solo byte de los archivos listados durante el desarrollo del backtest gatillará incondicionalmente la invalidación global de los resultados bajo la causal de fallo:
`BLOCKED_BY_RUNNER_DRIFT`
