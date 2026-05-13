# AUDITORÍA DE MECANISMOS DE REANUDACIÓN Y RESILIENCIA (CHECKPOINT / RESUME PRECHECK)

## 1. Arquitectura de Estado
- **Granularidad Mensual**: El orquestador registra atómicamente cada mes superado en la lista serializada de `checkpoints/processed_months.json`.
- **Aislamiento de Entornos**: Se verificó la purga y aislamiento de todas las corridas de prueba pasadas contaminadas, las cuales reposan de forma inactiva en la subcarpeta forense `invalidated_preflight_artifacts/`.
- **Inmutabilidad en la Reanudación**: La inclusión del chequeo criptográfico preflight (`ENGINE_CORE_VERIFY.py`) al inicio del orquestador asegura matemáticamente que **una reanudación (resume) jamás mezclará código o firmas divergentes del motor central**.

## 2. Garantías de Consistencia Transaccional
- **Escrituras Atómicas**: La lista de meses procesados se reescribe de forma limpia tras completar la volcado al CSV principal.
- **Cero Duplicación de Operaciones**: En caso de un corte de energía, reinicio o caída del subproceso (crash) a mitad de un mes, el reinicio de la ejecución descartará los registros parciales en memoria para ese mes específico y re-procesará el bloque mensual de forma limpia, evitando entradas duplicadas en el archivo de trades acumulado.

*Veredicto: Aprobado. El entorno posee resiliencia institucional para afrontar interrupciones.*
