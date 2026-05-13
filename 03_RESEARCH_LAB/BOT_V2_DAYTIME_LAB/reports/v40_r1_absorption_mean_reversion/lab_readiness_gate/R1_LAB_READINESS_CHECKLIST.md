# MATRIZ DE AUDITORÍA PRE-EJECUCIÓN (LAB READINESS CHECKLIST)

## 1. Verificación de Inmutabilidad y Bloqueos
- [x] **Motor Causal Sancionado**: Reporte de Lockdown finalizado en estado `ENGINE_CORE_HARDENED_READY`.
- [x] **Verificación Criptográfica in situ**: `ENGINE_CORE_VERIFY.py` ejecutado en caliente arrojando `ENGINE_CORE_OK` (Paridad estricta de 72 archivos).
- [x] **Bypass Clausurado**: Licencia de excepción activa trasladada a la bóveda histórica para restablecer la postura incondicional de la barrera Git pre-commit.

## 2. Auditoría de Fuentes y Dependencias de R1
- [x] **Congelamiento de Hashes**: Cómputo SHA256 de archivos críticos de la estrategia e interfaces del motor (`R1_RUNNER_HASH_FREEZE.md`).
- [x] **Configuraciones Certificadas**: Espacio de búsqueda acotado a EURUSD, horario institucional y límite de frecuencia (`R1_CONFIG_FREEZE.md`).
- [x] **Integridad de Datos y Calendarios**: Disponibilidad del dataset de noticias *AM Fortress v3* cubriendo de forma estricta desde 2020 hasta 2026 sin gaps anómalos (`R1_DATA_NEWS_PRECHECK.md`).

## 3. Controles Operativos e Higiene Causal
- [x] **Frecuencia Máxima**: Blindaje en código del límite de 3 operaciones diarias con resolución puramente causal ante exceso de señales (`R1_FREQUENCY_PRECHECK.md`).
- [x] **Cierre de Fin de Mes (EOM)**: Desacople contable absoluto entre salidas forzadas por horario intradía y truncamientos de simulación a fin de mes (`R1_EOM_PRECHECK.md`).
- [x] **Resiliencia (Checkpoints)**: Escritura atómica de meses procesados y aislamiento de artefactos previos contaminados (`R1_CHECKPOINT_RESUME_PRECHECK.md`).

## 4. Presupuesto y Muerte Rápida
- [x] **Presupuesto de Recursos**: Proyección de CPU, RAM y almacenamiento requeridos para los 76 meses (`R1_RESOURCE_BUDGET.md`).
- [x] **Políticas de Parada Temprana**: Umbrales definidos para gatillar un cese inmediato de simulación ante degradación o violación de restricciones (`R1_EARLY_STOP_POLICY.md`).

## 5. Validación en Caliente
- [x] **Clean Preflight Final**: Ejecución de 1 mes desde cero superada exitosamente sin generar bloqueos en la carga causal (`R1_CLEAN_PREFLIGHT_FINAL.md`).
