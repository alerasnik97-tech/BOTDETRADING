# DIRECTIVA DE CONGELAMIENTO PRE-EJECUCIÓN (LAB READINESS LOCKDOWN)

## 1. Declaración de Cuarentena
Se establece un **Bloqueo Operacional Estricto** sobre el código fuente de la estrategia R1 (`src/R1/`) y su orquestador (`run_r1_micro_probe.py`). 
- A partir de este momento, queda estrictamente prohibida la introducción de ajustes paramétricos, optimizaciones de código o alteraciones en la lógica de filtrado de niveles.
- El objetivo exclusivo de esta fase es auditar exhaustivamente la higiene de dependencias, la correctitud de las interfaces con el motor canónico inmutable y la capacidad de reanudación (checkpointing) del orquestador antes de destinar tiempo de CPU al barrido completo.

## 2. Restricciones Activas durante la Verificación
- **Prohibición de Cómputo Masivo**: El preflight final se acota incondicionalmente a un máximo de 1-2 meses. La ejecución del barrido completo de 76 meses queda bloqueada hasta la sanción del manifiesto de decisión final.
- **Prohibición de Exportación**: No se generarán paquetes zip para ChatGPT ni sincronizaciones con ramas de nube hasta que la auditoría local demuestre cero desviaciones en las frecuencias operativas y de truncamiento.
