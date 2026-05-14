# V50B REAL SCHEDULE GUARD AUDIT

**Objetivo**: Entender el motivo del bloqueo de señales reales y proponer la configuración correcta.

## Hallazgos del Motor (`schedule_guard.py`)
1. **Ventana por Defecto**: 08:00 ?" 11:00 NY.
2. **Timezone**: Utiliza `America/New_York` para evaluar la hora de entrada (`_to_ny` convierte UTC a NY).
3. **Motivos de Bloqueo**:
   - **F01 (03:15 NY)**: Bloqueado por estar fuera de 08:00-11:00.
   - **F06 (11:45 NY)**: Bloqueado por estar fuera de 08:00-11:00.
4. **Timezone Consistency**: El motor espera `timestamp_utc`. Si el runner entrega UTC naive, el motor lo localiza a UTC y luego a NY. Esto es correcto y seguro.

## Diagnóstico
El bloqueo es **VALID** según la configuración actual del motor, pero **INSUFFICIENT** para las familias de investigación F01 y F06 que operan en ventanas extendidas.

## Solución Propuesta (Sin tocar Core)
- Configurar el `ScheduleGuard` desde el runner al instanciar el motor.
- Usar la ventana máxima operativa: **07:00 ?" 17:00 NY**.
- Cualquier señal fuera de este rango debe ser rechazada por el motor.

**Veredicto**: SCHEDULE_RUNNER_CONFIG_MISSING.
