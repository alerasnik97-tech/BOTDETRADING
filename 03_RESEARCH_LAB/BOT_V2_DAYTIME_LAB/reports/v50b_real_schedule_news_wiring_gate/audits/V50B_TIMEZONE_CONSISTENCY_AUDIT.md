# V50B TIMEZONE CONSISTENCY AUDIT

**Objetivo**: Validar la interpretación de los timestamps de mercado.

## Hallazgos del Probe
| Timestamp UTC | NY Time | Allowed 08-11 | Allowed 07-17 | Status |
| --- | --- | --- | --- | --- |
| 2022-05-03 03:15:00 | 2022-05-02 23:15:00 EDT | False | False | REJECTED_BOTH |
| 2022-05-04 08:30:00 | 2022-05-04 04:30:00 EDT | False | False | REJECTED_BOTH |
| 2022-05-04 09:15:00 | 2022-05-04 05:15:00 EDT | False | False | REJECTED_BOTH |
| 2022-05-03 11:45:00 | 2022-05-03 07:45:00 EDT | False | True | VALID |
| 2023-01-02 08:50:00 | 2023-01-02 03:50:00 EST | False | False | REJECTED_BOTH |

## Conclusión Crítica
- **TIMEZONE MISMATCH CONFIRMED**: Los detectores están operando sobre timestamps UTC pero con lógica de horas NY. 
- Una señal generada a las **08:30 UTC** corresponde a las **04:30 NY**, por lo que es correctamente rechazada por el motor (fuera de sesión).
- Para operar la apertura de NY (08:00 NY), el detector debe buscar señales a partir de las **12:00 UTC** (aprox).

## Acción Requerida
- El runner y los detectores deben ser agnósticos a la zona horaria o realizar la conversión a NY explícitamente antes de evaluar condiciones horarias.
- El motor (`UnifiedV7Engine`) ya realiza la conversión a NY internamente para el `ScheduleGuard`.

**Veredicto**: SCHEDULE_TIMEZONE_MISMATCH.
