# PLAN DE MIGRACIÓN DOCUMENTAL (MOVE PLAN) — RECONCILIACIÓN DE FRONTERAS

**Estado de Ejecución:** **PLANIFICADO / PENDIENTE (No Mover en Corrida Activa)**  

## Justificación del Movimiento
Hacer cumplir la regla de contorno aséptico trasladando los artefactos operativos creados por el Agente 1 desde la zona de gobernanza hacia su directorio operativo legítimo en el laboratorio de investigación.

## Detalle de Archivos a Reubicar

### Artefacto 1: Estado de Git Pre-Corrida
- **Origen:** `06_GOVERNANCE_AND_COMPLIANCE\architecture\manipulante3_htf_ltf_research\GIT_STATUS_BEFORE.txt`
- **Destino Propuesto:** `03_RESEARCH_LAB\BOT_V2_DAYTIME_LAB\reports\v38_manipulante3_htf_ltf\GIT_STATUS_BEFORE.txt`
- **Razón:** Volcado puramente operativo del estado del repositorio antes de un barrido de research.

### Artefacto 2: Manifiesto de Bloqueo
- **Origen:** `06_GOVERNANCE_AND_COMPLIANCE\architecture\manipulante3_htf_ltf_research\MANIPULANTE3_LOCKDOWN_STATUS.md`
- **Destino Propuesto:** `03_RESEARCH_LAB\BOT_V2_DAYTIME_LAB\reports\v38_manipulante3_htf_ltf\MANIPULANTE3_LOCKDOWN_STATUS.md`
- **Razón:** Declaración de parámetros de contorno de la tarea específica del Agente 1.

## Restricción de Seguridad
De acuerdo con el mandato de remediación, **no se borra ningún archivo** en este paso y el movimiento físico se pospone hasta confirmar la finalización o pausa segura de cualquier checkpoint en curso.
