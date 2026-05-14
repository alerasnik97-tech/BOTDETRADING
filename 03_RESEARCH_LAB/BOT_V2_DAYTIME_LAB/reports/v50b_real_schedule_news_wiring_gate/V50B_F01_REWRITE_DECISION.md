# V50B F01 REWRITE DECISION

**Estado**: **F01_REWRITE_REQUIRED_TO_NY_WINDOW**

## Motivo del Rechazo
La implementación actual de F01 genera señales a las 03:15 NY (Open de Londres). Esta ventana estǭ fuera de la poltica operativa del laboratorio (07:00 ?" 17:00 NY).

## Requerimiento de Reescritura
Para que F01 sea admitida en el Gauntlet V50B, su lgica debe evolucionar hacia:
- **Bias de Londres**: Identificar la direccin en la apertura de Londres (03:00).
- **Ejecucin NY**: Buscar entradas (Pullbacks o Continuaciones) a partir de las **07:00 NY**.

## Impacto
F01 queda **EXCLUIDA** del Micro-Probe actual y del Limited Gauntlet hasta que se presente una nueva definicin de familia aprobada por Governance.
