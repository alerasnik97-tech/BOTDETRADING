# V50B FAMILY PREFLIGHT GAUNTLET ?" LOCKDOWN

**Fecha**: 2026-05-14
**Objetivo**: Implementacin y evaluacin de 4 nuevas familias de investigacin (F01, F06, F08, F12).

## Restricciones Absolutas
- **NO TEST**: `test_start_year=2025`. Bloqueo fsco de 2025-2026.
- **NO CORE DRIFT**: Prohibido modificar `src/v7_engine` o `src/v6_utils`.
- **NO R1 RECOVERY**: Prohibido intentar "arreglar" R1 bajo otros nombres.
- **NO DATA MUTATION**: Los datos del Vault permanecen en modo lectura.
- **NO V50 FINAL**: Esta fase solo autoriza preflight representativo.

## Compromiso
Toda familia que no supere el gate combinado de TRAIN (PF >= 1.0) y VAL (PF >= 1.15) serǭ descartada rǭpidamente. El laboratorio no tolera el overfit ni la concentracin temporal.
