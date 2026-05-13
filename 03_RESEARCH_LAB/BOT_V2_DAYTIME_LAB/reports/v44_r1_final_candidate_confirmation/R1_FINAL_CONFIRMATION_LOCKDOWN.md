# MANIFIESTO DE BLOQUEO ARQUITECTÓNICO — R1 FINAL CANDIDATE CONFIRMATION

## 1. Perímetro de Inmutabilidad Funcional (Core Lockdown)
Se sanciona formalmente la apertura de la fase **V44** (*Final Candidate Confirmation*), bajo un candado absoluto de no-modificación:
- **Cero Deriva en el Core**: Se prohíbe alterar un solo byte de `src/v7_engine` y `src/v6_utils`. 
- **Cero Deriva en la Estrategia**: Se prohíbe modificar cualquier parámetro de la candidata `cfg_r1_factory_opt_001`. El objetivo es confirmar, no optimizar.

## 2. Restricciones Físicas de la Corrida
- **Instrumento Único**: `EURUSD`.
- **Deducciones Obligatorias**: Slippage de 0.2 pips oficial aplicado en cada transacción.
- **Veto de Selección**: Se prohíbe el uso de la partición TEST para cualquier ajuste posterior.
