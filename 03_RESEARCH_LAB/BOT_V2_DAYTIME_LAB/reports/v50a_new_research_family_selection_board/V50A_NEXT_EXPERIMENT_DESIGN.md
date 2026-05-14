# V50A NEXT EXPERIMENT DESIGN

**Nombre de la Fase**: **V50B ?" Family Preflight Gauntlet**

## Objetivo
Evaluar las 4 familias del Shortlist (F01, F06, F08, F12) bajo una corrida representativa de 10 meses para detectar rǭpidamente si alguna posee edge robusto (TRAIN+VAL).

## Diseño Experimental
- **Data**: EURUSD (Marzo 2020 a Octubre 2024 representativo).
- **Aislamiento**: `test_start_year=2025` (TEST cerrado).
- **Configs**: ~100 por familia (Total ~400-500).
- **Gauntlet Gate**: 
  - PF_train >= 1.0
  - PF_val >= 1.15
  - N_train >= 30, N_val >= 20
  - Concentracin mensual < 50%
- **Costos**: Comisin 5.0/lot, Slippage 0.2 pips.

## Proceso
1. Implementar detectores para cada familia (v7_engine compatible).
2. Correr Preflight de 3 meses para validacin de lógica.
3. Lanzar Gauntlet de 10 meses.
4. Auditar resultados y seleccionar la familia ganadora para V50C (Full Scope).

**Advertencia**: Si ninguna familia pasa el Gauntlet, se volverǭ a la mesa de diseİo V50A. **NO se autoriza V50C ni V51 hasta que haya un ganador claro**.
