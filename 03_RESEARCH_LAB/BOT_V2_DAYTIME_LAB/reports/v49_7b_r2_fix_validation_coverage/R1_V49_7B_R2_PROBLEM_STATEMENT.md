# R1 V49.7B-R2 ?" PROBLEM STATEMENT

**Sntoma**: El ranking de V49.7B muestra `N_val = 0` para todas las configuraciones, a pesar de que el proceso se complet.

**Causa**: El motor `UnifiedV7Engine` utiliza por defecto `test_start_year=2023`. Al no pasar este parǭmetro explcitamente en el runner anterior, el guardiǭn de seguridad (`TestLeakageGuard`) bloque preventivamente todos los meses de 2023 y 2024 como si fueran parte del conjunto de prueba restringido (TEST).

**Objetivo de R2**:
1. Modificar el runner externo para inyectar `test_start_year=2025`.
2. Verificar que el guardiǭn permita 2023 y 2024.
3. Completar la corrida representativa con datos de validacin (Out-of-Sample) reales.
