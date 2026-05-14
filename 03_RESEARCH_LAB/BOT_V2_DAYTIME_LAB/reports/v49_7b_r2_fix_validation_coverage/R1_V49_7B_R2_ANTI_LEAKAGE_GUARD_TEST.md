# R1 V49.7B-R2 ?" ANTI-LEAKAGE GUARD TEST

**Objetivo**: Verificar que el cambio a `test_start_year=2025` permite VAL y bloquea TEST.

## Resultados SintǸticos
- **2022 accepted**: YES
- **2023 accepted**: YES
- **2024 accepted**: YES
- **2025 blocked**: YES
- **test_start_year used**: 2025

**Conclusin**: El guardiǭn funciona correctamente. La fase `VAL` (2023-2024) estǭ ahora desbloqueada mientras que el conjunto de prueba fsco (2025-2026) permanece blindado.
